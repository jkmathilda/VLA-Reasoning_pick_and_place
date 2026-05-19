"""
VLAPolicy: SmolVLA-based pick-and-place execution with VLM task generation.

Flow:
  1. Robot connects; optionally moves to HOME_JOINTS (top-view overview position).
  2. Camera captures the workspace image and passes it to SmolVLM.
  3. SmolVLM returns a single-sentence task instruction.
  4. SmolVLA executes the task for `timeout` seconds.

Usage:
  policy = VLAPolicy.load("path/to/checkpoint")
  policy.run(timeout=18.0)
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from PIL import Image

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

from vlm import SceneVLM

ROBOT_PORT = "/dev/tty.usbmodem5B141136531"
FPS = 30
CAMERA_NAME = "camera1"
CAMERA_DEFAULT_W, CAMERA_DEFAULT_H = 1920, 1080
CAMERA_CAPTURE_W, CAMERA_CAPTURE_H = 640, 480

# Joint positions (degrees) that put the wrist camera roughly top-down over the workspace.
# Tune these for your physical setup by teleoperating to a good overview pose.
HOME_JOINTS: dict[str, float] = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -60.0,
    "elbow_flex.pos": 80.0,
    "wrist_flex.pos": 80.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}
HOME_MOVE_TIME_S = 3.0


def _find_camera_index(target_w: int, target_h: int) -> int:
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == target_w and h == target_h:
            return i
    raise RuntimeError(f"No camera with default resolution {target_w}x{target_h} found.")


class VLAPolicy:
    def __init__(
        self,
        policy: SmolVLAPolicy,
        preprocessor,
        postprocessor,
        robot: SO101Follower,
        vlm: SceneVLM,
        device: str,
    ):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot
        self._vlm = vlm
        self._device = device

        lerobot_features = hw_to_dataset_features(
            robot.observation_features, "observation", use_video=False
        )
        self._lerobot_features = lerobot_features
        self._action_names: list[str] = hw_to_dataset_features(
            robot.action_features, "action"
        )["action"]["names"]

    @classmethod
    def load(cls, checkpoint: str, device: str | None = None) -> "VLAPolicy":
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        cam_idx = _find_camera_index(CAMERA_DEFAULT_W, CAMERA_DEFAULT_H)
        camera_configs = {
            CAMERA_NAME: OpenCVCameraConfig(
                index_or_path=cam_idx,
                width=CAMERA_CAPTURE_W,
                height=CAMERA_CAPTURE_H,
                fps=FPS,
            )
        }
        robot = SO101Follower(
            SO101FollowerConfig(
                port=ROBOT_PORT,
                id="eth_rl_so101_follower",
                cameras=camera_configs,
            )
        )

        print(f"[Policy] Loading SmolVLA from {checkpoint} on {device}...")
        policy = SmolVLAPolicy.from_pretrained(checkpoint)
        policy.to(device)
        policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        vlm = SceneVLM()
        return cls(policy, preprocessor, postprocessor, robot, vlm, device)

    def _move_to_home(self) -> None:
        print("[Policy] Moving to overview (home) position...")
        deadline = time.perf_counter() + HOME_MOVE_TIME_S
        while time.perf_counter() < deadline:
            self._robot.send_action(HOME_JOINTS)
            time.sleep(1.0 / FPS)
        print("[Policy] Overview position reached.")

    def _capture_pil_image(self) -> Image.Image:
        obs = self._robot.get_observation()
        frame: np.ndarray = obs[CAMERA_NAME]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        return Image.fromarray(frame)

    def _make_batch(self, raw_obs: dict, task: str) -> dict:
        frame = build_dataset_frame(self._lerobot_features, raw_obs, "observation")
        batch: dict = {"task": task}
        for k, v in frame.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 3:
                    # (H, W, C) uint8 → (C, H, W) float32 [0, 1]
                    batch[k] = torch.from_numpy(v).permute(2, 0, 1).float() / 255.0
                else:
                    batch[k] = torch.from_numpy(v.astype(np.float32))
            else:
                batch[k] = v
        return batch

    def run(self, timeout: float = 18.0) -> None:
        self._robot.connect()
        self._policy.reset()

        try:
            self._move_to_home()

            print("[Policy] Capturing overview image and querying VLM...")
            image = self._capture_pil_image()
            task = self._vlm.query(image)
            print(f"[Policy] VLM task: {task!r}")

            print(f"[Policy] Executing for {timeout:.1f}s at {FPS} Hz...")
            start = time.perf_counter()
            step = 0
            while time.perf_counter() - start < timeout:
                t0 = time.perf_counter()

                raw_obs = self._robot.get_observation()
                batch = self._make_batch(raw_obs, task)
                batch = self._preprocessor(batch)

                action = self._policy.select_action(batch)  # (1, action_dim)
                action = self._postprocessor(action)        # (1, action_dim) denormalized

                action_vals = action.squeeze(0).tolist()
                action_dict = dict(zip(self._action_names, action_vals))
                self._robot.send_action(action_dict)

                step += 1
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, 1.0 / FPS - elapsed))

            print(f"[Policy] Done. {step} steps over {timeout:.1f}s.")

        except KeyboardInterrupt:
            print("[Policy] Interrupted.")
        finally:
            if self._robot.is_connected:
                self._robot.disconnect()
