import sys
import types as _types

# lerobot/policies/__init__.py imports groot, which has a Python 3.14 dataclass bug.
# Stub it out before importing anything from lerobot.policies.
for _mod in ["lerobot.policies.groot", "lerobot.policies.groot.configuration_groot"]:
    if _mod not in sys.modules:
        _s = _types.ModuleType(_mod)
        _s.GrootConfig = None
        sys.modules[_mod] = _s

import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

# Matches "Put the <object> in[to] [the] <color> bowl." (case-insensitive).
# Anything that doesn't match is treated as corrupted and mapped to None (dropped).
MODEL_ID = "lerobot/smolvla_base"

# Image key used by the pretrained SmolVLA checkpoint
IMG_KEY = "observation.images.camera1"

# Robot hardware constants (match record.py / teleop.py)
ROBOT_PORT = "/dev/tty.usbmodem5B141136531"
CAMERA_DEFAULT_W, CAMERA_DEFAULT_H = 1920, 1080
CAMERA_CAP_W, CAMERA_CAP_H = 640, 480


class VLAPolicy:
    """
    Wrapper around SmolVLAPolicy (lerobot/smolvla_base).

    SmolVLA has a built-in action expert (flow matching) — no separate MLP needed.
    The VLM backbone (SmolVLM2-500M) is frozen; only the action expert is trained.
    Output is a queue of n_action_steps actions; call reset() between episodes.
    """

    def __init__(self, model_id: str = MODEL_ID, dataset_stats: dict | None = None,
                 device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = _SmolVLAPolicy.from_pretrained(model_id).to(self.device).eval()
        self._dataset_stats = dataset_stats
        self.preprocess, self.postprocess = make_smolvla_pre_post_processors(
            self.model.config, dataset_stats=dataset_stats
        )

    def reset(self):
        """Call before each new episode to flush the internal action queue."""
        self.model.reset()

    def select_action(self, pil_image: Image.Image, task: str, state_np) -> np.ndarray:
        """Single inference step — returns action [6] as numpy array."""
        frame = torch.tensor(np.array(pil_image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs = {
            IMG_KEY: frame,
            "observation.state": torch.tensor(list(state_np), dtype=torch.float32),
            "task": task,
        }
        batch = self.preprocess(obs)
        with torch.inference_mode():
            action = self.model.select_action(batch)
        return self.postprocess(action).numpy()

    def run(self, task: str, timeout: float = 18.0):
        """Inference loop: capture wrist-camera frame → predict → send to SO101."""
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        camera_idx = None
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                break
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == CAMERA_DEFAULT_W and h == CAMERA_DEFAULT_H:
                camera_idx = i
                break
        if camera_idx is None:
            raise RuntimeError(f"Wrist camera ({CAMERA_DEFAULT_W}×{CAMERA_DEFAULT_H}) not found")

        robot = SO101Follower(SO101FollowerConfig(
            port=ROBOT_PORT,
            id="eth_rl_so101_follower",
            cameras={"wrist_camera": OpenCVCameraConfig(
                index_or_path=camera_idx,
                width=CAMERA_CAP_W, height=CAMERA_CAP_H, fps=30,
            )},
        ))
        robot.connect()
        self.reset()

        start = time.time()
        try:
            while time.time() - start < timeout:
                obs      = robot.capture_observation()
                frame_np = obs["observation.images.wrist_camera"]
                state_np = obs["observation.state"]

                frame_pil = Image.fromarray(frame_np) if isinstance(frame_np, np.ndarray) else frame_np
                action_np = self.select_action(frame_pil, task, state_np)

                robot.send_action({"action": action_np})
        finally:
            robot.disconnect()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        if self._dataset_stats is not None:
            serializable = {
                k: {kk: vv.tolist() for kk, vv in v.items()}
                for k, v in self._dataset_stats.items()
            }
            with open(os.path.join(path, "dataset_stats.json"), "w") as f:
                json.dump(serializable, f)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "VLAPolicy":
        stats_path = os.path.join(path, "dataset_stats.json")
        dataset_stats = None
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                raw = json.load(f)
            dataset_stats = {
                k: {kk: torch.tensor(vv) for kk, vv in v.items()}
                for k, v in raw.items()
            }
        wrapper                = cls.__new__(cls)
        wrapper.device         = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wrapper._dataset_stats = dataset_stats
        wrapper.model          = _SmolVLAPolicy.from_pretrained(path).to(wrapper.device).eval()
        wrapper.preprocess, wrapper.postprocess = make_smolvla_pre_post_processors(
            wrapper.model.config, dataset_stats=dataset_stats
        )
        return wrapper
