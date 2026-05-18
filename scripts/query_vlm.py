#!/usr/bin/env python3
"""
Capture one overview frame from the robot camera, query SmolVLM, and print
the task string to stdout (nothing else). All other logging goes to stderr.

Usage (standalone):
    python scripts/query_vlm.py --task "Place the banana in the leftmost bowl"

Usage (from bash — capture task into a variable):
    TASK=$(python scripts/query_vlm.py --task "...")
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from PIL import Image

# Must match the camera used during recording.
CAMERA_DEFAULT_W = 640
CAMERA_DEFAULT_H = 480

# Mirror of HOME_JOINTS in src/policy.py — keep in sync.
HOME_JOINTS = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -50.0,
    "elbow_flex.pos":    15.0,
    "wrist_flex.pos":    80.0,
    "wrist_roll.pos":    0.0,
    "gripper.pos":       0.0,
}
HOME_MOVE_TIME_S = 3.0
FPS = 30


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_camera(target_w: int, target_h: int) -> int:
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == target_w and h == target_h:
            return i
    raise RuntimeError(f"No camera found with default resolution {target_w}x{target_h}")


def capture_frame(idx: int) -> Image.Image:
    cap = cv2.VideoCapture(idx)
    for _ in range(5):  # warm up — discard stale frames
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame from camera")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def move_to_home_and_capture() -> Image.Image:
    """Connect robot, move to overview position, capture frame, disconnect."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    ROBOT_PORT = "/dev/ttyACM0"
    CAMERA_NAME = "camera1"

    cam_idx = find_camera(CAMERA_DEFAULT_W, CAMERA_DEFAULT_H)
    robot = SO101Follower(SO101FollowerConfig(
        port=ROBOT_PORT,
        id="eth_rl_so101_follower",
        cameras={
            CAMERA_NAME: OpenCVCameraConfig(
                index_or_path=cam_idx, width=640, height=480, fps=FPS,
            )
        },
    ))
    robot.connect()
    try:
        eprint("[query_vlm] Moving to home position...")
        deadline = time.perf_counter() + HOME_MOVE_TIME_S
        while time.perf_counter() < deadline:
            robot.send_action(HOME_JOINTS)
            time.sleep(1.0 / FPS)

        eprint("[query_vlm] Capturing overview frame...")
        obs = robot.get_observation()
        frame = obs[CAMERA_NAME]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        return Image.fromarray(frame)
    finally:
        robot.disconnect()


def main():
    from vlm import SceneVLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="High-level task specification to resolve into a bowl position.")
    parser.add_argument("--no-home", action="store_true", help="Skip moving robot to home position before capture.")
    args = parser.parse_args()

    if not args.no_home:
        image = move_to_home_and_capture()
    else:
        eprint("[query_vlm] Skipping home move — capturing directly from camera...")
        idx = find_camera(CAMERA_DEFAULT_W, CAMERA_DEFAULT_H)
        image = capture_frame(idx)

    eprint(f"[query_vlm] Querying VLM with task spec: {args.task!r}")
    vlm = SceneVLM()
    task = vlm.query(image, task_spec=args.task)

    # Only the task goes to stdout — everything else used stderr.
    print(task)


if __name__ == "__main__":
    main()
