"""
VLM task-resolution helpers, intended to be imported by the async-inference
robot client (see scripts/robot_client_with_vlm.py).

Given a connected lerobot Robot with a camera, this module can move the arm to
a fixed overview pose, capture a frame, and ask SmolVLM to resolve a high-level
task spec ("place the banana in the leftmost bowl") into the concrete task
string the policy was trained with.
"""
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

# Allow `from vlm import SceneVLM` (src/vlm.py).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mirror of HOME_JOINTS in src/policy.py — keep in sync.
HOME_JOINTS = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -35.0,
    "elbow_flex.pos":    -8.0,
    "wrist_flex.pos":    100.0,
    "wrist_roll.pos":    0.0,
    "gripper.pos":       0.0,
}
HOME_MOVE_TIME_S = 3.0
HOME_FPS = 30


def move_to_home(robot, move_time_s: float = HOME_MOVE_TIME_S, fps: int = HOME_FPS) -> None:
    deadline = time.perf_counter() + move_time_s
    while time.perf_counter() < deadline:
        robot.send_action(HOME_JOINTS)
        time.sleep(1.0 / fps)


def capture_overview_frame(robot, camera_name: str) -> Image.Image:
    obs = robot.get_observation()
    frame = obs[camera_name]
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    return Image.fromarray(frame)


def resolve_task_via_vlm(robot, camera_name: str, task_spec: str, logger=None) -> str:
    """Move to home, snap an overview frame, and resolve task_spec via SmolVLM.

    Returns the concrete task string to feed the policy.
    """
    from vlm import SceneVLM

    log = logger.info if logger is not None else print

    log(f"[vlm] Moving to home position ({HOME_MOVE_TIME_S:.1f}s)...")
    move_to_home(robot)

    log("[vlm] Capturing overview frame...")
    image = capture_overview_frame(robot, camera_name)

    log(f"[vlm] Querying SmolVLM with task spec: {task_spec!r}")
    vlm = SceneVLM()
    task = vlm.query(image, task_spec=task_spec)
    log(f"[vlm] Resolved task: {task!r}")
    return task
