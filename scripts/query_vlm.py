"""
Thin wrapper around src/vlm.py SceneVLM, used by the async-inference
robot client (scripts/robot_client_with_vlm.py).

Only handles VLM loading and inference — robot connection, homing, and
frame capture all live in the inference script.
"""
from pathlib import Path
import sys

from PIL import Image

# Allow `from vlm import SceneVLM` (src/vlm.py).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def query_vlm(image: Image.Image, task_spec: str, logger=None) -> str:
    """Run SmolVLM on an overview frame to resolve task_spec into a concrete task."""
    from vlm import SceneVLM

    log = logger.info if logger is not None else print
    log(f"[vlm] Querying SmolVLM with task spec: {task_spec!r}")
    vlm = SceneVLM()
    task = vlm.query(image, task_spec=task_spec)
    log(f"[vlm] Resolved task: {task!r}")
    return task
