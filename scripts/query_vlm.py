"""
Thin wrappers around src/vlm.py SceneVLM, used by the async-inference
robot client (scripts/robot_client_with_vlm.py).

Split into `load_vlm()` and `query_vlm(vlm, image, task_spec)` so the model
can be loaded BEFORE the robot starts moving — evaluation timing begins at
first motion, so loading during the gRPC handshake (which doesn't move the
arm) keeps load time off the clock.
"""
from pathlib import Path
import sys

from PIL import Image

# Allow `from vlm import SceneVLM` (src/vlm.py).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_vlm(logger=None):
    """Load SmolVLM. Call this before homing so model load doesn't count toward eval time."""
    from vlm import SceneVLM

    log = logger.info if logger is not None else print
    log("[vlm] Loading SceneVLM (this can take a few seconds)...")
    vlm = SceneVLM()
    log("[vlm] SceneVLM loaded.")
    return vlm


def query_vlm(vlm, image: Image.Image, task_spec: str, logger=None) -> str:
    """Run SmolVLM on an overview frame to resolve task_spec into a concrete task."""
    log = logger.info if logger is not None else print
    log(f"[vlm] Querying SmolVLM with task spec: {task_spec!r}")
    task = vlm.query(image, task_spec=task_spec)
    log(f"[vlm] Resolved task: {task!r}")
    return task
