"""Standalone VLM smoke test for eval 2 (banana/bowl) and eval 3 (coke/celebrity).

Grabs a single frame from a webcam (or loads an image from disk), runs the
selected SceneVLM, and prints both the phase-1 reasoning and the final
canonical task sentence. Useful for iterating on the VLM prompts without
spinning up the full robot client / policy server.

Examples:
    python scripts/test_vlm.py --eval 2 --task "Put the banana in the blue bowl"
    python scripts/test_vlm.py --eval 3 --task "Place the coke on Yann LeCun"
    python scripts/test_vlm.py --eval 3 --task "Place the coke on Obama" \
        --image /tmp/snapshot.png

The phase-1 reasoning is already printed to stderr by SceneVLM itself; this
script also prints it to stdout at the end alongside the final answer for
easy copy/paste.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stderr
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _capture_from_camera(index: int) -> Image.Image:
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    # Drop a few frames so the auto-exposure has time to settle.
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from camera index {index}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _load_vlm(eval_id: int):
    if eval_id == 2:
        from vlm import SceneVLM
    elif eval_id == 3:
        from vlm_eval3 import SceneVLM
    else:
        raise ValueError(f"--eval must be 2 or 3, got {eval_id}")
    return SceneVLM()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=int, required=True, choices=[2, 3],
                        help="Which VLM to test: 2=banana/bowl, 3=coke/celebrity.")
    parser.add_argument("--task", type=str, required=True,
                        help="Task spec to send to the VLM (e.g. 'Place the coke on Obama').")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional image file to use instead of capturing from the camera.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for OpenCV (ignored if --image is given).")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save the captured image to.")
    args = parser.parse_args()

    if args.image:
        print(f">>> Loading image from {args.image}")
        image = Image.open(args.image).convert("RGB")
    else:
        print(f">>> Capturing frame from camera index {args.camera}")
        image = _capture_from_camera(args.camera)

    if args.save:
        image.save(args.save)
        print(f">>> Saved input image to {args.save}")

    print(f">>> Loading eval-{args.eval} SceneVLM...")
    vlm = _load_vlm(args.eval)

    print(f">>> Querying VLM with task spec: {args.task!r}")
    captured = io.StringIO()
    with redirect_stderr(captured):
        result = vlm.query(image, task_spec=args.task)
    vlm_log = captured.getvalue()
    sys.stderr.write(vlm_log)

    print()
    print("=" * 72)
    print("VLM internal log (stderr):")
    print("=" * 72)
    print(vlm_log.rstrip())
    print()
    print("=" * 72)
    print(f"Task spec:    {args.task!r}")
    print(f"Final task:   {result!r}")
    print("=" * 72)


if __name__ == "__main__":
    main()
