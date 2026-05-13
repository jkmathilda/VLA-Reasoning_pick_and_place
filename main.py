"""
Usage:
  python main.py --task "Put the banana in the blue bowl."
  python main.py --task "Put the banana into the 2nd bowl from the left."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from policy import VLAPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       type=str,   required=True,
                        help="Natural language task instruction")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/policy.pt")
    parser.add_argument("--timeout",    type=float, default=18.0,
                        help="Max execution time in seconds (default 18)")
    args = parser.parse_args()

    policy = VLAPolicy.load(args.checkpoint)
    policy.run(task=args.task, timeout=args.timeout)


if __name__ == "__main__":
    main()
