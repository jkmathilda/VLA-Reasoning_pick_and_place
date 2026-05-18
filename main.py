"""
Usage:
  python main.py --checkpoint path/to/checkpoint
  python main.py --checkpoint path/to/checkpoint --timeout 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from policy import VLAPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the SmolVLA checkpoint directory.")
    parser.add_argument("--timeout",    type=float, default=18.0,
                        help="Max execution time in seconds (default 18).")
    args = parser.parse_args()

    policy = VLAPolicy.load(args.checkpoint)
    policy.run(timeout=args.timeout)


if __name__ == "__main__":
    main()
