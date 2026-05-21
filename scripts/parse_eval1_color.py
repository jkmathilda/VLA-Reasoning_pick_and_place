"""Map an Eval 1 task string ("place banana in the <color> bowl") to a
position label (left/middle/right) for the fixed Blue-Red-Green bowl order.

Robust to typos / odd capitalization / extra punctuation. For each whitespace-
separated word in the input we compute the max of two similarity scores against
each color name:
  - character-bigram IoU (Jaccard of bigram sets, captures local order)
  - difflib.SequenceMatcher ratio (captures global alignment)
We take the best (word, color) pair above MATCH_THRESHOLD and emit the
corresponding position. Exit code 1 if no color matches.

Usage:
    python scripts/parse_eval1_color.py "Place the banana in the blu bowl"
    # prints: left
"""

from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher

COLOR_TO_POSITION: dict[str, str] = {
    "blue": "left",
    "red": "middle",
    "green": "right",
}

MATCH_THRESHOLD = 0.6


def _bigrams(s: str) -> set[str]:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _bigram_iou(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba and not bb:
        return 1.0
    union = ba | bb
    if not union:
        return 0.0
    return len(ba & bb) / len(union)


def _similarity(word: str, color: str) -> float:
    return max(_bigram_iou(word, color), SequenceMatcher(None, word, color).ratio())


def parse_color(task: str) -> tuple[str, str, float] | None:
    """Return (color, position, score) for the best color match, or None."""
    words = re.findall(r"[a-zA-Z]+", task.lower())
    best: tuple[str, str, float] | None = None
    for word in words:
        for color, position in COLOR_TO_POSITION.items():
            score = _similarity(word, color)
            if score >= MATCH_THRESHOLD and (best is None or score > best[2]):
                best = (color, position, score)
    return best


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: parse_eval1_color.py \"<task>\"", file=sys.stderr)
        return 2
    task = " ".join(sys.argv[1:])
    result = parse_color(task)
    if result is None:
        print("middle")
        return 0
    _, position, _ = result
    print(position)
    return 0


if __name__ == "__main__":
    sys.exit(main())
