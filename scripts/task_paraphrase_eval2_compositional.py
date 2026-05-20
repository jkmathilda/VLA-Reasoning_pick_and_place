"""Compositional task paraphrasing for Eval 2.

Recorded task strings encode the physical bowl arrangement and target position as
"<perm> : <Position>", e.g. "GBR : Middle" means the bowls left-to-right from the
robot perspective are Green, Blue, Red, and the banana goes into the Middle bowl
(the Blue one in this case).

We do NOT collapse to left/middle/right. Instead, we generate compositional
prompts that require reasoning over the bowl arrangement, mirroring the eval
rubric: ordinal references, relative-to-color references, between-references,
and negation references. The direct-color form is also included as a sanity
anchor.
"""

from __future__ import annotations

import random
import re
from itertools import permutations

COLOR_CODE_TO_NAME: dict[str, str] = {
    "R": "red",
    "G": "green",
    "B": "blue",
}

POSITION_NAME_TO_INDEX: dict[str, int] = {
    "left": 0,
    "middle": 1,
    "right": 2,
}

# Format-1 recordings ("Place the banana into the <color> bowl") used a fixed
# physical bowl order of Blue, Red, Green from left to right.
FORMAT1_BOWLS: list[str] = ["blue", "red", "green"]

ORDINAL_FROM_LEFT = ["1st", "2nd", "3rd"]
ORDINAL_FROM_RIGHT = ["3rd", "2nd", "1st"]
ORDINAL_WORDS_FROM_LEFT = ["first", "second", "third"]
ORDINAL_WORDS_FROM_RIGHT = ["third", "second", "first"]

PERSPECTIVE_SUFFIXES = [
    " from the robot perspective",
    " from the robot's perspective",
    "",
]


_ESC_PREFIX_RE = re.compile(r"^(?:\x1b?\[[A-D])+")


def parse_task(task: str) -> tuple[list[str], int] | None:
    """Parse a task string into (bowl_colors_left_to_right, target_index).

    Supports two recorded formats:
      Format 2: "<perm> : <Position>" e.g. "GBR : Middle".
      Format 1: "Place the banana into the <color> bowl" — assumes the
                Format-1 fixed bowl order Blue, Red, Green (left to right).

    Returns None if the task does not match either format.
    """
    task = _ESC_PREFIX_RE.sub("", task).strip()
    if ":" in task:
        perm_str, pos_str = task.split(":", 1)
        perm_str = perm_str.strip().upper()
        pos_str = pos_str.strip().lower()

        if (
            len(perm_str) == 3
            and all(c in COLOR_CODE_TO_NAME for c in perm_str)
            and len(set(perm_str)) == 3
            and pos_str in POSITION_NAME_TO_INDEX
        ):
            bowls = [COLOR_CODE_TO_NAME[c] for c in perm_str]
            return bowls, POSITION_NAME_TO_INDEX[pos_str]

    lowered = task.lower()
    for color in FORMAT1_BOWLS:
        if color in lowered:
            return list(FORMAT1_BOWLS), FORMAT1_BOWLS.index(color)

    return None


def _direct_color(bowls: list[str], target_idx: int) -> list[str]:
    color = bowls[target_idx]
    return [
        f"Put the banana in the {color} bowl.",
        f"Put the banana into the {color} bowl.",
        f"Place the banana in the {color} bowl.",
        f"Move the banana to the {color} bowl.",
        f"Drop the banana in the {color} bowl.",
    ]


def _ordinal_from_side(bowls: list[str], target_idx: int) -> list[str]:
    from_left_num = ORDINAL_FROM_LEFT[target_idx]
    from_left_word = ORDINAL_WORDS_FROM_LEFT[target_idx]
    from_right_num = ORDINAL_FROM_RIGHT[target_idx]
    from_right_word = ORDINAL_WORDS_FROM_RIGHT[target_idx]

    prompts: list[str] = []
    for persp in PERSPECTIVE_SUFFIXES:
        prompts.extend(
            [
                f"Put the banana into the {from_left_num} bowl from the left{persp}.",
                f"Put the banana in the {from_left_word} bowl from the left{persp}.",
                f"Place the banana in the {from_left_num} bowl from the left{persp}.",
                f"Put the banana into the {from_right_num} bowl from the right{persp}.",
                f"Put the banana in the {from_right_word} bowl from the right{persp}.",
                f"Place the banana in the {from_right_num} bowl from the right{persp}.",
            ]
        )
    return prompts


def _relative_to_color(bowls: list[str], target_idx: int) -> list[str]:
    prompts: list[str] = []
    if target_idx - 1 >= 0:
        ref = bowls[target_idx - 1]
        for persp in PERSPECTIVE_SUFFIXES:
            prompts.extend(
                [
                    f"Put the banana into the bowl on the right of the {ref} bowl{persp}.",
                    f"Put the banana in the bowl to the right of the {ref} bowl{persp}.",
                    f"Place the banana in the bowl right of the {ref} bowl{persp}.",
                ]
            )
    if target_idx + 1 <= 2:
        ref = bowls[target_idx + 1]
        for persp in PERSPECTIVE_SUFFIXES:
            prompts.extend(
                [
                    f"Put the banana into the bowl on the left of the {ref} bowl{persp}.",
                    f"Put the banana in the bowl to the left of the {ref} bowl{persp}.",
                    f"Place the banana in the bowl left of the {ref} bowl{persp}.",
                ]
            )
    return prompts


def _between(bowls: list[str], target_idx: int) -> list[str]:
    if target_idx != 1:
        return []
    left, right = bowls[0], bowls[2]
    return [
        f"Put the banana in the bowl between the {left} bowl and the {right} bowl.",
        f"Put the banana into the bowl between the {right} bowl and the {left} bowl.",
        f"Place the banana in the bowl in between the {left} and {right} bowls.",
    ]


def _negation(bowls: list[str], target_idx: int) -> list[str]:
    target_color = bowls[target_idx]
    others = [c for c in bowls if c != target_color]
    if len(others) != 2:
        return []
    a, b = others
    return [
        f"Put the banana into the bowl that is not {a} and not {b}.",
        f"Put the banana in the bowl that is neither {a} nor {b}.",
        f"Place the banana in the bowl that is not {b} and not {a}.",
        f"Put the banana in the bowl that isn't {a} or {b}.",
    ]


def generate_paraphrases(task: str) -> list[str]:
    """Return all compositional paraphrases for the given task string.

    Falls back to an empty list if the task does not parse.
    """
    parsed = parse_task(task)
    if parsed is None:
        return []
    bowls, target_idx = parsed
    return (
        _direct_color(bowls, target_idx)
        + _ordinal_from_side(bowls, target_idx)
        + _relative_to_color(bowls, target_idx)
        + _between(bowls, target_idx)
        + _negation(bowls, target_idx)
    )


def paraphrase_task(task: str, rng: random.Random | None = None) -> str:
    """Return a randomly chosen compositional paraphrase, or the original task if unparseable."""
    choices = generate_paraphrases(task)
    if not choices:
        return task
    return (rng or random).choice(choices)


def all_configurations() -> list[str]:
    """All 18 possible "<perm> : <Position>" task strings."""
    out: list[str] = []
    for perm in permutations("GBR", 3):
        perm_str = "".join(perm)
        for pos in ("Left", "Middle", "Right"):
            out.append(f"{perm_str} : {pos}")
    return out


if __name__ == "__main__":
    rng = random.Random(0)
    for task in all_configurations():
        paraphrases = generate_paraphrases(task)
        print(f"\n=== {task}  ({len(paraphrases)} paraphrases) ===")
        for p in paraphrases:
            print(f"  - {p}")

    print("\n\n### Format-1 sanity checks (assume BRG bowl order) ###")
    for task in [
        "Place the banana into the blue bowl.",
        "Place the banana into the red bowl.",
        "Place the banana into the green bowl.",
    ]:
        parsed = parse_task(task)
        print(f"{task!r} -> {parsed}, sample = {paraphrase_task(task, rng=rng)!r}")
