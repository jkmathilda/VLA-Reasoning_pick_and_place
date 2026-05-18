"""Task paraphrasing for color-invariant policy training.

Two recorded data formats are normalized to position-based natural language:
  Format 1: "Place the banana into the <color> bowl."  (fixed bowl order BRG)
  Format 2: "<perm> : <Position>"  (e.g. "GBR : Middle"), produced by the VLM

Both are converted to one of many paraphrases of:
  "Place the banana in the <left|middle|right> bowl."
"""

from __future__ import annotations

import random


_POSITION_VERB_TEMPLATES: list[str] = [
    "Put the banana in the {position} bowl.",
    "Put the banana into the {position} bowl.",
    "Place the banana in the {position} bowl.",
    "Place the banana into the {position} bowl.",
    "Move the banana to the {position} bowl.",
    "Move the banana into the {position} bowl.",
    "Drop the banana in the {position} bowl.",
    "Drop the banana into the {position} bowl.",
    "Set the banana in the {position} bowl.",
    "Set the banana into the {position} bowl.",
    "Transfer the banana to the {position} bowl.",
    "Transfer the banana into the {position} bowl.",
    "Pick up the banana and put it in the {position} bowl.",
    "Pick up the banana and place it in the {position} bowl.",
    "Pick up the banana and drop it in the {position} bowl.",
    "Grab the banana and put it in the {position} bowl.",
    "Grab the banana and place it in the {position} bowl.",
    "Take the banana and put it in the {position} bowl.",
    "Take the banana and place it in the {position} bowl.",
    "Relocate the banana to the {position} bowl.",
    "Deposit the banana in the {position} bowl.",
    "Deposit the banana into the {position} bowl.",
    "The banana goes in the {position} bowl.",
    "The banana belongs in the {position} bowl.",
    "Banana into the {position} bowl, please.",
    "Please put the banana in the {position} bowl.",
    "Please place the banana in the {position} bowl.",
    "Can you put the banana in the {position} bowl?",
    "Move the yellow fruit into the {position} bowl.",
    "Place the yellow fruit in the {position} bowl.",
    "Put the yellow fruit in the {position} bowl.",
]

POSITIONS: tuple[str, ...] = ("left", "middle", "right")

# Format 1 recordings used a fixed physical bowl order: Blue, Red, Green from left to right.
FORMAT1_COLOR_TO_POSITION: dict[str, str] = {
    "blue": "left",
    "red": "middle",
    "green": "right",
}

TASK_PARAPHRASES_BY_POSITION: dict[str, list[str]] = {
    pos: [tmpl.format(position=pos) for tmpl in _POSITION_VERB_TEMPLATES] for pos in POSITIONS
}


def parse_position(task: str) -> str | None:
    """Return 'left' | 'middle' | 'right' for a task in either format, else None.

    Format 2 ("<perm> : <Position>") is checked first so that an explicit
    position annotation wins over any incidental color word in the prefix.
    Format 1 falls back to the fixed BRG color->position mapping.
    """
    lowered = task.lower()

    if ":" in lowered:
        right_side = lowered.rsplit(":", 1)[1].strip()
        for pos in POSITIONS:
            if pos in right_side:
                return pos

    for pos in POSITIONS:
        if pos in lowered:
            return pos

    for color, pos in FORMAT1_COLOR_TO_POSITION.items():
        if color in lowered:
            return pos

    return None


def paraphrase_task(task: str, rng: random.Random | None = None) -> str:
    """Return a randomly chosen position-based paraphrase, or the original task if unparseable."""
    position = parse_position(task) or "middle"
    choices = TASK_PARAPHRASES_BY_POSITION[position]
    return (rng or random).choice(choices)


if __name__ == "__main__":
    examples = [
        "Place the banana into the red bowl",
        "Place the banana into the blue bowl",
        "Place the banana into the green bowl",
        "Pt the bannna in the red bwl",
        "GBR : Middle",
        "RGB : Left",
        "BRG : Right",
        "something unrelated",
    ]

    rng = random.Random(0)
    print(f"{'INPUT':<45} -> POSITION  PARAPHRASE")
    print("-" * 100)
    for ex in examples:
        pos = parse_position(ex)
        out = paraphrase_task(ex, rng=rng)
        print(f"{ex:<45} -> {str(pos):<8}  {out}")
