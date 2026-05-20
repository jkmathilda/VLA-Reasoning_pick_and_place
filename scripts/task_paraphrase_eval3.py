"""Task paraphrasing for the eval 3 celebrity coke-can task.

Recorded tasks are simple position labels ('left' / 'middle' / 'right').
At inference a VLM maps the celebrity name in the user prompt to one of
these positions; the policy is trained on position-conditioned paraphrases
referring to the coke can and the target image.
"""

from __future__ import annotations

import random


_POSITION_VERB_TEMPLATES: list[str] = [
    "Put the coke on the {position} image.",
    "Put the coke can on the {position} image.",
    "Place the coke on the {position} image.",
    "Place the coke can on the {position} image.",
    "Set the coke on the {position} image.",
    "Set the coke can on the {position} image.",
    "Drop the coke on the {position} image.",
    "Drop the coke can on the {position} image.",
    "Move the coke onto the {position} image.",
    "Move the coke can onto the {position} image.",
    "Place the can on the {position} picture.",
    "Put the can on the {position} picture.",
    "Place the coke on the {position} picture.",
    "Place the coke can on the {position} picture.",
    "Put the coke on the {position} picture.",
    "Put the coke can on the {position} picture.",
    "Place the coke on the {position} photo.",
    "Place the coke can on the {position} photo.",
    "Put the coke on the {position} photo.",
    "Put the coke can on the {position} photo.",
    "Place the coke on top of the {position} image.",
    "Place the coke can on top of the {position} image.",
    "Pick up the coke and place it on the {position} image.",
    "Pick up the coke can and place it on the {position} image.",
    "Pick up the coke and put it on the {position} image.",
    "Grab the coke and place it on the {position} image.",
    "Grab the coke can and place it on the {position} image.",
    "Take the coke and put it on the {position} image.",
    "Please place the coke on the {position} image.",
    "Please put the coke on the {position} image.",
    "Can you place the coke on the {position} image?",
    "Can you put the coke can on the {position} image?",
    "The coke goes on the {position} image.",
    "The coke can goes on the {position} image.",
    "Coke on the {position} image, please.",
]

POSITIONS: tuple[str, ...] = ("left", "middle", "right")

TASK_PARAPHRASES_BY_POSITION: dict[str, list[str]] = {
    pos: [tmpl.format(position=pos) for tmpl in _POSITION_VERB_TEMPLATES] for pos in POSITIONS
}


def parse_position(task: str) -> str | None:
    lowered = task.lower()
    for pos in POSITIONS:
        if pos in lowered:
            return pos
    return None


def paraphrase_task(task: str, rng: random.Random | None = None) -> str:
    position = parse_position(task) or "middle"
    choices = TASK_PARAPHRASES_BY_POSITION[position]
    return (rng or random).choice(choices)


if __name__ == "__main__":
    for ex in ["left", "middle", "right"]:
        print(ex, "->", paraphrase_task(ex))
