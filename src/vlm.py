"""
Scene understanding via SmolVLM2-500M-Video-Instruct (works fine on still
images too, and is markedly stronger at perception than SmolVLM v1).

Captures a top-down workspace image and returns a single-sentence
pick-and-place task instruction for the robot.
"""

from __future__ import annotations

import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

_PERCEPTION_PROMPT = (
    "Look at this robot workspace. The scene contains:\n"
    "  - Exactly THREE bowls arranged in a horizontal row. The bowls are "
    "RED, GREEN, and BLUE (one of each, in some order).\n"
    "  - A single YELLOW BANANA in the middle area. The banana is NOT a "
    "bowl — ignore it when identifying bowl colors.\n"
    "\n"
    "Some bowls may be partially cut off at the edge of the frame, but "
    "there are always exactly three bowls (one red, one green, one blue). "
    "If you only clearly see two bowls, the third one's color must be "
    "whichever of {red, green, blue} is not visible.\n"
    "\n"
    "From the robot's perspective, going LEFT to RIGHT across the row of "
    "bowls (not including the banana), what color is each bowl?\n"
    "\n"
    "Reply on ONE line in this exact format (lowercase, no other words):\n"
    "left=<color>, middle=<color>, right=<color>"
)

_REASONING_PROMPT = (
    "You are resolving a pick-and-place instruction into one of three target "
    "positions: left, middle, or right.\n"
    "\n"
    "The bowl layout from the robot's perspective is:\n"
    "  left = {left_color}\n"
    "  middle = {middle_color}\n"
    "  right = {right_color}\n"
    "\n"
    "Reply with EXACTLY one word: left, middle, or right. Nothing else.\n"
    "\n"
    "Examples (the layout shown applies only to that example):\n"
    "\n"
    "Layout: left=blue, middle=red, right=green\n"
    "Task: Put the banana in the red bowl.\n"
    "Answer: middle\n"
    "\n"
    "Layout: left=green, middle=blue, right=red\n"
    "Task: Put the banana in the leftmost bowl.\n"
    "Answer: left\n"
    "\n"
    "Layout: left=red, middle=green, right=blue\n"
    "Task: Put the banana in the 2nd bowl from the left.\n"
    "Answer: middle\n"
    "\n"
    "Layout: left=blue, middle=red, right=green\n"
    "Task: Put the banana in the bowl on the right of the red bowl.\n"
    "Answer: right\n"
    "\n"
    "Layout: left=green, middle=red, right=blue\n"
    "Task: Put the banana in the bowl to the left of the red bowl.\n"
    "Answer: left\n"
    "\n"
    "Layout: left=blue, middle=green, right=red\n"
    "Task: Put the banana in the bowl that is not green and not blue.\n"
    "Answer: right\n"
    "\n"
    "Layout: left=red, middle=blue, right=green\n"
    "Task: Put the banana in the bowl that is neither red nor green.\n"
    "Answer: middle\n"
    "\n"
    "Layout: left=green, middle=blue, right=red\n"
    "Task: Put the banana in the rightmost bowl.\n"
    "Answer: right\n"
    "\n"
    "Layout: left=red, middle=green, right=blue\n"
    "Task: Put the banana in the bowl between the red bowl and the blue bowl.\n"
    "Answer: middle\n"
    "\n"
    "Layout: left={left_color}, middle={middle_color}, right={right_color}\n"
    "Task: {task_spec}\n"
    "Answer:"
)

class SceneVLM:
    def __init__(self, model_id: str = VLM_MODEL_ID, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        print(f"[VLM] Loading {model_id} on {self.device}...", file=sys.stderr)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("[VLM] Ready.", file=sys.stderr)

    @torch.no_grad()
    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        # SmolVLM2 expects the image inline in the message content and
        # tokenization to happen through apply_chat_template.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        target_dtype = torch.float16 if self.device != "cpu" else torch.float32
        inputs = {
            k: (v.to(self.device, dtype=target_dtype) if v.dtype.is_floating_point else v.to(self.device))
            for k, v in inputs.items()
        }
        # Sanity check: confirm pixel_values made it in.
        has_pixels = "pixel_values" in inputs
        px_shape = tuple(inputs["pixel_values"].shape) if has_pixels else None
        print(f"[VLM] input keys={list(inputs.keys())} pixel_values_shape={px_shape}", file=sys.stderr)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip().lower()

    def _parse_layout(self, raw: str) -> dict[str, str]:
        """Extract {left,middle,right} -> color from a free-form VLM response."""
        import re
        colors = ("red", "green", "blue")
        layout: dict[str, str] = {}
        for pos in ("left", "middle", "right"):
            m = re.search(rf"{pos}\s*[:=]\s*(red|green|blue)", raw)
            if m:
                layout[pos] = m.group(1)
        # Fill in the missing slot if exactly one is unknown.
        missing_pos = [p for p in ("left", "middle", "right") if p not in layout]
        if len(missing_pos) == 1:
            used = set(layout.values())
            remaining = [c for c in colors if c not in used]
            if len(remaining) == 1:
                layout[missing_pos[0]] = remaining[0]
        # Final fallback for anything still missing.
        for p in ("left", "middle", "right"):
            if p not in layout:
                used = set(layout.values())
                layout[p] = next((c for c in colors if c not in used), "red")
        return layout

    def query(self, image: Image.Image, task_spec: str) -> str:
        image.save("/tmp/vlm_input.png")
        print(f"[VLM] Input image saved to /tmp/vlm_input.png (size={image.size})", file=sys.stderr)

        # Pass 0 (sanity check): ask the model to describe what it sees.
        # If this comes back as generic / hallucinated, the image is not
        # actually being attended to and perception/reasoning will fail.
        describe_raw = self._generate(
            image,
            "Describe what you see in this image in one or two sentences. "
            "What objects are present and what are their colors?",
            max_new_tokens=120,
        )
        print(f"[VLM] Image description: {describe_raw!r}", file=sys.stderr)

        # Pass 1: perception — extract bowl layout.
        perception_raw = self._generate(image, _PERCEPTION_PROMPT, max_new_tokens=64)
        print(f"[VLM] Perception raw: {perception_raw!r}", file=sys.stderr)
        layout = self._parse_layout(perception_raw)
        print(f"[VLM] Parsed layout: {layout}", file=sys.stderr)

        # Pass 2: reasoning — given the layout, resolve task_spec to a position.
        reasoning_prompt = _REASONING_PROMPT.format(
            left_color=layout["left"],
            middle_color=layout["middle"],
            right_color=layout["right"],
            task_spec=task_spec,
        )
        reasoning_raw = self._generate(image, reasoning_prompt, max_new_tokens=8)
        print(f"[VLM] Reasoning raw: {reasoning_raw!r}", file=sys.stderr)

        position = next((w for w in ("left", "middle", "right") if w in reasoning_raw), None)
        if position is None:
            print(f"[VLM] Unexpected reasoning answer: {reasoning_raw!r}, defaulting to middle",
                  file=sys.stderr)
            position = "middle"

        task = f"Put the banana in the {position} bowl."
        print(f"[VLM] layout={layout} | task_spec={task_spec!r} → position={position!r} → task={task!r}",
              file=sys.stderr)
        return task
