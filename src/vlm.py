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

VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

_PERCEPTION_PROMPT = (
    "Look carefully at this robot workspace. The scene contains:\n"
    "  - Exactly THREE bowls (one RED, one GREEN, one BLUE — one of each).\n"
    "  - A yellow banana. The banana is NOT a bowl — ignore it for this task.\n"
    "\n"
    "IMPORTANT — camera geometry:\n"
    "The camera is mounted on the robot looking down and forward at an "
    "angle, so the bowls form an ARC, not a straight horizontal line. "
    "Specifically:\n"
    "  - The LEFT bowl appears in the bottom-left area of the image.\n"
    "  - The MIDDLE bowl is the one closest to the TOP of the image "
    "(it is farther from the robot, so perspective makes it appear higher "
    "up in the frame), and is horizontally centered.\n"
    "  - The RIGHT bowl appears in the bottom-right area of the image.\n"
    "\n"
    "So to identify the middle bowl, do NOT just pick the one in the "
    "geometric center of the image — pick the one that is highest in the "
    "frame. The other two (left and right) sit lower and to the sides.\n"
    "\n"
    "Task: identify the color of each of the three bowls (left, middle, right).\n"
    "\n"
    "Reply on ONE line in this exact format (lowercase, no extra words):\n"
    "left=<color>, middle=<color>, right=<color>"
)

_COMBINED_PROMPT = (
    "You are resolving a pick-and-place instruction into one of three target "
    "positions: left, middle, or right.\n"
    "\n"
    "SCENE:\n"
    "The image shows a robot workspace containing:\n"
    "  - Exactly THREE bowls (one RED, one GREEN, one BLUE — one of each).\n"
    "  - A yellow banana. The banana is NOT a bowl — ignore it for color "
    "identification.\n"
    "\n"
    "CAMERA GEOMETRY (very important):\n"
    "The camera looks down and forward at an angle, so the bowls form an "
    "ARC, not a straight horizontal line:\n"
    "  - The LEFT bowl appears in the bottom-left area of the image.\n"
    "  - The MIDDLE bowl is the one closest to the TOP of the image "
    "(farther from the robot — perspective makes it appear higher up), "
    "horizontally centered.\n"
    "  - The RIGHT bowl appears in the bottom-right area of the image.\n"
    "Do NOT identify the middle bowl as the one in the geometric center "
    "of the image — pick the one that is highest in the frame.\n"
    "\n"
    "INSTRUCTIONS:\n"
    "Step 1: Identify the color of each bowl in the image (left, middle, right).\n"
    "Step 2: Read the user's task and figure out which bowl it refers to.\n"
    "Step 3: Output the bowl POSITION (left, middle, or right).\n"
    "\n"
    "Your final output must be EXACTLY one of these three sentences (and "
    "nothing else after it):\n"
    "  Put the banana in the left bowl.\n"
    "  Put the banana in the middle bowl.\n"
    "  Put the banana in the right bowl.\n"
    "\n"
    "Examples (each example has its own scene — use them to learn the "
    "REASONING and OUTPUT FORMAT; the actual bowl layout comes from the image):\n"
    "\n"
    "Example A:\n"
    "Task: Put the banana in the red bowl.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Example B:\n"
    "Task: Put the banana in the leftmost bowl.\n"
    "Put the banana in the left bowl.\n"
    "\n"
    "Example C:\n"
    "Task: Put the banana in the 2nd bowl from the left.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Example D:\n"
    "Task: Put the banana in the bowl on the right of the red bowl.\n"
    "Put the banana in the right bowl.\n"
    "\n"
    "Example E:\n"
    "Task: Put the banana in the bowl to the left of the red bowl.\n"
    "Put the banana in the left bowl.\n"
    "\n"
    "Example F:\n"
    "Task: Put the banana in the bowl that is not green and not blue.\n"
    "Put the banana in the right bowl.\n"
    "\n"
    "Example G:\n"
    "Task: Put the banana in the bowl that is neither red nor green.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Example H:\n"
    "Task: Put the banana in the rightmost bowl.\n"
    "Put the banana in the right bowl.\n"
    "\n"
    "Example I:\n"
    "Task: Put the banana in the bowl between the red bowl and the blue bowl.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Now look at the actual image above and answer.\n"
    "Task: {task_spec}\n"
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

    def query(self, image: Image.Image, task_spec: str) -> str:
        image.save("/tmp/vlm_input.png")
        print(f"[VLM] Input image saved to /tmp/vlm_input.png (size={image.size})", file=sys.stderr)

        prompt = _COMBINED_PROMPT.format(task_spec=task_spec)
        raw = self._generate(image, prompt, max_new_tokens=32)
        print(f"[VLM] Raw output: {raw!r}", file=sys.stderr)

        # The VLM is asked to emit one of the canonical training-time task
        # sentences. Take the last non-empty line and return it directly.
        task = next((ln.strip() for ln in reversed(raw.splitlines()) if ln.strip()),
                    "Put the banana in the middle bowl.")
        print(f"[VLM] task_spec={task_spec!r} → task={task!r}", file=sys.stderr)
        return task
