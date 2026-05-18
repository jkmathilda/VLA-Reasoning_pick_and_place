"""
Scene understanding via SmolVLM-256M-Instruct.

Captures a top-down workspace image and returns a single-sentence
pick-and-place task instruction for the robot.
"""
from __future__ import annotations

import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM-256M-Instruct")

_PROMPT_TEMPLATE = (
    "Look at this robot workspace. "
    "There are three bowls: one on the left, one in the middle, and one on the right. "
    "The user's instruction is: \"{task_spec}\". "
    "Based on this instruction and the scene, which bowl should the banana be placed in? Reason yourself."
    # "Reply with exactly one word: left, middle, or right."
    # "What is the color of the bowl on the left?"
    "Reply with left, middle, or right with your reasoning."
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
    def query(self, image: Image.Image, task_spec: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(task_spec=task_spec)
        print(f"TEST1: {prompt}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=5000, do_sample=False)
        answer = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip().lower()

        print(f"TEST2: {answer}")

        position = next((w for w in ("left", "middle", "right") if w in answer), None)
        if position is None:
            raise ValueError(f"VLM returned unexpected answer: {answer!r}")

        if position == 'left':
            task = "Put the banana in the blue bowl."
        elif position == 'middle':
            task = "Put the banana in the red bowl."
        else: # right
            task = "Put the banana in the green bowl."


        import sys
        print(f"[VLM] Position: {position!r} → task: {task!r}", file=sys.stderr)
        return task
