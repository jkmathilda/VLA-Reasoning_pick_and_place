""
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
    "There are three bowls arranged in a row from the robot's perspective: "
    "one on the left, one in the middle, and one on the right. "
    "One bowl is blue, one is red, and one is green — the colors and positions vary. "
    "A user gives an abstract task and you must reply with which bowl the banana goes in. "
    "Reply with only one word: left, middle, or right.\n"
    "\n"
    "Important: the camera view may be partial — sometimes one of the three bowls "
    "is cut off at the edge of the frame or only barely visible. Since there are "
    "always exactly three bowls (one red, one green, one blue), you can infer the "
    "color of any bowl that is not fully visible: it must be whichever of "
    "{{red, green, blue}} is not present among the bowls you can clearly see. "
    "Use this deduction when reasoning about color-based instructions.\n"
    "\n"
    "Here are some example questions and answers (the scene in each example is "
    "described in parentheses; in the real question you must look at the image):\n"
    "\n"
    "Example 1 (scene: left=blue, middle=red, right=green):\n"
    "  Q: Put the banana in the red bowl.\n"
    "  A: middle\n"
    "\n"
    "Example 2 (scene: left=green, middle=blue, right=red):\n"
    "  Q: Put the banana in the leftmost bowl.\n"
    "  A: left\n"
    "\n"
    "Example 3 (scene: left=red, middle=green, right=blue):\n"
    "  Q: Put the banana into the 2nd bowl from the left from the robot perspective.\n"
    "  A: middle\n"
    "\n"
    "Example 4 (scene: left=blue, middle=red, right=green):\n"
    "  Q: Put the banana into the bowl on the right of the red bowl from the robot perspective.\n"
    "  A: right\n"
    "\n"
    "Example 5 (scene: left=green, middle=red, right=blue):\n"
    "  Q: Put the banana into the bowl to the left of the red bowl.\n"
    "  A: left\n"
    "\n"
    "Example 6 (scene: left=blue, middle=green, right=red):\n"
    "  Q: Put the banana in the bowl that is not green and not blue.\n"
    "  A: right\n"
    "\n"
    "Example 7 (scene: left=red, middle=blue, right=green):\n"
    "  Q: Put the banana in the bowl that is neither red nor green.\n"
    "  A: middle\n"
    "\n"
    "Example 8 (scene: left=green, middle=blue, right=red):\n"
    "  Q: Put the banana in the rightmost bowl.\n"
    "  A: right\n"
    "\n"
    "Example 9 (scene: left=red, middle=green, right=blue):\n"
    "  Q: Put the banana in the bowl between the red bowl and the blue bowl.\n"
    "  A: middle\n"
    "\n"
    "Now look at the actual scene in the image above and answer this question.\n"
    "Q: {task_spec}\n"
    "A:"
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
        image.save("/tmp/vlm_input.png"); print(f"[VLM] Input image saved to /tmp/vlm_input.png (size={image.size})", file=sys.stderr)
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
            print(f"VLM returned unexpected answer: {answer!r}, defaulting to middle")
            position = "middle"

        task = f"Put the banana in the {position} bowl."
        


        import sys
        print(f"[VLM] Position: {position!r} → task: {task!r}", file=sys.stderr)
        return task
