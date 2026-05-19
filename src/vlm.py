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

_REASONING_PROMPT = (
    "Look carefully at the image of the robot workspace. The scene contains:\n"
    "  - Exactly THREE bowls (one RED, one GREEN, one BLUE — one of each).\n"
    "  - A yellow banana. The banana is NOT a bowl — ignore it for color "
    "identification.\n"
    "\n"
    "CAMERA GEOMETRY (very important):\n"
    "The camera looks down and forward at an angle, so the bowls form an "
    "ARC, not a straight horizontal line:\n"
    "  - The LEFT bowl appears in the bottom-left area of the image.\n"
    "  - The MIDDLE bowl is the one closest to the TOP of the image "
    "(farther from the robot — perspective makes it appear higher up).\n"
    "  - The RIGHT bowl appears in the bottom-right area of the image.\n"
    "Do NOT identify the middle bowl as the one in the geometric center "
    "of the image — pick the one that is HIGHEST in the frame.\n"
    "\n"
    "Your task is to figure out which bowl (left, middle, or right) the "
    "banana should go into, based on this instruction:\n"
    "\n"
    "  Instruction: {task_spec}\n"
    "\n"
    "Reason in plain English, step by step:\n"
    "  1. State the color of each bowl: left=?, middle=?, right=?\n"
    "  2. Interpret the instruction.\n"
    "  3. Conclude which bowl position (left, middle, or right) is the target.\n"
    "\n"
    "Use as many sentences as you want — this is your scratch space. You "
    "will write the final one-sentence answer in a follow-up message."
)

_FINAL_PROMPT = (
    "Based on your reasoning above, write your FINAL ANSWER now.\n"
    "\n"
    "STRICT OUTPUT RULES: reply with EXACTLY ONE of these three sentences "
    "verbatim, and absolutely nothing else (no preamble, no explanation, "
    "no extra punctuation):\n"
    "  Put the banana in the left bowl.\n"
    "  Put the banana in the middle bowl.\n"
    "  Put the banana in the right bowl.\n"
    "\n"
    "The only words allowed in the position slot are: left, middle, or right."
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
    "STRICT OUTPUT RULES — READ CAREFULLY:\n"
    "Your reply must be EXACTLY one of these THREE sentences, verbatim, "
    "and absolutely NOTHING ELSE — no preamble, no reasoning, no extra "
    "words, no quotation marks, no punctuation beyond the final period:\n"
    "  Put the banana in the left bowl.\n"
    "  Put the banana in the middle bowl.\n"
    "  Put the banana in the right bowl.\n"
    "\n"
    "You are FORBIDDEN from outputting any other text. The only words "
    "that may appear at the position slot are: left, middle, or right. "
    "Do not output colors, do not output coordinates, do not explain "
    "yourself. Just emit ONE of the three sentences above and stop.\n"
    "\n"
    "Tasks may be phrased many different ways. Examples below show "
    "different phrasings and reasoning patterns. Each example uses its "
    "own imaginary bowl ordering — the REAL bowl order comes from the "
    "image you are actually looking at.\n"
    "\n"
    "Example A (bowl order left=blue, middle=red, right=green):\n"
    "Task: Put the banana in the red bowl.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Example B (bowl order left=green, middle=blue, right=red):\n"
    "Task: Put the banana in the leftmost bowl.\n"
    "Put the banana in the left bowl.\n"
    "\n"
    "Example C (bowl order left=red, middle=green, right=blue):\n"
    "Task: Put the banana in the 2nd bowl from the left.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Example D (bowl order left=blue, middle=red, right=green):\n"
    "Task: Put the banana in the bowl on the right of the red bowl.\n"
    "Put the banana in the right bowl.\n"
    "\n"
    "Example E (bowl order left=green, middle=red, right=blue):\n"
    "Task: Put the banana in the bowl to the left of the red bowl.\n"
    "Put the banana in the left bowl.\n"
    "\n"
    "Example F (bowl order left=blue, middle=green, right=red):\n"
    "Task: Put the banana in the bowl that is not green and not blue.\n"
    "Put the banana in the right bowl.\n"
    "\n"
    "Example G (bowl order left=red, middle=blue, right=green):\n"
    "Task: Put the banana in the bowl that is neither red nor green.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Example H (bowl order left=green, middle=blue, right=red):\n"
    "Task: Put the banana in the rightmost bowl.\n"
    "Put the banana in the right bowl.\n"
    "\n"
    "Example I (bowl order left=red, middle=green, right=blue):\n"
    "Task: Put the banana in the bowl between the red bowl and the blue bowl.\n"
    "Put the banana in the middle bowl.\n"
    "\n"
    "Example J (bowl order left=blue, middle=red, right=green):\n"
    "Task: Put the banana in the green bowl.\n"
    "Put the banana in the right bowl.\n"
    "\n"
    "Example K (bowl order left=green, middle=red, right=blue):\n"
    "Task: Put the banana in the bowl directly to the right of the green bowl.\n"
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
    def _generate_messages(self, messages: list, max_new_tokens: int) -> str:
        """Run generation on a free-form messages list (supports multi-turn)."""
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
        has_pixels = "pixel_values" in inputs
        px_shape = tuple(inputs["pixel_values"].shape) if has_pixels else None
        print(f"[VLM] input keys={list(inputs.keys())} pixel_values_shape={px_shape}", file=sys.stderr)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Single-turn convenience wrapper (image + text user message)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        return self._generate_messages(messages, max_new_tokens).lower()

    def query(self, image: Image.Image, task_spec: str) -> str:
        image.save("/tmp/vlm_input.png")
        print(f"[VLM] Input image saved to /tmp/vlm_input.png (size={image.size})", file=sys.stderr)

        # Phase 1: free-form reasoning. The model writes whatever it wants
        # about the scene and how it interprets the instruction.
        reasoning_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _REASONING_PROMPT.format(task_spec=task_spec)},
            ],
        }]
        reasoning_text = self._generate_messages(reasoning_messages, max_new_tokens=256)
        print(f"[VLM] Phase 1 reasoning: {reasoning_text!r}", file=sys.stderr)

        # Phase 2: feed the reasoning back as an assistant turn, then ask
        # for the strict canonical final sentence.
        final_messages = reasoning_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": reasoning_text}]},
            {"role": "user", "content": [{"type": "text", "text": _FINAL_PROMPT}]},
        ]
        final_raw = self._generate_messages(final_messages, max_new_tokens=24).lower()
        print(f"[VLM] Phase 2 final raw: {final_raw!r}", file=sys.stderr)

        # Return the last non-empty line verbatim.
        task = next((ln.strip() for ln in reversed(final_raw.splitlines()) if ln.strip()),
                    "Put the banana in the middle bowl.")
        # Capitalize first letter so it matches training-time task casing.
        if task and task[0].islower():
            task = task[0].upper() + task[1:]
        print(f"[VLM] task_spec={task_spec!r} → task={task!r}", file=sys.stderr)
        return task
