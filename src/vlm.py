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
# Sampling temperature for the free-form reasoning phase. 0.0 = greedy decoding.
VLM_REASONING_TEMP = float(os.getenv("VLM_REASONING_TEMP", "0.3"))

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
    "Below are WORKED EXAMPLES that show exactly the style of reasoning "
    "you should use. Each example invents an imaginary bowl ordering — "
    "the REAL bowl order for your answer must come from the image you "
    "are actually looking at. Study the reasoning carefully:\n"
    "\n"
    "=== Example A — direct color lookup ===\n"
    "Imagined bowl order: left=blue, middle=red, right=green.\n"
    "Instruction: Put the banana in the red bowl.\n"
    "Reasoning: The bowls are blue (left), red (middle), green (right). "
    "The instruction asks for the red bowl. The red bowl is at the middle "
    "position. Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example B — ordinal (leftmost) ===\n"
    "Imagined bowl order: left=green, middle=blue, right=red.\n"
    "Instruction: Put the banana in the leftmost bowl.\n"
    "Reasoning: The bowls are green (left), blue (middle), red (right). "
    "'Leftmost' means the bowl that is furthest to the left, which is the "
    "left bowl regardless of its color. The left bowl is green. Therefore "
    "the target is left.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example C — ordinal (Nth from the left) ===\n"
    "Imagined bowl order: left=red, middle=green, right=blue.\n"
    "Instruction: Put the banana in the 2nd bowl from the left.\n"
    "Reasoning: The bowls are red (left), green (middle), blue (right). "
    "Counting from the left: 1st = left (red), 2nd = middle (green), 3rd = "
    "right (blue). The 2nd bowl from the left is the middle bowl. "
    "Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example D — spatial relative (right of <color>) ===\n"
    "Imagined bowl order: left=blue, middle=red, right=green.\n"
    "Instruction: Put the banana in the bowl on the right of the red bowl.\n"
    "Reasoning: The bowls are blue (left), red (middle), green (right). "
    "The red bowl is at the middle position. The bowl immediately to the "
    "right of the middle position is the right position, which holds the "
    "green bowl. Therefore the target is right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example E — spatial relative (left of <color>) ===\n"
    "Imagined bowl order: left=green, middle=red, right=blue.\n"
    "Instruction: Put the banana in the bowl to the left of the red bowl.\n"
    "Reasoning: The bowls are green (left), red (middle), blue (right). "
    "The red bowl is at the middle position. The bowl immediately to the "
    "left of the middle position is the left position, which holds the "
    "green bowl. Therefore the target is left.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example F — negation (not X and not Y) ===\n"
    "Imagined bowl order: left=blue, middle=green, right=red.\n"
    "Instruction: Put the banana in the bowl that is not green and not blue.\n"
    "Reasoning: The bowls are blue (left), green (middle), red (right). "
    "There are three colors: red, green, blue. The instruction excludes "
    "green and blue, so the target color must be red. The red bowl is at "
    "the right position. Therefore the target is right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example G — negation (neither X nor Y) ===\n"
    "Imagined bowl order: left=red, middle=blue, right=green.\n"
    "Instruction: Put the banana in the bowl that is neither red nor green.\n"
    "Reasoning: The bowls are red (left), blue (middle), green (right). "
    "'Neither red nor green' means not red and not green, so the target "
    "color must be blue. The blue bowl is at the middle position. "
    "Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example H — ordinal (rightmost) ===\n"
    "Imagined bowl order: left=green, middle=blue, right=red.\n"
    "Instruction: Put the banana in the rightmost bowl.\n"
    "Reasoning: The bowls are green (left), blue (middle), red (right). "
    "'Rightmost' means the bowl furthest to the right, which is the right "
    "bowl regardless of color. The right bowl is red. Therefore the target "
    "is right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example I — between two colors ===\n"
    "Imagined bowl order: left=red, middle=green, right=blue.\n"
    "Instruction: Put the banana in the bowl between the red bowl and the blue bowl.\n"
    "Reasoning: The bowls are red (left), green (middle), blue (right). "
    "The red bowl is at the left position, the blue bowl is at the right "
    "position. The bowl between them must be the middle position, which "
    "holds the green bowl. Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example J — direct color lookup, different ordering ===\n"
    "Imagined bowl order: left=blue, middle=red, right=green.\n"
    "Instruction: Put the banana in the green bowl.\n"
    "Reasoning: The bowls are blue (left), red (middle), green (right). "
    "The green bowl is at the right position. Therefore the target is right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example K — spatial relative, color landmark on the side ===\n"
    "Imagined bowl order: left=green, middle=red, right=blue.\n"
    "Instruction: Put the banana in the bowl directly to the right of the green bowl.\n"
    "Reasoning: The bowls are green (left), red (middle), blue (right). "
    "The green bowl is at the left position. The bowl immediately to the "
    "right of the left position is the middle position, which holds the "
    "red bowl. Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== END OF EXAMPLES ===\n"
    "\n"
    "Now apply the same style of reasoning to the ACTUAL image you are "
    "looking at and the ACTUAL instruction above. Identify the real bowl "
    "order from the image (remember: middle = highest in the frame), then "
    "interpret the instruction, then conclude which position (left, middle, "
    "or right) is the target.\n"
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
    def _generate_messages(self, messages: list, max_new_tokens: int, temperature: float = 0.0) -> str:
        """Run generation on a free-form messages list (supports multi-turn).

        temperature=0.0 → greedy decoding. temperature>0 → sampling with that
        temperature (and top_p=0.95).
        """
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
        print(f"[VLM] input keys={list(inputs.keys())} pixel_values_shape={px_shape} "
              f"temperature={temperature}", file=sys.stderr)
        gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        outputs = self.model.generate(**inputs, **gen_kwargs)
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
        reasoning_text = self._generate_messages(
            reasoning_messages, max_new_tokens=2000, temperature=VLM_REASONING_TEMP,
        )
        print(f"[VLM] Phase 1 reasoning: {reasoning_text!r}", file=sys.stderr)

        # Phase 2: build a single user turn that contains the original
        # reasoning prompt, the model's own reasoning text appended in-line,
        # and the final strict-output instruction.
        combined_text = (
            _REASONING_PROMPT.format(task_spec=task_spec)
            + "\n\nYour reasoning so far:\n"
            + reasoning_text
            + "\n\n"
            + _FINAL_PROMPT
        )
        final_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": combined_text},
            ],
        }]
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
