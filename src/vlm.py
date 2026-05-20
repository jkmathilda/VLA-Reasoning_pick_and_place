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

VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "LiquidAI/LFM2.5-VL-1.6B")
# Sampling temperature for the free-form reasoning phase. 0.0 = greedy decoding.
VLM_REASONING_TEMP = float(os.getenv("VLM_REASONING_TEMP", "0.1"))

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
    "Reasoning: Bowls are blue (left), red (middle), green (right). "
    "Red is at the middle position.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example B — ordinal (leftmost) ===\n"
    "Imagined bowl order: left=green, middle=blue, right=red.\n"
    "Instruction: Put the banana in the leftmost bowl.\n"
    "Reasoning: 'Leftmost' is positional; ignore colors. The leftmost "
    "bowl is at the left position.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example C — spatial relative ===\n"
    "Imagined bowl order: left=blue, middle=red, right=green.\n"
    "Instruction: Put the banana in the bowl on the right of the red bowl.\n"
    "Reasoning: Bowls are blue (left), red (middle), green (right). "
    "Red is at middle; one step right of middle is the right position.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example D — negation ===\n"
    "Imagined bowl order: left=red, middle=blue, right=green.\n"
    "Instruction: Put the banana in the bowl that is neither red nor green.\n"
    "Reasoning: Bowls are red (left), blue (middle), green (right). "
    "Excluding red and green leaves blue, which is at the middle position.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example E — between two colors ===\n"
    "Imagined bowl order: left=red, middle=green, right=blue.\n"
    "Instruction: Put the banana in the bowl between the red bowl and the blue bowl.\n"
    "Reasoning: Bowls are red (left), green (middle), blue (right). "
    "Between left (red) and right (blue) is the middle position.\n"
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
    "Write a thorough reasoning of about 200 words. Walk through it "
    "step by step: identify each bowl's color using the camera-geometry "
    "rule, restate the instruction, list candidate bowls, rule out the "
    "wrong ones, and conclude with the target POSITION (left, middle, "
    "or right) — NOT a color. End with a single line:\n"
    "  Conclusion: <left|middle|right>.\n"
    "The last word must be a position word, not a color."
)

_FINAL_PROMPT = (
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

    "Based on the reasoning above, write your FINAL ANSWER now.\n"
    "\n"
    "Step 1: Look at the reasoning above and find the POSITION word "
    "(left, middle, or right) that the reasoning concluded with — "
    "typically stated in a sentence like 'the target is X' or "
    "'Conclusion: X' near the end of the reasoning. This position word "
    "is your answer.\n"
    "\n"
    "Step 2: Plug that position into the canonical sentence template:\n"
    "  'Put the banana in the <POSITION> bowl.'\n"
    "\n"
    "Do NOT re-derive the answer from scratch. Do NOT re-identify the "
    "bowl colors. Do NOT use color names (red/green/blue) in your "
    "answer. The only valid position words are: left, middle, right. "
    "Copy whichever position the reasoning concluded with.\n"
    "\n"
    "STRICT OUTPUT RULES: reply with EXACTLY ONE of these three "
    "sentences verbatim, and absolutely nothing else (no preamble, no "
    "explanation, no extra punctuation):\n"
    "  Put the banana in the left bowl.\n"
    "  Put the banana in the middle bowl.\n"
    "  Put the banana in the right bowl.\n"
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
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
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
        target_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        inputs = {
            k: (v.to(self.device, dtype=target_dtype) if v.dtype.is_floating_point else v.to(self.device))
            for k, v in inputs.items()
        }
        has_pixels = "pixel_values" in inputs
        px_shape = tuple(inputs["pixel_values"].shape) if has_pixels else None
        print(f"[VLM] input keys={list(inputs.keys())} pixel_values_shape={px_shape} "
              f"temperature={temperature}", file=sys.stderr)
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            # Discourage degenerate "the bowl is blue. the bowl is blue. …" loops.
            # 1.0 = no penalty; >1 penalizes already-seen tokens.
            "repetition_penalty": 1.05,
        }
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature, min_p=0.15)
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
            reasoning_messages, max_new_tokens=400, temperature=VLM_REASONING_TEMP,
        )
        print(f"[VLM] Phase 1 reasoning: {reasoning_text!r}", file=sys.stderr)

        # Phase 2: minimal prompt. We DO NOT re-include the worked examples
        # from phase 1 — those end with "Conclusion: <position>" lines, and
        # if the model sees them again it tends to mimic that format
        # instead of the canonical sentence we want. We only need: the
        # original task spec, the model's own reasoning, and the strict
        # output instruction.
        combined_text = (
            f"The user's instruction was: {task_spec}\n\n"
            "Your reasoning:\n"
            f"{reasoning_text}\n\n"
            f"{_FINAL_PROMPT}"
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
