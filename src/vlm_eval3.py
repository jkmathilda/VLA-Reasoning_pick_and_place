"""
Scene understanding for eval 3 (celebrity coke-can task).

The workspace has three DIN A5 celebrity printouts laid out in a semicircle
in front of the robot, with a slim coke can standing in the middle. The
user's instruction names a celebrity ("Place the coke on Taylor Swift");
the VLM must identify which position (left / middle / right) holds that
celebrity's image and emit a canonical position-conditioned sentence that
the policy was trained on.
"""

from __future__ import annotations

import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
VLM_REASONING_TEMP = float(os.getenv("VLM_REASONING_TEMP", "0.1"))

_REASONING_PROMPT = (
    "Look carefully at the image of the robot workspace. The scene contains:\n"
    "  - Exactly THREE printed photographs of celebrities (DIN A5 color "
    "prints), laid flat on the table in a semicircle.\n"
    "  - A slim coke can standing somewhere in the middle area.\n"
    "\n"
    "CAMERA GEOMETRY (very important):\n"
    "The camera looks down and forward at an angle, so the three images "
    "form an ARC, not a straight horizontal line:\n"
    "  - The LEFT image appears in the bottom-left area of the frame.\n"
    "  - The MIDDLE image is the one closest to the TOP of the frame "
    "(farther from the robot — perspective makes it appear higher up).\n"
    "  - The RIGHT image appears in the bottom-right area of the frame.\n"
    "Do NOT identify the middle image as the one in the geometric center "
    "of the frame — pick the one that is HIGHEST in the frame.\n"
    "\n"
    "Your task is to figure out which position (left, middle, or right) "
    "the coke can should be placed on, based on this instruction:\n"
    "\n"
    "  Instruction: {task_spec}\n"
    "\n"
    "Reason in plain English, step by step:\n"
    "  1. Identify the person in each image: left=?, middle=?, right=?\n"
    "     Use any visible cues — face, hair, clothing, age, well-known "
    "     features. The known in-distribution people are Taylor Swift, "
    "     Barack Obama, and Yann LeCun, but the image at eval time may "
    "     show OTHER famous people too.\n"
    "  2. Interpret the instruction — which person is it asking for?\n"
    "  3. Conclude which position (left, middle, or right) holds that "
    "     person's image. That position is the target.\n"
    "\n"
    "=== Example A — known person, direct lookup ===\n"
    "Imagined image order: left=Obama, middle=Swift, right=LeCun.\n"
    "Instruction: Place the coke on Taylor Swift.\n"
    "Reasoning: Images show Obama (left), Swift (middle), LeCun (right). "
    "The instruction asks for Swift, which is at the middle position.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example B — known person, edge position ===\n"
    "Imagined image order: left=LeCun, middle=Obama, right=Swift.\n"
    "Instruction: Place the coke on Yann LeCun.\n"
    "Reasoning: Images show LeCun (left), Obama (middle), Swift (right). "
    "LeCun is at the left position.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example C — out-of-distribution celebrity (soccer) ===\n"
    "Imagined image order: left=Einstein, middle=Beyonce, right=Messi.\n"
    "Instruction: Place the coke on Lionel Messi.\n"
    "Reasoning: Images show Einstein (left, white hair, mustache), "
    "Beyonce (middle, singer), Messi (right, Argentine soccer player). "
    "Messi is on the right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example D — first-name only ===\n"
    "Imagined image order: left=Swift, middle=LeCun, right=Obama.\n"
    "Instruction: Place the coke on Barack.\n"
    "Reasoning: Images show Swift (left), LeCun (middle), Obama (right). "
    "'Barack' refers to Barack Obama, who is at the right position.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example E — out-of-distribution celebrity (tech) ===\n"
    "Imagined image order: left=Elon Musk, middle=Trump, right=Ronaldo.\n"
    "Instruction: Place the coke on Elon Musk.\n"
    "Reasoning: Images show Elon Musk (left, Tesla/SpaceX CEO), "
    "Donald Trump (middle, former US president), Cristiano Ronaldo "
    "(right, Portuguese soccer player). Musk is on the left.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example F — out-of-distribution celebrity (music/film) ===\n"
    "Imagined image order: left=Brad Pitt, middle=Rihanna, right=Messi.\n"
    "Instruction: Place the coke on Rihanna.\n"
    "Reasoning: Images show Brad Pitt (left, American actor), "
    "Rihanna (middle, singer), Messi (right, soccer player). "
    "Rihanna is in the middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example G — out-of-distribution celebrity (royalty/sports) ===\n"
    "Imagined image order: left=Serena Williams, middle=Queen Elizabeth, right=LeBron James.\n"
    "Instruction: Place the coke on LeBron James.\n"
    "Reasoning: Images show Serena Williams (left, tennis player), "
    "Queen Elizabeth (middle, late British monarch), LeBron James "
    "(right, NBA basketball player). LeBron is on the right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== END OF EXAMPLES ===\n"
    "\n"
    "Now apply the same style of reasoning to the ACTUAL image you are "
    "looking at and the ACTUAL instruction above. Identify the real "
    "people in the three images using the camera-geometry rule "
    "(remember: middle = highest in the frame), then interpret the "
    "instruction, then conclude which position (left, middle, or right) "
    "is the target.\n"
    "\n"
    "Write a thorough reasoning of about 200 words. Walk through it "
    "step by step: identify each person using the camera-geometry rule, "
    "restate the instruction, list candidate images, rule out the wrong "
    "ones, and conclude with the target POSITION (left, middle, or "
    "right) — NOT a name. End with a single line:\n"
    "  Conclusion: <left|middle|right>.\n"
    "The last word must be a position word, not a name."
)

_FINAL_PROMPT = (
    "CAMERA GEOMETRY (very important):\n"
    "The camera looks down and forward at an angle, so the three images "
    "form an ARC, not a straight horizontal line:\n"
    "  - The LEFT image appears in the bottom-left area of the frame.\n"
    "  - The MIDDLE image is the one closest to the TOP of the frame "
    "(farther from the robot — perspective makes it appear higher up).\n"
    "  - The RIGHT image appears in the bottom-right area of the frame.\n"
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
    "  'Put the coke on the <POSITION> image.'\n"
    "\n"
    "Do NOT re-derive the answer from scratch. Do NOT re-identify the "
    "people. Do NOT use any name in your answer. The only valid "
    "position words are: left, middle, right. Copy whichever position "
    "the reasoning concluded with.\n"
    "\n"
    "STRICT OUTPUT RULES: reply with EXACTLY ONE of these three "
    "sentences verbatim, and absolutely nothing else (no preamble, no "
    "explanation, no extra punctuation):\n"
    "  Put the coke on the left image.\n"
    "  Put the coke on the middle image.\n"
    "  Put the coke on the right image.\n"
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

        task = next((ln.strip() for ln in reversed(final_raw.splitlines()) if ln.strip()),
                    "Put the coke on the middle image.")
        if task and task[0].islower():
            task = task[0].upper() + task[1:]
        print(f"[VLM] task_spec={task_spec!r} → task={task!r}", file=sys.stderr)
        return task
