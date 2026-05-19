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
    "Reasoning: I begin by identifying each bowl. Looking at the frame, "
    "the bowl in the bottom-left is blue, the bowl highest in the frame "
    "(which is the middle position because of the camera angle — the "
    "middle bowl is farther from the robot and appears higher) is red, "
    "and the bowl in the bottom-right is green. Sanity check: I have "
    "one of each color (blue, red, green) — good. Now I interpret the "
    "instruction. The user wants the banana placed in the red bowl, a "
    "direct color lookup. From my map: red is at the middle position. "
    "I rule out the alternatives: left (blue) is the wrong color, "
    "right (green) is the wrong color, only middle (red) matches.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example B — ordinal (leftmost) ===\n"
    "Imagined bowl order: left=green, middle=blue, right=red.\n"
    "Instruction: Put the banana in the leftmost bowl.\n"
    "Reasoning: First I identify each bowl. The bowl in the bottom-left "
    "of the frame is green, the bowl highest in the frame (middle "
    "position due to camera perspective) is blue, and the bowl in the "
    "bottom-right is red. All three colors red/green/blue are present, "
    "good. Now I interpret the instruction. 'Leftmost' is a purely "
    "positional descriptor — it refers to the bowl furthest to the left "
    "from the robot's perspective, irrespective of color. That bowl is "
    "at the left position. Its color happens to be green, but the color "
    "is irrelevant; only the position matters here. Therefore the "
    "target is the left position.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example C — ordinal (Nth from the left) ===\n"
    "Imagined bowl order: left=red, middle=green, right=blue.\n"
    "Instruction: Put the banana in the 2nd bowl from the left.\n"
    "Reasoning: First, map the bowls. The bowl in the bottom-left of "
    "the frame is red, the bowl highest in the frame (middle position) "
    "is green, and the bowl in the bottom-right is blue. All three "
    "colors present. Now I interpret the instruction: count bowls "
    "starting from the left side. Going left-to-right: 1st bowl = left "
    "position (red), 2nd bowl = middle position (green), 3rd bowl = "
    "right position (blue). The instruction asks for the 2nd bowl from "
    "the left, which is the middle position holding the green bowl. "
    "Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example D — spatial relative (right of <color>) ===\n"
    "Imagined bowl order: left=blue, middle=red, right=green.\n"
    "Instruction: Put the banana in the bowl on the right of the red bowl.\n"
    "Reasoning: First I identify the bowls. Left in frame = blue, "
    "middle (highest in frame) = red, right in frame = green. All three "
    "colors are present. Now interpret: I need to find the red bowl, "
    "then move one position to its right. The red bowl is at the middle "
    "position. One step to the right of middle is the right position, "
    "which holds the green bowl. So the target is the right position. "
    "Color of the target doesn't matter here, what matters is the "
    "positional relationship.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example E — spatial relative (left of <color>) ===\n"
    "Imagined bowl order: left=green, middle=red, right=blue.\n"
    "Instruction: Put the banana in the bowl to the left of the red bowl.\n"
    "Reasoning: Map the bowls. Left in frame = green, middle (highest "
    "in frame) = red, right in frame = blue. All three colors present. "
    "Interpret: find the red bowl, then move one position to its left. "
    "Red is at the middle position. One step left of middle is the "
    "left position, which holds the green bowl. So the target is the "
    "left position.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example F — negation (not X and not Y) ===\n"
    "Imagined bowl order: left=blue, middle=green, right=red.\n"
    "Instruction: Put the banana in the bowl that is not green and not blue.\n"
    "Reasoning: Identify the bowls. Left in frame = blue, middle "
    "(highest in frame) = green, right in frame = red. All three colors "
    "present. Interpret: this is a negation. There are exactly three "
    "colors — red, green, blue. The instruction excludes green AND "
    "blue, which leaves only red as the valid color. Now I find the "
    "red bowl. From my map: red is at the right position. Sanity "
    "check by elimination: left = blue (excluded), middle = green "
    "(excluded), right = red (the only one not excluded). Therefore "
    "the target is right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example G — negation (neither X nor Y) ===\n"
    "Imagined bowl order: left=red, middle=blue, right=green.\n"
    "Instruction: Put the banana in the bowl that is neither red nor green.\n"
    "Reasoning: Map bowls to colors. Left in frame = red, middle "
    "(highest in frame) = blue, right in frame = green. All three "
    "colors present. Interpret: 'neither X nor Y' is a negation "
    "meaning 'not X and not Y'. The instruction excludes red and "
    "green. Since there are only three possible colors and two are "
    "excluded, the target color must be the third color: blue. Blue "
    "is at the middle position. Cross-check by elimination: left = "
    "red (excluded), middle = blue (valid ✓), right = green "
    "(excluded). Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example H — ordinal (rightmost) ===\n"
    "Imagined bowl order: left=green, middle=blue, right=red.\n"
    "Instruction: Put the banana in the rightmost bowl.\n"
    "Reasoning: Identify the bowls first. Left in frame = green, "
    "middle (highest in frame) = blue, right in frame = red. All three "
    "colors present. Interpret: 'rightmost' is a positional descriptor "
    "meaning the bowl furthest to the right from the robot's "
    "perspective. That bowl is at the right position. The color of "
    "that bowl is red, but the color is irrelevant here — only the "
    "position matters. Therefore the target is the right position.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example I — between two colors ===\n"
    "Imagined bowl order: left=red, middle=green, right=blue.\n"
    "Instruction: Put the banana in the bowl between the red bowl and the blue bowl.\n"
    "Reasoning: Identify each bowl. Left in frame = red, middle "
    "(highest in frame) = green, right in frame = blue. All three "
    "colors present. Interpret: the instruction names two colored "
    "bowls and asks for the bowl between them. The red bowl is at the "
    "left position, the blue bowl is at the right position. The bowl "
    "spatially between left and right is the middle position, which "
    "holds the green bowl. Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example J — direct color lookup, different ordering ===\n"
    "Imagined bowl order: left=blue, middle=red, right=green.\n"
    "Instruction: Put the banana in the green bowl.\n"
    "Reasoning: Map the bowls. Left in frame = blue, middle (highest "
    "in frame) = red, right in frame = green. Good, all three colors "
    "present. Interpret: this is a direct color lookup — I just need "
    "to find the green bowl. Green is at the right position. Verify by "
    "elimination: left (blue) is wrong color, middle (red) is wrong "
    "color, right (green) is the match. Therefore the target is right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example K — spatial relative, color landmark on the side ===\n"
    "Imagined bowl order: left=green, middle=red, right=blue.\n"
    "Instruction: Put the banana in the bowl directly to the right of the green bowl.\n"
    "Reasoning: Identify the bowls. Left in frame = green, middle "
    "(highest in frame) = red, right in frame = blue. All three colors "
    "present. Interpret: I need to find the green bowl, then move one "
    "position to its right. Green is at the left position. One step "
    "right of left is the middle position, which holds the red bowl. "
    "Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example L — color landmark on the right side, instruction \"left of\" ===\n"
    "Imagined bowl order: left=blue, middle=green, right=red.\n"
    "Instruction: Put the banana in the bowl to the left of the red bowl.\n"
    "Reasoning: Map the bowls. Left in frame = blue, middle (highest "
    "in frame) = green, right in frame = red. All three colors present. "
    "Interpret: find the red bowl, then move one position to its left. "
    "Red is at the right position. One step left of right is the "
    "middle position, which holds the green bowl. Therefore the target "
    "is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example M — ordinal counted from the right ===\n"
    "Imagined bowl order: left=red, middle=blue, right=green.\n"
    "Instruction: Put the banana in the 2nd bowl from the right.\n"
    "Reasoning: Map the bowls. Left in frame = red, middle (highest in "
    "frame) = blue, right in frame = green. All three colors present. "
    "Interpret: count starting from the right side. Going right-to-"
    "left: 1st = right position (green), 2nd = middle position (blue), "
    "3rd = left position (red). The instruction asks for the 2nd from "
    "the right, which is the middle position. Therefore the target is "
    "middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example N — color landmark adjacency (right of blue) ===\n"
    "Imagined bowl order: left=green, middle=blue, right=red.\n"
    "Instruction: Put the banana in the bowl to the right of the blue bowl.\n"
    "Reasoning: Identify bowls. Left in frame = green, middle (highest "
    "in frame) = blue, right in frame = red. All three colors present. "
    "Interpret: find the blue bowl, then move one position to its "
    "right. Blue is at the middle position. One step right of middle "
    "is the right position, which holds the red bowl. Therefore the "
    "target is right.\n"
    "Conclusion: right.\n"
    "\n"
    "=== Example O — single-color negation (not blue) ===\n"
    "Imagined bowl order: left=green, middle=blue, right=red.\n"
    "Instruction: Put the banana in the bowl that is not blue and not red.\n"
    "Reasoning: Map bowls. Left = green, middle (highest) = blue, "
    "right = red. All three colors present. Interpret: this is a "
    "two-color negation. The instruction excludes blue and red, "
    "leaving green as the only valid color. The green bowl is at the "
    "left position. Cross-check: left = green (valid ✓), middle = blue "
    "(excluded), right = red (excluded). Therefore the target is left.\n"
    "Conclusion: left.\n"
    "\n"
    "=== Example P — middle ordinal phrased differently (center / in the middle) ===\n"
    "Imagined bowl order: left=red, middle=green, right=blue.\n"
    "Instruction: Put the banana in the center bowl.\n"
    "Reasoning: Identify bowls. Left = red, middle (highest in frame) "
    "= green, right = blue. All three colors present. Interpret: "
    "'center' is a positional descriptor synonymous with 'middle'. It "
    "refers to the bowl that sits between the leftmost and rightmost "
    "ones — the middle position. The color of the bowl at that "
    "position (green) is irrelevant. Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example Q — color lookup with phrasing variation ===\n"
    "Imagined bowl order: left=red, middle=blue, right=green.\n"
    "Instruction: Place the banana into the blue bowl.\n"
    "Reasoning: Identify bowls. Left = red, middle (highest in frame) "
    "= blue, right = green. All three colors present. Interpret: this "
    "is a direct color lookup; the verb 'place ... into' is the same "
    "as 'put ... in'. Blue is at the middle position. Verify: left = "
    "red (wrong), middle = blue (match ✓), right = green (wrong). "
    "Therefore the target is middle.\n"
    "Conclusion: middle.\n"
    "\n"
    "=== Example R — chained spatial reasoning (next to + color) ===\n"
    "Imagined bowl order: left=red, middle=green, right=blue.\n"
    "Instruction: Put the banana in the bowl next to the red bowl.\n"
    "Reasoning: Map bowls. Left = red, middle (highest in frame) = "
    "green, right = blue. All three colors present. Interpret: 'next "
    "to' is ambiguous when there are two neighbors, but here the red "
    "bowl is at the LEFT edge, so it only has ONE neighbor — the "
    "bowl immediately to its right. That neighbor is at the middle "
    "position, holding the green bowl. Therefore the target is "
    "middle. (If the color landmark had been in the middle, 'next to' "
    "would be ambiguous and we'd default to one side — typically the "
    "right neighbor.)\n"
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
    "IMPORTANT — write a LONG, DETAILED multi-paragraph reasoning. The "
    "worked examples above are intentionally brief; your own reasoning "
    "should be MORE thorough than the examples. Aim for at least 200 "
    "words. Walk through every step very carefully: describe the scene "
    "(where the banana is, where each bowl sits in the frame), use the "
    "camera-geometry rule to identify which bowl is in the middle "
    "position (the one HIGHEST in the frame, not the geometric center), "
    "name each bowl's color one at a time and justify the call ('the "
    "bowl in the bottom-left of the frame looks XXX because...'), "
    "double-check by listing the three colors you found and verifying "
    "all of red/green/blue are present, restate the instruction in your "
    "own words, list every candidate bowl, explicitly rule out the "
    "wrong ones step by step with reasoning, and finally state your "
    "conclusion. Do NOT skip the bowl-identification step even if the "
    "instruction looks easy. Do NOT write the final canonical answer "
    "sentence here — that comes in a follow-up message.\n"
    "\n"
    "CRITICAL — HOW TO END YOUR REASONING:\n"
    "Your reasoning MUST end with a single line of the form:\n"
    "  Conclusion: <position>.\n"
    "where <position> is EXACTLY one of these three words: left, "
    "middle, or right. It MUST be a POSITION word, NOT a color word.\n"
    "\n"
    "DO NOT end with a color. The following are WRONG and will cause "
    "the system to fail:\n"
    "  ✗ 'Conclusion: red.'   (red is a color, not a position)\n"
    "  ✗ 'Conclusion: the red bowl.'\n"
    "  ✗ 'Conclusion: place it in the green bowl.'\n"
    "  ✗ 'The target is the blue bowl.'\n"
    "\n"
    "The following are CORRECT:\n"
    "  ✓ 'Conclusion: left.'\n"
    "  ✓ 'Conclusion: middle.'\n"
    "  ✓ 'Conclusion: right.'\n"
    "\n"
    "If you have identified the target by color (e.g. 'the target color "
    "is red'), you MUST then translate that color to a position using "
    "your bowl map (e.g. 'red is at the middle position, so the target "
    "position is middle') and end with 'Conclusion: middle.'. The very "
    "last word before the period MUST be left, middle, or right."
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
