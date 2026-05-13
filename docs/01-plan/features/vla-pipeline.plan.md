# Plan: vla-pipeline

## Overview

Build a single end-to-end VLA (Vision-Language-Action) model for three pick-and-place tasks.
One model takes a camera frame + language instruction and directly outputs 6-DOF joint commands.
No separate VLM reasoning step — language conditioning is handled inside the VLA backbone.

---

## Evaluation Tasks

### Eval 1 — Direct color pick-and-place (50 pts)
- **Setup**: 3 bowls in fixed order: **blue (left), red (middle), green (right)**
- **Prompt**: "Put the banana in the [blue/red/green] colored bowl."
- **VLA role**: language instruction + image → directly output correct pick-and-place trajectory
- **Time limit**: 20 seconds, 9 rollouts (3 per color)

### Eval 2 — Compositional instructions (50 pts)
- **Setup**: same 3 bowls but **colors in varying/unknown order**
- **Prompt examples**:
  - "Put the banana into the 2nd bowl from the left from the robot perspective"
  - "Put the banana into the bowl on the right of the red bowl from the robot perspective"
  - "Put the banana into the bowl that is not green and not blue."
- **VLA role**: visual grounding + compositional language reasoning → correct trajectory
- **Why harder**: model must visually understand the scene layout and reason about it
- **Time limit**: 20 seconds

### Eval 3 — Celebrity image pick-and-place (50 pts)
- **Setup**: 3 DIN A5 celebrity portraits in semicircle, coke can in the middle
- **Celebrities**: Taylor Swift, Barack Obama, Yann LeCun (known); OOD celebrities (unknown)
- **Prompt**: "Place the coke on [celebrity name]"
- **VLA role**: face recognition in scene + instruction → correct trajectory
- **Time limit**: 20 seconds, 9 rollouts

---

## Architecture

```
camera frame + language instruction
        │
        ▼
  VLA model  (single model — no separate VLM step)
  ┌──────────────────────────────────┐
  │  Vision encoder (pretrained)     │
  │  Language model (pretrained)     │
  │  Action head (fine-tuned)        │
  └──────────────────────────────────┘
        │
        ▼
  6-DOF joint commands → SO101 Robot
```

**Key constraints:**
- No cloud APIs, no YOLO, no external VLMs at inference time — all runs locally
- Must use a pretrained VLM backbone (fine-tuned on teleop data)
- CLI interface: `python main.py --task "..."` → press Enter → robot executes
- Single model = smaller parameter count → bonus points

---

## Bowl Layout

| Eval | Bowl order | VLA needs to... |
|------|-----------|-----------------|
| Eval 1 | **Fixed**: blue (left), red (middle), green (right) | Map color in instruction → trajectory |
| Eval 2 | **Varying** — unknown at inference time | See scene + reason spatially → trajectory |
| Eval 3 | Celebrity portraits in semicircle | Recognise face → trajectory |

---

## Training Data

| Dataset | Use |
|---------|-----|
| `ehalicki/eth-rl-so101-may4` | Teleop data (~32K frames, 3 bowls, 3 colors) |
| Additional teleop | More episodes: different positions/lighting/backgrounds |
| Synthetic augmentation | Random illumination, color jitter, ±5cm position jitter |

**Training sample format:**
```
input:  (camera_frame, task_instruction_string, current_joint_state)
output: action [6-DOF joint commands]
```

No `position_id` or `bowl_color` label needed — the raw task string is the language conditioning.

**Data quality issues:**
- task_indices 3–7 are corrupted (arrow keys, typos) but bowl intent is recoverable via regex
- task_index 6 (`"[D11"`) has no intent — dropped
- task_indices 0 and 8 both mean green bowl (different spelling) — kept as-is (same task string either way)

---

## Implementation Plan

### Phase 1 — Data pipeline (`src/train.py`)
- [x] Load dataset, map task_index → clean task string
- [x] Load video frames aligned with action frames (PyAV, 256×256)
- [x] Image augmentations: color jitter, random illumination
- [ ] Switch dataset to yield `(image, state, task_string, action)` instead of `position_id`

### Phase 2 — VLA model (`src/policy.py`)
- [ ] Replace ResNet-18 + MLP with SmolVLA / fine-tuned SmolVLM2 + action head
- [ ] Overfit test on single episode
- [ ] Full training on all clean teleop data
- [ ] Evaluate on held-out episodes

### Phase 3 — CLI entry point (`main.py`)
- [ ] `python main.py --task "Put the banana in the blue bowl."`
- [ ] Capture frame → VLA(frame, task) → actions → robot
- [ ] 20-second timeout enforcement

---

## Constraints & Requirements

| Requirement | Status |
|-------------|--------|
| Pretrained VLM backbone | ✅ SmolVLA / SmolVLM2 |
| Single model (no separate VLM + policy) | ✅ by design |
| Fine-tuned on own compute | ⏳ |
| No cloud APIs at inference | ✅ |
| CLI interface | ⏳ `main.py` |
| Image resolution 256×256 | ✅ |
| Single camera only | ✅ wrist_camera |

---

## Starting Points (from TAs)

- **SmolVLA**, **Smol-0-VLA**, **FlowerVLA**, **TinyVLA** — reuse OSS training code
- Batch size 64 on 1×A100 80GB with gradient accumulation 2

---

## Bonus (smallest model)

Single VLA model is inherently more compact than VLM + separate policy.
- Eval 1 bonus: 10/9/8 pts
- Eval 2 bonus: 20/18/16 pts
- Eval 3 bonus: 20/18/16 pts

---

## Files

| File | Role |
|------|------|
| `main.py` | CLI entry point |
| `src/policy.py` | VLA model definition + inference loop |
| `src/train.py` | Dataset loading + VLA training loop |
| `src/record.py` | Teleop data recording |
| `src/teleop.py` | Teleoperation |

## References

- Dataset: `ehalicki/eth-rl-so101-may4`
- Project spec: `docs/P1-VLA.pdf`
- Suggested architectures: SmolVLA, Smol-0-VLA, FlowerVLA, TinyVLA
