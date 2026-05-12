# Plan: vla-pipeline

## Overview

Build a vision-language-conditioned manipulation policy for three evaluation tasks of increasing difficulty.
The system must use a pretrained VLM backbone (fine-tuned), run fully locally at inference time, and
expose a CLI interface. Total: 150 pts + 50 bonus (smallest model).

---

## Evaluation Tasks

### Eval 1 — Direct color pick-and-place (50 pts)
- **Setup**: 3 bowls in fixed order: **blue (left), red (middle), green (right)**
- **Prompt**: "Put the banana in the [blue/red/green] colored bowl."
- **VLM role**: extract bowl color from prompt → map to fixed position
- **Policy role**: execute pick-and-place to the target bowl position
- **Time limit**: 20 seconds, 9 rollouts (3 per color)

### Eval 2 — Compositional instructions (50 pts)
- **Setup**: same 3 bowls but **colors in varying/unknown order** (the `?` diagram confirms this)
- **Prompt examples**:
  - "Put the banana into the 2nd bowl from the left from the robot perspective"
  - "Put the banana into the bowl on the right of the red bowl from the robot perspective"
  - "Put the banana into the bowl that is not green and not blue."
- **VLM role**:
  1. **Visual grounding**: inspect camera frame → identify which color bowl is at which position
  2. **Compositional reasoning**: apply instruction to the observed layout → resolve to target position
- **Why harder**: cannot hardcode bowl layout — VLM must actually see and understand the scene
- **Time limit**: 20 seconds

### Eval 3 — Celebrity image pick-and-place (50 pts)
- **Setup**: 3 DIN A5 celebrity portraits in semicircle, coke can in the middle
- **Celebrities**: Taylor Swift, Barack Obama, Yann LeCun (known); OOD celebrities (unknown)
- **Prompt**: "Place the coke on [celebrity name]"
- **VLM role**: visual recognition of celebrity face in image → identify correct position
- **Time limit**: 20 seconds, 9 rollouts

---

## Architecture

```
CLI input: task instruction string + camera frame
        │
        ▼
  VLM (SmolVLM2 or SmolVLA/TinyVLA)
  → always outputs: target_position ∈ {left, middle, right}
        │
        ▼
  Action Policy (BC, conditioned on position)
  one shared policy for Eval 1 & 2
        │
        ▼
  6-DOF Joint Commands → SO101 Robot
```

**Key constraint**: No cloud APIs, no YOLO, no external VLMs at inference time.
Everything must run locally. Cloud/large VLMs may be used to generate annotated training data.

---

## Bowl Layout

| Eval | Bowl order | VLM needs to... |
|------|-----------|-----------------|
| Eval 1 | **Fixed**: blue (left), red (middle), green (right) | Extract color from prompt only |
| Eval 2 | **Varying** — unknown at inference time | See the scene + reason about layout |
| Eval 3 | Celebrity portraits in semicircle | Identify face in image |

---

## VLM Strategy

The policy is always conditioned on **position** (`left` / `middle` / `right`), never on color.
This lets Eval 1 and Eval 2 share the same trained policy and the same teleop data.

| Eval | VLM input | VLM reasoning | Output |
|------|-----------|---------------|--------|
| Eval 1 | prompt only (color fixed, layout known) | color → fixed position lookup | `position ∈ {left, middle, right}` |
| Eval 2 | camera frame + prompt | visual grounding → identify bowl colors in scene → apply compositional logic | `position ∈ {left, middle, right}` |
| Eval 3 | camera frame + prompt | face recognition → identify celebrity position | `position ∈ {left, middle, right}` |

**Teleop data label mapping** (Eval 1 fixed order: blue=left, red=middle, green=right):
```
bowl color → position
  blue  → left   (task_index 0,1,4,7,8 after dedup)
  red   → middle (task_index 2,3)
  green → right  (task_index 0,5,8 — wait: see note)
```
> Note: the teleop data was collected with a specific physical setup. Verify that the recorded
> episodes match the fixed Eval 1 order (blue=left, red=middle, green=right) before remapping.

**Eval 1 VLM prompt**:
> "The bowl layout is fixed: blue on the left, red in the middle, green on the right.
> Instruction: '{task}'. Which position is the target? Answer: left, middle, or right."

**Eval 2 VLM prompt**:
> "Look at the camera image. Three bowls are in front of the robot.
> First, identify the color of the bowl on the left, middle, and right.
> Then, given the instruction '{task}', which position is the target?
> Answer: left, middle, or right."

---

## Data

| Dataset | Use |
|---------|-----|
| `ehalicki/eth-rl-so101-may4` | Existing teleop data (32K frames, 3 bowls) |
| Additional teleop | More episodes at different positions/lighting |
| Synthetic augmentation | Random illumination, ±5cm object position jitter |

**Data quality issues found**:
- task_indices 3–7 are corrupted (arrow keys captured, typos) but color is recoverable via regex
- task_index 6 ("``[D11``") has no color — dropped
- task_indices 0 and 8 are both green (different spelling) — merged

---

## Implementation Plan

### Phase 1 — Data pipeline (`src/train.py`)
- [x] Load dataset, map task_index → bowl color
- [ ] Load video frames (wrist_camera) aligned with action frames
- [ ] Resize to 256×256 (recommended by TAs)
- [ ] Add image augmentations: random illumination, color jitter

### Phase 2 — VLM module (`src/vlm2.py`)
- [x] SmolVLM2-500M inference working
- [ ] Prompt engineering for Eval 1/2 (color + spatial reasoning)
- [ ] Prompt engineering for Eval 3 (celebrity recognition)
- [ ] Output parsing: extract `bowl_color` or `position` from VLM response

### Phase 3 — Action policy (`src/policy.py`)
- [ ] BC policy: (image, state, target_color) → 6-DOF actions
- [ ] Overfit test on single episode
- [ ] Full training on all clean teleop data
- [ ] Evaluate on held-out episodes

### Phase 4 — CLI entry point (`main.py`)
- [ ] `python main.py --task "Put the banana in the blue bowl."`
- [ ] Runs VLM → resolves target → runs policy → sends commands to robot
- [ ] 20-second timeout enforcement

---

## Constraints & Requirements

| Requirement | Status |
|-------------|--------|
| Pretrained VLM backbone | ✅ SmolVLM2-500M |
| Fine-tuned / trained on own compute | ⏳ |
| No cloud APIs at inference | ✅ (all local) |
| CLI interface | ⏳ `main.py` |
| LeRobot data format | ✅ |
| Image resolution 256×256 | ⏳ |
| Single camera only | ✅ wrist_camera |

---

## Starting Points (from TAs)

- **FlowerVLA**, **SmolVLA**, **Smol-0-VLA**, **TinyVLA** — reuse OSS training code
- Batch size 64 on 1×A100 80GB with gradient accumulation 2

---

## Bonus (smallest model)

SmolVLM2-500M is already small. Further compression (quantization, smaller backbone) could win:
- Eval 1 bonus: 10/9/8 pts
- Eval 2 bonus: 20/18/16 pts
- Eval 3 bonus: 20/18/16 pts

---

## Files

| File | Role |
|------|------|
| `main.py` | CLI entry point |
| `src/vlm2.py` | VLM inference (SmolVLM2) |
| `src/train.py` | Dataset loading + policy training |
| `src/policy.py` | BC policy model |
| `src/record.py` | Teleop data recording |
| `src/teleop.py` | Teleoperation |

## References

- Dataset: `ehalicki/eth-rl-so101-may4`
- Project spec: `docs/P1-VLA.pdf`
- Suggested architectures: FlowerVLA, SmolVLA, Smol-0-VLA, TinyVLA
