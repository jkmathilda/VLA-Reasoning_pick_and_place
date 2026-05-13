# Design: vla-pipeline

> Phase: Design | Depends on: `docs/01-plan/features/vla-pipeline.plan.md`

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                         │
│  $ python main.py --task "Put the banana in the blue bowl."  │
└─────────────────────────┬────────────────────────────────────┘
                          │  task string + camera frame
              ┌───────────▼───────────┐
              │      VLAPolicy        │  src/policy.py
              │                       │
              │  vision encoder       │  ← pretrained (SmolVLM2)
              │  language model       │  ← pretrained (SmolVLM2)
              │  action head          │  ← fine-tuned on teleop data
              │                       │
              │  (frame, task, state) │
              │        → action [6]   │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │     SO101 Robot       │  via LeRobot
              │     6-DOF commands    │
              └───────────────────────┘
```

No separate VLM router. Language reasoning and visual grounding happen inside the VLA backbone.

---

## 2. File Structure

```
VLA-Reasoning_pick_and_place/
├── main.py                   # CLI entry point
├── src/
│   ├── policy.py             # VLA model + training + inference loop
│   ├── train.py              # Dataset loading + training script
│   ├── record.py             # Teleop data recording (existing)
│   └── teleop.py             # Teleoperation (existing)
├── checkpoints/
│   └── policy.pt             # Trained VLA weights
├── docs/
│   ├── P1-VLA.pdf
│   ├── 01-plan/features/vla-pipeline.plan.md
│   └── 02-design/features/vla-pipeline.design.md
└── requirements.txt
```

**Removed:** `src/vlm_router.py`, `src/vlm.py`, `src/vlm2.py` (no longer needed at inference)

---

## 3. Component Designs

### 3.1 `main.py` — CLI Entry Point

```python
# Usage:
#   python main.py --task "Put the banana in the blue bowl."
#   python main.py --task "Put the banana into the 2nd bowl from the left."

import argparse
from src.policy import VLAPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
args = parser.parse_args()

policy = VLAPolicy.load("checkpoints/policy.pt")
policy.run(task=args.task, timeout=18.0)
# captures frames, sends joint commands to robot until timeout
```

**Constraints:**
- Total inference budget: < 20 seconds
- No network calls at runtime

---

### 3.2 `src/policy.py` — VLA Model

**Single model: language + vision → action.**

**Architecture (SmolVLM2 backbone + action head):**

```
camera frame [3, 256, 256]
        │
   Vision encoder (SmolVLM2, pretrained, frozen or lightly fine-tuned)
        │  [visual tokens]
        │
task instruction tokens (SmolVLM2 tokenizer + LM, pretrained)
        │  [text tokens]
        │
   ┌────┴─────────────────┐
   │  Cross-attention /   │
   │  token concatenation │
   └────────┬─────────────┘
            │
   state [6] ──── concat
            │
      Action head MLP
      (hidden_dim → 256 → 6)
            │
      action [6-DOF]
```

**Class interface:**

```python
class VLAPolicy(nn.Module):
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                 hidden_dim=256, freeze_backbone=True):
        # loads SmolVLM2 processor + model
        # adds action head on top

    def forward(self, image, task_tokens, state) -> Tensor:
        # returns predicted action [B, 6]

    def run(self, task: str, timeout: float = 18.0):
        # inference loop:
        #   while not done and elapsed < timeout:
        #     frame = robot.camera.capture()
        #     state = robot.get_joints()
        #     action = self(frame, task, state)
        #     robot.set_joints(action)

    def save(self, path: str): ...

    @classmethod
    def load(cls, path: str) -> "VLAPolicy": ...
```

**Starting point:** SmolVLA or Smol-0-VLA repo — fork their training loop and swap in this backbone.

---

### 3.3 `src/train.py` — Training Loop

**Data pipeline:**

```
HF dataset "ehalicki/eth-rl-so101-may4"
    │
    ├─ data/chunk-*/file-*.parquet  →  action, state, task_index, timestamp
    └─ video files (wrist_camera)   →  frames per episode (256×256)
         │
         ▼
    build_task_index_to_bowl()   →  task_index → bowl color (for data cleaning)
    build_task_index_to_string() →  task_index → canonical task string
         │
         ▼
    VLAPolicyDataset
      __getitem__(idx):
        - load frame from video at timestamp  (resize to 256×256, augment)
        - return (image, state, task_string, action)
```

**Training sample:**
```python
image:       Tensor [3, 256, 256]   # camera frame
state:       Tensor [6]             # current joint positions
task_string: str                    # "Put the banana in the blue bowl."
action:      Tensor [6]             # target joint positions
```

**Training config:**

| Param | Value |
|-------|-------|
| Image size | 256 × 256 |
| Batch size | 64 (gradient accumulation if needed) |
| Optimizer | AdamW, lr=1e-4 |
| Loss | MSE on 6-DOF actions |
| Backbone | Frozen during initial training; unfreeze last layers later |
| Augmentations | ColorJitter (brightness/contrast/saturation), random flip |

---

### 3.4 Task String Mapping

The teleop dataset has corrupted task_index entries. We clean them to canonical strings:

```python
# task_index → canonical task string (for language conditioning)
TASK_INDEX_TO_STRING = {
    0: "Put the banana in the green bowl.",
    1: "Put the banana in the blue bowl.",
    2: "Put the banana in the red bowl.",
    3: "Put the banana in the red bowl.",    # corrupted typo → same intent
    4: "Put the banana in the blue bowl.",   # corrupted [D → same intent
    5: "Put the banana in the green bowl.",  # corrupted [D → same intent
    6: None,                                  # pure garbage → dropped
    7: "Put the banana in the blue bowl.",   # corrupted [C → same intent
    8: "Put the banana in the green bowl.",  # alternate spelling → same intent
}
```

This replaces the `BOWL_TO_POSITION` mapping from the two-model design — no `position_id` needed.

---

## 4. Data Flow Diagrams

**Training:**
```
(frame, "Put the banana in the blue bowl.", state) → VLAPolicy → MSE loss vs action
```

**Eval 1 inference:**
```
"Put the banana in the blue bowl." + frame
        ↓
   VLAPolicy (language understood via LM backbone)
        ↓
   [j1, j2, j3, j4, j5, j6]  →  robot
```

**Eval 2 inference:**
```
"Put the banana into the 2nd bowl from the left." + frame
        ↓
   VLAPolicy (VLM backbone sees scene + reasons compositionally)
        ↓
   [j1, j2, j3, j4, j5, j6]  →  robot
```

No intermediate step. The model resolves language and vision together.

---

## 5. Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Single model | VLA (no separate VLM router) | TA feedback; simpler; bonus for smallest model |
| Backbone | SmolVLM2-500M (vision + language) | Pretrained, small, already working in codebase |
| Action head | MLP on top of backbone output | Simple, fast to train |
| Language input | Raw task string (not position_id) | Works for all evals including compositional |
| Backbone training | Frozen initially, unfreeze later | Compute budget; avoid catastrophic forgetting |
| Image resolution | 256×256 | TA recommendation |
| Loss | MSE on raw actions | Simple BC baseline |
| Frame loading | PyAV (`av`) | No torchcodec dependency |

---

## 6. Open Questions

- [ ] Use SmolVLA repo directly (fork + adapt) or implement action head from scratch on SmolVLM2?
- [ ] Does teleop bowl order (blue=left, red=middle, green=right) match Eval 1 fixed order?
- [ ] Freeze full backbone or only freeze vision encoder during training?
- [ ] Camera setup: wrist camera (capture start frame) or shoulder mount?

---

## 7. Implementation Order

1. `src/train.py` — update `VLAPolicyDataset` to yield `task_string` instead of `position_id`
2. `src/policy.py` — replace ResNet-18 + MLP with `VLAPolicy` (SmolVLM2 + action head)
3. `main.py` — wire CLI → `VLAPolicy.run(task, timeout)`
4. Overfit test on single episode, then full training
