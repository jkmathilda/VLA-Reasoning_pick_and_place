# Design: vla-pipeline

> Phase: Design | Depends on: `docs/01-plan/features/vla-pipeline.plan.md`

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                        │
│  $ python main.py --task "Put the banana in the blue bowl." │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    VLMRouter        │  src/vlm_router.py
              │  task + frame in    │
              │  position out       │
              │  {left|middle|right}│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   ActionPolicy      │  src/policy.py
              │  (state, image,     │
              │   position) → actions│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   SO101 Robot       │  via LeRobot
              │   6-DOF commands    │
              └─────────────────────┘
```

---

## 2. File Structure

```
VLA-Reasoning_pick_and_place/
├── main.py                   # CLI entry point
├── src/
│   ├── vlm_router.py         # VLM inference + position resolution (NEW)
│   ├── policy.py             # BC action policy model + inference (NEW)
│   ├── train.py              # Dataset loading + policy training
│   ├── vlm2.py               # SmolVLM2 video inference (existing)
│   ├── record.py             # Teleop data recording (existing)
│   └── teleop.py             # Teleoperation (existing)
├── docs/
│   ├── P1-VLA.pdf
│   ├── 01-plan/features/vla-pipeline.plan.md
│   └── 02-design/features/vla-pipeline.design.md
└── requirements.txt
```

---

## 3. Component Designs

### 3.1 `main.py` — CLI Entry Point

```python
# Usage:
#   python main.py --task "Put the banana in the blue bowl."
#   python main.py --task "Put the banana into the 2nd bowl from the left."

import argparse
from src.vlm_router import VLMRouter
from src.policy import ActionPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
args = parser.parse_args()

router = VLMRouter()
policy = ActionPolicy.load("checkpoints/policy.pt")

# 1. Capture frame from robot camera
frame = capture_frame()  # via LeRobot camera API

# 2. VLM resolves instruction → position
position = router.resolve(task=args.task, frame=frame)
# position ∈ {"left", "middle", "right"}

# 3. Policy executes pick-and-place to target position
policy.run(position=position)  # sends actions to robot, 20s timeout
```

**Constraints:**
- Total inference budget: < 20 seconds
- No network calls at runtime

---

### 3.2 `src/vlm_router.py` — VLM Position Resolver

**Responsibility:** given a task string + camera frame, return `position ∈ {left, middle, right}`.

```python
class VLMRouter:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        # load processor + model (reuse src/vlm2.py pattern)

    def resolve(self, task: str, frame: PIL.Image) -> str:
        # Returns "left", "middle", or "right"
```

**Eval 1 prompt** (fixed layout, color extraction only):
```
System: The bowl layout is fixed: blue=left, red=middle, green=right.
User:   Instruction: "{task}"
        Which position is the target bowl?
        Answer with exactly one word: left, middle, or right.
```

**Eval 2 prompt** (varying layout, visual grounding required):
```
System: You are a robot assistant. Look at the camera image carefully.
User:   [image attached]
        There are three bowls in front of the robot (left, middle, right).
        First identify the color of each bowl by position.
        Then, given the instruction: "{task}"
        Which position is the target bowl?
        Answer with exactly one word: left, middle, or right.
```

**Output parsing:**
```python
response = generate(prompt)
for word in ["left", "middle", "right"]:
    if word in response.lower():
        return word
raise ValueError(f"VLM returned unrecognised position: {response}")
```

**Eval routing logic:**
- If task contains a color word (`blue|red|green`) AND no spatial qualifier
  (`left|right|middle|2nd|first|second|third|not`) → use Eval 1 prompt (faster, no image needed)
- Otherwise → use Eval 2 prompt with camera frame

---

### 3.3 `src/policy.py` — Behavior Cloning Action Policy

**Input:** `(image: Tensor[3,256,256], state: Tensor[6], position_id: int)` → `action: Tensor[6]`

**Architecture:**

```
image (256×256×3)
    │
    ▼
Image Encoder (frozen SmolVLM2 vision encoder OR lightweight CNN)
    │  [512-dim embedding]
    │
state (6-DOF)  ──────┐
position_id (0/1/2) ──┤  concat
                      ▼
              MLP (512+6+3 → 256 → 256 → 6)
                      │
                      ▼
              predicted action (6-DOF)
```

**Position encoding:** one-hot or learned embedding for `{left=0, middle=1, right=2}`.

**Training objective:** MSE loss on 6-DOF joint actions.

```python
class ActionPolicy(nn.Module):
    def __init__(self, image_encoder, hidden_dim=256):
        ...
    def forward(self, image, state, position_id) -> Tensor:
        ...
    def run(self, position: str, timeout: float = 18.0):
        # inference loop: capture frame → predict action → send to robot
        ...
    @classmethod
    def load(cls, path: str) -> "ActionPolicy":
        ...
```

---

### 3.4 `src/train.py` — Training Loop

**Data pipeline:**

```
HF dataset "ehalicki/eth-rl-so101-may4"
    │
    ├─ data/chunk-*/file-*.parquet  →  action, state, task_index, frame_index
    └─ video files (wrist_camera)   →  frames per episode
         │
         ▼
    build_task_index_to_bowl()  →  task_index → bowl color
    BOWL_TO_POSITION            →  bowl color → position {left/middle/right}
         │
         ▼
    VLAPolicyDataset
      __getitem__(idx):
        - load frame from video at frame_index  (resize to 256×256)
        - return (image, state, position_id, action)
```

**Training config:**
| Param | Value |
|-------|-------|
| Image size | 256 × 256 |
| Batch size | 64 (gradient accumulation if needed) |
| Optimizer | AdamW, lr=1e-4 |
| Loss | MSE on 6-DOF actions |
| Epochs | until validation loss converges |
| Augmentations | random brightness/contrast, ±5cm position jitter |

**Dataset class:**
```python
class VLAPolicyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_root: str):
        # df has columns: action, observation.state, position, episode_index, frame_index
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_frame(row["episode_index"], row["frame_index"])
        image = transforms.Resize((256, 256))(image)
        state = torch.tensor(row["observation.state"], dtype=torch.float32)
        position_id = POSITION_TO_ID[row["position"]]
        action = torch.tensor(row["action"], dtype=torch.float32)
        return image, state, position_id, action
```

---

## 4. Data Flow Diagram

```
Eval 1 inference:
  "Put the banana in the blue bowl."
        │
        ▼ VLMRouter (Eval 1 prompt, no image)
        "left"
        │
        ▼ ActionPolicy(image, state, position_id=0)
        [j1, j2, j3, j4, j5, j6]  →  robot

Eval 2 inference:
  "Put the banana into the 2nd bowl from the left." + camera frame
        │
        ▼ VLMRouter (Eval 2 prompt, image attached)
        SmolVLM2 sees: blue=right, red=left, green=middle
        Resolves "2nd from left" → "red" → "left"
        "left"
        │
        ▼ ActionPolicy(image, state, position_id=0)
        [j1, j2, j3, j4, j5, j6]  →  robot
```

---

## 5. Position ID Mapping

```python
POSITION_TO_ID = {"left": 0, "middle": 1, "right": 2}
ID_TO_POSITION = {0: "left", 1: "middle", 2: "right"}
```

---

## 6. Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Policy conditioning | position (left/middle/right) not color | Reuses same policy for Eval 1 & 2 |
| Image encoder | Frozen SmolVLM2 vision encoder or lightweight CNN | Compute budget; avoids full VLM at every step |
| VLM routing | Eval 1 prompt (fast) vs Eval 2 prompt (visual) | Avoid unnecessary image processing for simple color tasks |
| Image resolution | 256×256 | TA recommendation; sufficient for bowl discrimination |
| Loss function | MSE on raw actions | Simple BC baseline; can switch to chunked actions (ACT) later |
| Video frame loading | PyAV (`av`) | Already installed, no torchcodec dependency |

---

## 7. Open Questions Before Implementation

- [ ] Are video files for `wrist_camera` available locally or need separate download?
- [ ] Does the teleop dataset bowl order (blue=left, red=middle, green=right) match the Eval 1 fixed order?
- [ ] Is SmolVLM2-500M reliable enough for Eval 2 visual grounding, or do we need a larger model?
- [ ] Use SmolVLM2 vision encoder as image encoder in policy, or train a separate CNN?

---

## 8. Implementation Order

1. `src/train.py` — add `VLAPolicyDataset`, video frame loading, training loop
2. `src/policy.py` — define `ActionPolicy`, train, overfit test on 1 episode
3. `src/vlm_router.py` — `VLMRouter` with both prompt strategies + output parser
4. `main.py` — wire CLI → `VLMRouter` → `ActionPolicy` → robot commands
