# policy-inference Design Document

> **Summary**: Class interface and data flow for `VLAPolicy` — two-stage inference combining `SceneVLM` (scene understanding) and SmolVLA (motor control).
>
> **Project**: VLA-Reasoning Pick and Place
> **Author**: junkyunglee
> **Date**: 2026-05-18
> **Status**: Draft
> **Planning Doc**: [policy-inference.plan.md](../01-plan/features/policy-inference.plan.md)

---

## 1. Overview

### 1.1 Design Goals

- Clean two-file split: `src/vlm.py` handles scene understanding, `src/policy.py` handles robot execution
- `--task` becomes optional in `main.py` — when omitted, VLM observes the scene and generates the instruction
- 30 fps action loop with no blocking VLM calls inside the hot path

### 1.2 Design Principles

- **Single responsibility**: `SceneVLM` is pure inference (image → str); robot I/O lives only in `VLAPolicy`
- **One VLM call per episode**: run SceneVLM once at start, cache the grounded task for the whole episode
- **Fail loudly**: no silent fallbacks; raise on bad checkpoint, bad camera key, or unrecognizable VLM output

---

## 2. Architecture

### 2.1 Component Diagram

```
main.py
  │
  ├─ VLAPolicy.load(checkpoint)
  │       ├─ loads SmolVLA (lerobot PreTrainedPolicy)
  │       └─ loads SceneVLM (SmolVLM2-500M-Video-Instruct)
  │
  └─ VLAPolicy.run(task=None, timeout=18.0, move_to_home=True)
          │
          ├─ connect SO101Follower
          ├─ [optional] move to HOME_JOINTS
          │
          ├─ capture first frame
          ├─ task provided? ──Yes──▶ use as-is
          │        │
          │       No
          │        ▼
          │  SceneVLM.query(frame) → generated_task
          │
          ├─ 30 fps action loop (timeout seconds)
          │       ├─ observe() → frame
          │       ├─ preprocessor(obs + task)
          │       ├─ policy.select_action(obs) → action
          │       └─ robot.send_action(action)
          │
          └─ disconnect SO101Follower
```

### 2.2 Data Flow

```
Frame (H,W,C uint8 numpy)
  │
  │  [SceneVLM path — once per episode]
  ▼
SmolVLM2 processor → tokens → model.generate() → decoded str
  │
  ▼
task_str  (e.g. "Put the banana in the blue bowl.")
  │
  │  [SmolVLA path — every step, 30 fps]
  ▼
preprocessor: rename camera1, numpy→tensor float32 [0,1], normalize
  │
  ▼
SmolVLA.select_action(obs_dict) → action tensor (6,)
  │
  ▼
postprocessor: unnormalize
  │
  ▼
SO101Follower.send_action(action)
```

### 2.3 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| `VLAPolicy` | `lerobot` `PreTrainedPolicy`, `SO101Follower` | SmolVLA loading + robot control |
| `SceneVLM` | `transformers` `AutoProcessor`, `AutoModelForImageTextToText` | Scene description |
| `main.py` | `VLAPolicy` | CLI entry point |

---

## 3. Module Interfaces

### 3.1 `src/vlm.py` — `SceneVLM`

```python
MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

GROUNDING_PROMPT = (
    "You are controlling a robot arm. Look at the image of the workspace.\n"
    "The objects present are a banana and bowls colored blue, red, or green.\n"
    "Identify which task the robot should perform and state it in exactly one sentence:\n"
    "'Put the banana in the {color} bowl.'\n"
    "Reply with that sentence only."
)

class SceneVLM:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None): ...
    def query(self, image: np.ndarray) -> str:
        """
        Args:
            image: (H, W, C) uint8 numpy array, top-down RGB frame
        Returns:
            Task instruction string, e.g. "Put the banana in the blue bowl."
        Raises:
            ValueError: if model output doesn't contain a recognized color
        """
```

**Color validation**: after decoding, check for `blue|red|green` with a regex. If none found, raise `ValueError("SceneVLM output did not contain a recognized color: {raw_output!r}")`.

### 3.2 `src/policy.py` — `VLAPolicy`

```python
ROBOT_PORT  = "/dev/tty.usbmodem5B141136531"   # same as record.py
CAMERA_KEY  = "camera1"                          # matches async_inference.bash + finetuned model
FPS         = 30
HOME_JOINTS = None   # placeholder — set to actual top-view pose after hardware tuning

class VLAPolicy:
    def __init__(
        self,
        policy,           # lerobot PreTrainedPolicy
        preprocessor,
        postprocessor,
        vlm: SceneVLM,
        device,
    ): ...

    @classmethod
    def load(cls, checkpoint: str) -> "VLAPolicy":
        """
        Load SmolVLA from checkpoint path or HF repo_id.
        Also instantiates SceneVLM.
        Device order: MPS → CUDA → CPU.
        """

    def run(
        self,
        task: str | None = None,
        timeout: float = 18.0,
        move_to_home: bool = True,
    ) -> None:
        """
        Connect robot, optionally move to home, run inference loop, disconnect.
        If task is None, captures first frame and calls SceneVLM.query().
        Loop runs at FPS until timeout or KeyboardInterrupt.
        Robot always disconnected in finally block.
        """
```

### 3.3 `main.py` — updated CLI

```
python main.py --checkpoint <path>              # VLM determines task from scene
python main.py --checkpoint <path> --task "..."  # explicit task, skip VLM
python main.py --checkpoint <path> --no-home     # skip HOME_JOINTS movement
```

Updated `argparse`:
- `--task`: `required=False`, default `None`
- `--no-home`: `action="store_true"`
- `--checkpoint`: still required

---

## 4. Observation Format

The SmolVLA preprocessor expects observations keyed as `observation.images.{camera_key}`.

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `observation.images.camera1` | `torch.Tensor` float32 | `(C, H, W)` in `[0,1]` | converted from uint8 numpy |
| `observation.state` | `torch.Tensor` float32 | `(6,)` | joint positions from `robot.get_observation()` |
| `task` | `str` | — | grounded task string |

**Camera key note**: `record.py` uses `wrist_camera` but `async_inference.bash` and the finetuned model expect `camera1`. Verify that the training run used `rename_map: {wrist_camera: camera1}` or that the checkpoint was trained with `camera1` directly.

---

## 5. Action Loop Pseudocode

```python
def run(self, task, timeout, move_to_home):
    robot = SO101Follower(SO101FollowerConfig(port=ROBOT_PORT, cameras={...}))
    robot.connect()
    try:
        if move_to_home and HOME_JOINTS is not None:
            robot.move_to_joint_positions(HOME_JOINTS)

        # Stage 1: VLM grounding (once)
        if task is None:
            raw_obs = robot.capture_observation()
            frame = to_numpy_hwc(raw_obs[f"observation.images.{CAMERA_KEY}"])
            task = self.vlm.query(frame)
            print(f"[VLM] Generated task: {task!r}")

        print(f"[Policy] Running task: {task!r} for {timeout}s")
        deadline = time.monotonic() + timeout

        # Stage 2: SmolVLA action loop (30 fps)
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            obs = robot.capture_observation()
            obs_tensor = self._preprocess(obs, task)
            with torch.inference_mode():
                action = self.policy.select_action(obs_tensor)
            action = self._postprocess(action)
            robot.send_action(action)
            # pace to FPS
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, 1/FPS - elapsed))
    finally:
        robot.disconnect()
```

---

## 6. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| SceneVLM output has no color | Raise `ValueError` — abort before robot moves |
| `--checkpoint` path not found | `PreTrainedPolicy.from_pretrained` raises `OSError` — propagate |
| Robot not connected (port missing) | `SO101Follower.connect()` raises — propagate with original message |
| `KeyboardInterrupt` during loop | Caught by `finally` — robot disconnects cleanly |
| `HOME_JOINTS is None` and `move_to_home=True` | Log warning, skip movement, continue |
| Camera key mismatch in preprocessor | `KeyError` — propagate with message suggesting `rename_map` |

---

## 7. Test Plan

| Scenario | How to test |
|----------|-------------|
| `SceneVLM.query()` returns valid task | Unit test: mock model output, assert color extracted |
| `SceneVLM.query()` raises on no color | Unit test: mock model output without color, assert `ValueError` |
| `VLAPolicy.load()` with valid checkpoint | Integration: load actual checkpoint, check no exception |
| Explicit `--task` skips VLM | Manual: pass `--task`, confirm `SceneVLM.query` not called |
| Timeout terminates loop | Manual: set `--timeout 3`, confirm robot stops at ~3s |
| Clean disconnect on Ctrl-C | Manual: Ctrl-C during run, confirm `robot.is_connected` False |

---

## 8. Implementation Order

1. [ ] `src/vlm.py` — `SceneVLM` (no robot dependency, testable standalone)
2. [ ] `src/policy.py` — `VLAPolicy.load()` (model loading only, no robot)
3. [ ] `src/policy.py` — `VLAPolicy.run()` action loop (requires robot hardware)
4. [ ] `main.py` — add `--no-home`, make `--task` optional
5. [ ] Hardware test: explicit `--task` first, then VLM-guided mode

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-18 | Initial draft | junkyunglee |
