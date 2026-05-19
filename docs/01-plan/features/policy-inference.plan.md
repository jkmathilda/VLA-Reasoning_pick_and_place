# policy-inference Planning Document

> **Summary**: Implement `VLAPolicy` — a two-stage inference pipeline combining SmolVLM (visual reasoning) and SmolVLA (motor control) for pick-and-place tasks.
>
> **Project**: VLA-Reasoning Pick and Place
> **Author**: junkyunglee
> **Date**: 2026-05-18
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

`main.py` accepts a natural language task (e.g. `"Put the banana into the 2nd bowl from the left."`) but `src/policy.py` does not exist yet. This feature implements the `VLAPolicy` class that bridges the user's instruction to physical robot execution via a two-stage ML pipeline.

### 1.2 Background

The project has two trained components:

- **Task 1 — SmolVLA** (motor policy): fine-tuned via `scripts/train_lerobot_eval1.py` on demonstration episodes. Understands color-based instructions like `"Put the banana in the blue bowl."` and outputs 6-DOF joint actions.
- **Task 2 — SmolVLM** (`HuggingFaceTB/SmolVLM-256M-Instruct`): vision-language model capable of spatial and semantic reasoning. Needed for instructions that SmolVLA was not trained on, such as positional references (`"2nd bowl from the left"`) or abstract descriptions.

SmolVLM grounds the raw instruction + camera image into a normalized, color-based instruction that SmolVLA can execute. When the instruction is already color-based, SmolVLM can be bypassed.

### 1.3 Related Files

- Entry point: `main.py`
- Target file: `src/policy.py`
- VLM scratch: `scripts/vlm.py`
- Training: `scripts/train_lerobot_eval1.py`
- Recording: `src/record.py`

---

## 2. Scope

### 2.1 In Scope

- [ ] `VLAPolicy` class with `load(checkpoint)` and `run(task, timeout)` interface
- [ ] SmolVLM model loading and scene grounding (Task 2)
- [ ] SmolVLA checkpoint loading and action inference (Task 1)
- [ ] Robot connection + action execution loop at 30 fps
- [ ] Timeout-based episode termination
- [ ] Grounding bypass when task is already color-based

### 2.2 Out of Scope

- Training or fine-tuning either model
- Recording new episodes
- Evaluation harness / success detection
- Multi-camera support beyond the single wrist camera

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `VLAPolicy.load(checkpoint)` loads the SmolVLA checkpoint and initializes SmolVLM | High | Pending |
| FR-02 | `VLAPolicy.run(task, timeout)` connects the robot, runs the inference loop, and disconnects on exit | High | Pending |
| FR-03 | SmolVLM receives the raw task string + current camera frame and returns a grounded, color-based instruction | High | Pending |
| FR-04 | If the task already contains a color keyword (`blue`, `red`, `green`), skip SmolVLM grounding | Medium | Pending |
| FR-05 | SmolVLA takes the grounded instruction + camera frame sequence and returns a joint action vector | High | Pending |
| FR-06 | Actions are sent to the SO101 Follower at 30 fps | High | Pending |
| FR-07 | The loop terminates when `timeout` seconds have elapsed or `KeyboardInterrupt` is raised | High | Pending |
| FR-08 | The robot disconnects cleanly in a `finally` block regardless of how the loop exits | High | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Latency | Full inference cycle (VLM + VLA) ≤ 33 ms per step (30 fps) | Profiling |
| Reliability | No uncaught exceptions during clean task execution | Manual test |
| Portability | Runs on Apple Silicon (MPS) and CUDA without code changes | Test on both devices |

---

## 4. Two-Stage Pipeline Design

```
User task string
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 1: SmolVLM Grounding          │
│                                      │
│  Input:  task + wrist_camera frame   │
│  Output: grounded_task (color-based) │
│                                      │
│  Skip if task already has a color.   │
└──────────────────────────────────────┘
    │  grounded_task
    ▼
┌──────────────────────────────────────┐
│  Stage 2: SmolVLA Action Inference   │
│                                      │
│  Input:  grounded_task + obs frames  │
│  Output: 6-DOF joint action          │
└──────────────────────────────────────┘
    │  action
    ▼
SO101 Follower (30 fps loop)
```

### Grounding Prompt Template (Stage 1)

```
You are controlling a robot arm. Look at the image.
Task: "{raw_task}"
The available bowls are colored blue, red, or green.
Rewrite the task using only a color to identify the bowl.
Reply with exactly one sentence in the form: "Put the banana in the {color} bowl."
```

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| SmolVLM inference > 33 ms breaks 30 fps | High | High | Run VLM only on first frame (or every N frames); cache grounded task |
| SmolVLM outputs unexpected text format | Medium | Medium | Parse with regex; fall back to raw task if no color found |
| SmolVLA checkpoint API differs from lerobot public API | High | Low | Check `PreTrainedPolicy.from_pretrained` signature against training script |
| MPS/CUDA device detection | Low | Low | Use `torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")` |
| Robot port hardcoded in `record.py` | Medium | Low | Read port from env var or CLI arg; default to the constants in `record.py` |

---

## 6. Architecture Decisions

| Decision | Selected | Rationale |
|----------|----------|-----------|
| VLM inference frequency | Once per episode (first frame), cached | VLM is slow; grounded task is stable across the episode |
| VLM model | `HuggingFaceTB/SmolVLM-256M-Instruct` | Already prototyped in `scripts/vlm.py` |
| VLA loading | `lerobot` `PreTrainedPolicy.from_pretrained` | Consistent with training pipeline |
| Robot port config | Constants from `src/record.py` (reused) | Single source of truth; already validated |
| Action execution | `robot.send_action(action)` at 30 fps | Matches recording fps |

---

## 7. Success Criteria

### 7.1 Definition of Done

- [ ] `python main.py --task "Put the banana in the blue bowl."` runs without error
- [ ] `python main.py --task "Put the banana into the 2nd bowl from the left."` grounds to a color and executes
- [ ] Robot disconnects cleanly after timeout or Ctrl-C
- [ ] SmolVLM grounding is skipped when a color keyword is present in the task

### 7.2 Quality Criteria

- [ ] No bare `except` clauses
- [ ] Device (MPS/CUDA/CPU) auto-detected
- [ ] No hardcoded paths except robot ports (already precedented in `record.py`)

---

## 8. Next Steps

1. [ ] Write design document (`policy-inference.design.md`) — class interface, grounding prompt, action loop pseudocode
2. [ ] Implement `src/policy.py`
3. [ ] Manual test on hardware

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-18 | Initial draft | junkyunglee |
