# VLA-Reasoning Pick and Place

Teleoperation and dataset recording scripts for the SO101 robot arm using [LeRobot](https://github.com/huggingface/lerobot).

## Hardware Requirements

- SO101 Follower robot arm (connected via USB)
- SO101 Leader teleoperator (connected via USB)
- USB camera (e.g. USB2.0_CAM — identified automatically by default resolution)

## Installation

**1. Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

## Configuration

Before running, update the USB port paths at the top of each script to match your system.

**Find your port names:**
```bash
# macOS
ls /dev/tty.usbmodem*

# Linux
ls /dev/ttyUSB* /dev/ttyACM*
```

**Edit `src/teleop.py` and `src/record.py`:**
```python
ROBOT_PORT  = "/dev/tty.usbmodem<follower-id>"
LEADER_PORT = "/dev/tty.usbmodem<leader-id>"
```

The camera index is resolved automatically at runtime by matching the camera's default resolution (`1920x1080`). No manual index configuration is needed.

## Usage

### Teleoperation

Stream live data from the robot and visualize it with Rerun:

```bash
python src/teleop.py
```

### Record Episodes

Record demonstration episodes and save them as a LeRobot dataset:

```bash
python src/record.py
```

After each episode you will be prompted to enter a task description. Use `--repeat-task` to reuse the same task for all episodes:

```bash
python src/record.py --repeat-task
```

**Keyboard controls during recording:**

| Key | Action |
|-----|--------|
| Right arrow | End episode early |
| Left arrow | Discard and re-record episode |
| Escape | Stop recording entirely |

### Dataset

Recorded episodes are cached locally at `~/.cache/huggingface/lerobot/<repo-id>`. Configure `REPO_ID` in `src/record.py` to set the Hugging Face dataset repository target.