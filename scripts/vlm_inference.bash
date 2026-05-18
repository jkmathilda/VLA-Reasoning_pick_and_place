#!/bin/bash
set -euo pipefail

CHECKPOINT_PATH=/home/eryk/.cache/huggingface/hub/models--ehalicki--eth-rl-so101-may4-policies/snapshots/8d0170fa7d9538901b648a6b8e2a0c5ff83f31f8/smolvla_finetune_so101_may4/checkpoints/030000/pretrained_model
SERVER_ADDRESS=eryk-pc:8080

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# move robot to home, capture overview image, get task from VLM.
echo ">>> Querying VLM for task..."
TASK=$(python "$PROJECT_ROOT/scripts/query_vlm.py")
echo ">>> Task: $TASK"

# run the robot client with the VLM-generated task.
python -m lerobot.async_inference.robot_client \
    --server_address=$SERVER_ADDRESS \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B141136531 \
    --robot.id=eth_rl_so101_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="$TASK" \
    --policy_type=smolvla \
    --pretrained_name_or_path=$CHECKPOINT_PATH \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
