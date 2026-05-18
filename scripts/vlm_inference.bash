#!/bin/bash
set -euo pipefail

CHECKPOINT_PATH=/home/team11/.cache/huggingface/hub/models--ehalicki--eth-rl-so101-may17-policies/snapshots/80dd4a5d09322b58a891c6221aebb270a6200fc4/checkpoints/025000/pretrained_model
SERVER_ADDRESS=127.0.0.1:50001

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TASK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 --task \"<task description>\"" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "Error: --task argument is required" >&2
    echo "Usage: $0 --task \"<task description>\"" >&2
    exit 1
fi

echo ">>> Task spec: $TASK"
echo ">>> Querying VLM to resolve bowl position..."
RESOLVED_TASK=$(python "$PROJECT_ROOT/scripts/query_vlm.py" --task "$TASK")
echo ">>> Resolved task: $RESOLVED_TASK"

# run the robot client with the VLM-generated task.
python -m lerobot.async_inference.robot_client \
    --server_address=$SERVER_ADDRESS \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=eth_rl_so101_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="$RESOLVED_TASK" \
    --policy_type=smolvla \
    --pretrained_name_or_path=$CHECKPOINT_PATH \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.3 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
