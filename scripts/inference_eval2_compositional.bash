#!/bin/bash
set -euo pipefail

# Update CHECKPOINT_PATH after training finishes / the HF cache resolves.
CHECKPOINT_PATH=/home/team11/.cache/huggingface/hub/models--ehalicki--eth-rl-so101-may20-eval2-compositional-policies/snapshots/main/checkpoints/last/pretrained_model
SERVER_ADDRESS=127.0.0.1:50001

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
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT_PATH="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 --task \"<task description>\" [--checkpoint <path>]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "Error: --task argument is required" >&2
    echo "Usage: $0 --task \"<task description>\" [--checkpoint <path>]" >&2
    echo "Examples:" >&2
    echo "  $0 --task \"Put the banana in the 2nd bowl from the left from the robot perspective\"" >&2
    echo "  $0 --task \"Put the banana in the bowl that is not green and not blue\"" >&2
    echo "  $0 --task \"Put the banana in the bowl on the right of the red bowl from the robot perspective\"" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> Task: $TASK"
echo ">>> Checkpoint: $CHECKPOINT_PATH"
echo ">>> Homing arm (no VLM) and starting control loop..."

SKIP_VLM=1 python "$PROJECT_ROOT/scripts/robot_client_with_vlm.py" \
    --server_address=$SERVER_ADDRESS \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=eth_rl_so101_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="$TASK" \
    --policy_type=smolvla \
    --pretrained_name_or_path=$CHECKPOINT_PATH \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.3 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
