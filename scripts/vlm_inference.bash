#!/bin/bash
set -euo pipefail

CHECKPOINT_PATH=/home/team11/.cache/huggingface/hub/models--ehalicki--eth-rl-so101-may19-policies/snapshots/3d7ea8252a3698b6642e3bc462f0f1242cdeeec1/checkpoints/025000/pretrained_model
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
echo ">>> Launching robot client (VLM resolves the task after handshake)..."

# The merged client connects to the server, moves the robot to the overview
# pose, queries SmolVLM with --task as the high-level spec, and then runs
# the normal async control loop with the resolved task.
python "$PROJECT_ROOT/scripts/robot_client_with_vlm.py" \
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
