#!/bin/bash
# Eval 1 / fixed-bowl-order inference.
#
# The physical bowl order is fixed as Blue-Red-Green from left to right, so
# color names map 1:1 to positions: blue=left, red=middle, green=right.
# This script just sed-substitutes the color word(s) in the task string with
# the matching position word and runs the same robot_client_with_vlm.py
# entrypoint as vlm_inference.bash — but with SKIP_VLM=1 so it homes the
# robot but skips the VLM load/query (no perception needed when the bowl
# order is known a priori).
#
# Usage:
#   bash scripts/color_inference.bash --task "Place the banana in the blue bowl"
#   → sends task "Place the banana in the left bowl" to the policy server.

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

# Case-insensitive substitution: blue→left, red→middle, green→right.
# Word boundaries (\b) avoid mangling things like "redo" or "bluebird".
RESOLVED_TASK=$(echo "$TASK" | sed -E '
    s/\b[Bb]lue\b/left/g
    s/\b[Rr]ed\b/middle/g
    s/\b[Gg]reen\b/right/g
')

echo ">>> Task spec:     $TASK"
echo ">>> Resolved task: $RESOLVED_TASK"
echo ">>> Launching robot client with SKIP_VLM=1 (homing only, no VLM)..."

SKIP_VLM=1 python "$PROJECT_ROOT/scripts/robot_client_with_vlm.py" \
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
