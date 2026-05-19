#!/bin/bash
# Eval 1 / fixed-bowl-order inference.
#
# The physical bowl order is fixed as Blue-Red-Green from left to right, so
# color names map 1:1 to positions: blue=left, red=middle, green=right.
# This script just sed-substitutes the color word(s) in the task string with
# the matching position word and sends that directly to the stock lerobot
# robot_client — no VLM step, no homing, no overview snapshot.
#
# Usage:
#   bash scripts/color_inference.bash --task "Place the banana in the blue bowl"
#   → sends task "Place the banana in the left bowl" to the policy server.

set -euo pipefail

CHECKPOINT_PATH=/home/team11/.cache/huggingface/hub/models--ehalicki--eth-rl-so101-may17-policies/snapshots/80dd4a5d09322b58a891c6221aebb270a6200fc4/checkpoints/025000/pretrained_model
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

echo ">>> Task spec:    $TASK"
echo ">>> Resolved task: $RESOLVED_TASK"

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
