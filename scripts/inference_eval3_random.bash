#!/bin/bash
# Eval 3 inference without any VLM.
#
# Training uses position-conditioned task strings ("Place the coke can on the
# <left|middle|right> image."). At eval time the TA's prompt references a
# celebrity, which we cannot resolve to a position without a VLM. To stay
# within the no-VLM constraint we simply pick one of the three positions
# uniformly at random and send that to the policy.
#
# Usage:
#   bash scripts/inference_eval3_random.bash
#   bash scripts/inference_eval3_random.bash --position left
#   POSITION=right bash scripts/inference_eval3_random.bash

set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/team11/.cache/huggingface/hub/models--ehalicki--eth-rl-so101-may19-policies/snapshots/3d7ea8252a3698b6642e3bc462f0f1242cdeeec1/checkpoints/025000/pretrained_model}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:50001}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

POSITION="${POSITION:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --position)
            POSITION="$2"
            shift 2
            ;;
        --position=*)
            POSITION="${1#*=}"
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
            echo "Usage: $0 [--position left|middle|right] [--checkpoint <path>]" >&2
            exit 1
            ;;
    esac
done

POSITIONS=(left middle right)
if [[ -z "$POSITION" ]]; then
    POSITION="${POSITIONS[$((RANDOM % 3))]}"
    echo ">>> No --position given; randomly chose: $POSITION"
else
    case "$POSITION" in
        left|middle|right) ;;
        *)
            echo "Error: --position must be left, middle, or right (got: $POSITION)" >&2
            exit 1
            ;;
    esac
fi

RESOLVED_TASK="Place the coke can on the ${POSITION} image."

echo ">>> Chosen position: $POSITION"
echo ">>> Resolved task:   $RESOLVED_TASK"
echo ">>> Checkpoint:      $CHECKPOINT_PATH"
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
