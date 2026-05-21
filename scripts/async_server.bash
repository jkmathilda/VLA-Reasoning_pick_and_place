#!/bin/bash
set -euo pipefail

REPO_ID="${REPO_ID:-ehalicki/eth-rl-so101-may19-policies}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-025000}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO_ID="$2"
            shift 2
            ;;
        --repo=*)
            REPO_ID="${1#*=}"
            shift
            ;;
        --checkpoint)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT_STEP="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--repo <hf_repo_id>] [--checkpoint <step>]" >&2
            exit 1
            ;;
    esac
done

CHECKPOINT_SUBPATH=checkpoints/$CHECKPOINT_STEP/pretrained_model

echo "Ensuring $REPO_ID checkpoint $CHECKPOINT_STEP is in the HF cache..."
SNAPSHOT_DIR=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$REPO_ID', allow_patterns=['$CHECKPOINT_SUBPATH/*']))
")
CHECKPOINT_PATH=$SNAPSHOT_DIR/$CHECKPOINT_SUBPATH
echo "Checkpoint available at:"
echo "  $CHECKPOINT_PATH"
echo "Pass this path to the client as --pretrained_name_or_path."

python -m lerobot.async_inference.policy_server \
     --host=0.0.0.0 \
     --port=50001
