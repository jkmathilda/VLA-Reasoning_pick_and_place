#!/bin/bash

REPO_ID=ehalicki/eth-rl-so101-may19-policies
CHECKPOINT_STEP=025000
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
