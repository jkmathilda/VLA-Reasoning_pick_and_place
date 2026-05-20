#!/bin/bash
set -euo pipefail

RUN_NAME=smolvla_finetune_so101_may20_eval2_compositional

# Resume by fine-tuning from a previously trained policy checkpoint on the Hub
# instead of starting from lerobot/smolvla_base. Override on the command line:
#   RESUME_REPO_ID=... RESUME_STEP=... ./scripts/finetune_eval2_compositional.bash
RESUME_REPO_ID="${RESUME_REPO_ID:-ehalicki/eth-rl-so101-may19-policies}"
RESUME_STEP="${RESUME_STEP:-030000}"
RESUME_SUBPATH="checkpoints/$RESUME_STEP/pretrained_model"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=$PROJECT_ROOT/.venv

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! ldconfig -p 2>/dev/null | grep -q libavutil; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
    $SUDO apt-get update && $SUDO apt-get install -y ffmpeg
fi

if [ ! -d "$VENV" ]; then
    uv venv --python 3.12 $VENV
fi
source $VENV/bin/activate
uv pip install -r $PROJECT_ROOT/requirements.txt

echo "Downloading resume checkpoint $RESUME_REPO_ID @ step $RESUME_STEP ..."
SNAPSHOT_DIR=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$RESUME_REPO_ID', allow_patterns=['$RESUME_SUBPATH/*']))
")
RESUME_CHECKPOINT_PATH="$SNAPSHOT_DIR/$RESUME_SUBPATH"
echo "Resuming from: $RESUME_CHECKPOINT_PATH"

uv run --python 3.12 accelerate launch --mixed_precision=bf16 $PROJECT_ROOT/scripts/train_lerobot_eval2_compositional.py \
  --dataset.repo_id=ehalicki/eth-rl-so101-combined \
  --dataset.revision=main \
  --output_dir=$PROJECT_ROOT/outputs/$RUN_NAME \
  --job_name=$RUN_NAME \
  --policy.device=cuda \
  --policy.path="$RESUME_CHECKPOINT_PATH" \
  --rename_map='{"observation.images.wrist_camera": "observation.images.camera1"}' \
  --tolerance_s=0.04 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=5000 \
  --steps=100000 \
  --batch_size=32 \
  --policy.repo_id=ehalicki/eth-rl-so101-may20-eval2-compositional-policies

# To resume from a checkpoint, replace --policy.path with:
#   --policy.type=smolvla \
#   --config_path=$PROJECT_ROOT/outputs/$RUN_NAME/checkpoints/last/pretrained_model/train_config.json
#   --resume=true
