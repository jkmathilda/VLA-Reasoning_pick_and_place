#!/bin/bash

RUN_NAME=smolvla_finetune_so101_may20_eval3

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

uv run --python 3.12 accelerate launch --mixed_precision=bf16 $PROJECT_ROOT/scripts/train_lerobot_eval3.py \
  --dataset.repo_id=ehalicki/eth-rl-so101-celebrities \
  --dataset.revision=main \
  --output_dir=$PROJECT_ROOT/outputs/$RUN_NAME \
  --job_name=$RUN_NAME \
  --policy.device=cuda \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.wrist_camera": "observation.images.camera1"}' \
  --tolerance_s=0.04 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=500 \
  --steps=100000 \
  --batch_size=32 \
  --policy.repo_id=ehalicki/eth-rl-so101-celebrities-policies

# To resume from a checkpoint, replace --policy.path with:
#   --policy.type=smolvla \
#   --config_path=$PROJECT_ROOT/outputs/$RUN_NAME/checkpoints/last/pretrained_model/train_config.json
#   --resume=true
