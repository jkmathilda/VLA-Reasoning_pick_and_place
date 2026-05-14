#!/bin/bash

RUN_NAME=smolvla_finetune_so101_may4

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=$PROJECT_ROOT/.venv

if [ ! -d "$VENV" ]; then
    uv venv --python 3.12 $VENV
fi
source $VENV/bin/activate
uv pip install -r $PROJECT_ROOT/requirements.txt

uv run --python 3.12 accelerate launch --mixed_precision=bf16 $PROJECT_ROOT/scripts/train_lerobot.py \
  --dataset.repo_id=ehalicki/eth-rl-so101-may4 \
  --output_dir=$PROJECT_ROOT/outputs/$RUN_NAME \
  --job_name=$RUN_NAME \
  --policy.device=cuda \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.wrist_camera": "observation.images.camera1"}' \
  --tolerance_s=0.04 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=5000 \
  --steps=50000 \
  --batch_size=32 
  --policy.repo_id=ehalicki/eth-rl-so101-may4-policy
# To resume from a checkpoint, replace --policy.path with:
#   --policy.type=smolvla \
#   --config_path=$PROJECT_ROOT/outputs/$RUN_NAME/checkpoints/last/pretrained_model/train_config.json
#   --resume=true

