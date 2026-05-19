#!/bin/bash

# Set these to the values printed by async_server.bash on the server machine.
CHECKPOINT_PATH=/home/team11/.cache/huggingface/hub/models--ehalicki--eth-rl-so101-may17-policies/snapshots/80dd4a5d09322b58a891c6221aebb270a6200fc4/checkpoints/025000/pretrained_model

SERVER_ADDRESS=127.0.0.1:50001

python -m lerobot.async_inference.robot_client \
    --server_address=$SERVER_ADDRESS \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=eth_rl_so101_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="Place the banana in the green bowl" \
    --policy_type=smolvla \
    --pretrained_name_or_path=$CHECKPOINT_PATH \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.7 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
