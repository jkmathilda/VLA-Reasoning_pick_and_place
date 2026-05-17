#!/usr/bin/env bash
set -euo pipefail

ROBOT_PORT="/dev/tty.usbmodem5B141136531"
LEADER_PORT="/dev/tty.usbmodem5B141125311"

ARM="${1:-leader}"

case "$ARM" in
    leader)
        lerobot-calibrate \
            --teleop.type=so101_leader \
            --teleop.port="$LEADER_PORT" \
            --teleop.id=eth_rl_so101_leader
        ;;
    follower)
        lerobot-calibrate \
            --robot.type=so101_follower \
            --robot.port="$ROBOT_PORT" \
            --robot.id=eth_rl_so101_follower
        ;;
    *)
        echo "Usage: $0 [leader|follower]"
        exit 1
        ;;
esac
