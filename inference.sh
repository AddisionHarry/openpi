#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"
CUDA_VISIBLE_DEVICES=2 TORCHINDUCTOR_COMPILE_THREADS=1 DEBUG_MODE=0 \
    uv run python3 ${SCRIPT_DIR}/scripts/inference/server.py \
    --model_path /root/openpi/checkpoints/pi05_zjhumanoid_grasp_can_relative_trajectory_chest_camera/relative_chest_camera/20000 \
    --device cuda:0 --config_name pi05_zjhumanoid_grasp_can --port 10043
    # --model_path ${SCRIPT_DIR}/checkpoints/pi05_zjhumanoid_grasp_can/my_experiment/25000 \
