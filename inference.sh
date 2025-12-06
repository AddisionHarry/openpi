#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"
CUDA_VISIBLE_DEVICES=2 TORCHINDUCTOR_COMPILE_THREADS=1 DEBUG_MODE=0 \
    uv run python3 ${SCRIPT_DIR}/scripts/inference/server.py \
    --model_path /root/openpi/checkpoints/pi05_zjhumanoid_industrial_sorting_jax/industrial_sorting_joint_space_right_arm_jax/9000 \
    --device cuda:0 --config_name pi05_zjhumanoid_industrial_sorting --port 10043
