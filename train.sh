#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=0 \
    uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_zjhumanoid_grasp_can_relative_trajectory_chest_camera --exp_name relative_chest_camera \
    --resume
# CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=1 \
#     uv run python scripts/train_pytorch.py pi05_zjhumanoid_grasp_can_relative_trajectory --exp_name relative \
#     --resume

# 显存泄露
# ps -ef | grep python | grep openpi
# ps -ef | grep python | grep openpi | awk '{print $2}' | xargs -r kill -9
