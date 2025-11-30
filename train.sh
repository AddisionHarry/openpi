#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"

# CUDA_VISIBLE_DEVICES=0,1 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_industrial_sorting
# cp /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats.json \
#     /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/industrial_sorting/
# mv /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats.json \
#     /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats_jointswithwaist_backup_20251128.json


ITER=0

while true; do
    ITER=$((ITER + 1))

    printf "\n\n\n\n\n\n\n\n\n\n"
    date
    echo "========== LOOP $ITER START =========="
    echo ""

    CUDA_VISIBLE_DEVICES=0,1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DEBUG_MODE=0 \
    uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_zjhumanoid_industrial_sorting \
    --exp_name industrial_sorting_joint_space --resume

    echo ""
    date
    echo "========== LOOP $ITER DONE =========="
    echo ""
done

# CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=1 \
#     uv run python scripts/train_pytorch.py pi05_zjhumanoid_grasp_can_relative_trajectory --exp_name relative \
#     --resume

# 显存泄露
# ps -ef | grep python | grep openpi
# ps -ef | grep python | grep openpi | awk '{print $2}' | xargs -r kill -9
