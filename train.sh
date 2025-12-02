#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"

# CUDA_VISIBLE_DEVICES=0,1 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_industrial_sorting
# cp /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats.json \
#     /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/industrial_sorting/
# mv /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats.json \
#     /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats_jointswithwaist_backup_20251128.json

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=1        # 禁用 InfiniBand，优先 NVLink / socket
# export NCCL_P2P_DISABLE=0      # 尝试启用 P2P（NVLink）

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo   # 使用回环接口，单机多卡安全但速度可能慢
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1
# # 禁用 IB（否则会卡死很久）
# export NCCL_IB_DISABLE=1
# # 强制使用 NVLink（你的 GPU0/1/2 之间是 NV8）
# export NCCL_P2P_DISABLE=0
# # 增强 NCCL 超时，避免误杀
# export NCCL_ASYNC_ERROR_HANDLING=1
# export PYTHONFAULTHANDLER=1
# export NCCL_DEBUG_FILE=/tmp/nccl_rank_%r.log

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=0 \
    uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_zjhumanoid_industrial_sorting \
    --exp_name industrial_sorting_joint_space
# CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=1 \
#     uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#     scripts/train_pytorch.py pi05_zjhumanoid_industrial_sorting \
#     --exp_name industrial_sorting_joint_space  --resume

# ITER=0
# while true; do
#     ITER=$((ITER + 1))

#     printf "\n\n\n\n\n\n\n\n\n\n"
#     date
#     echo "========== LOOP $ITER START =========="
#     echo ""

#     CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=0 \
#         uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#         scripts/train_pytorch.py pi05_zjhumanoid_industrial_sorting \
#         --exp_name industrial_sorting_joint_space --resume

#     echo ""
#     date
#     echo "========== LOOP $ITER DONE =========="
#     echo ""
# done

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=1 \
#     uv run python scripts/train_pytorch.py pi05_zjhumanoid_grasp_can_relative_trajectory --exp_name industrial_sorting_joint_space --resume
# CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=0 \
#     uv run python scripts/train_pytorch.py pi05_zjhumanoid_grasp_can_relative_trajectory --exp_name industrial_sorting_joint_space_1

# ITER=0
# while true; do
#     ITER=$((ITER + 1))

#     printf "\n\n\n\n\n\n\n\n\n\n"
#     date
#     echo "========== LOOP $ITER START =========="
#     echo ""

#     CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=0 \
#         uv run python scripts/train_pytorch.py pi05_zjhumanoid_grasp_can_relative_trajectory \
#         --exp_name industrial_sorting_joint_space_1 --resume

#     echo ""
#     date
#     echo "========== LOOP $ITER DONE =========="
#     echo ""
# done

# 显存泄露
# ps -ef | grep python | grep openpi
# ps -ef | grep python | grep openpi | awk '{print $2}' | xargs -r kill -9
