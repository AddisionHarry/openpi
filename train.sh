#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"

TASK="$1"
if [ -z "$TASK" ]; then
    echo "[Error]Must give task index."
    exit 1
fi

# CUDA_VISIBLE_DEVICES=0,1 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_industrial_sorting
# cp /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats.json \
#     /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/industrial_sorting/
# mv /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats.json \
#     /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251128/norm_stats_jointswithwaist_backup_20251204.json

# CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=0 \
#     uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#     scripts/train_pytorch.py pi05_zjhumanoid_industrial_sorting \
#     --exp_name industrial_sorting_joint_space
# CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=1 \
#     uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#     scripts/train_pytorch.py pi05_zjhumanoid_industrial_sorting \
#     --exp_name industrial_sorting_joint_space  --resume
# CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEBUG_MODE=0 \
#     uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#     scripts/train_pytorch.py pi05_zjhumanoid_industrial_sorting \
#     --exp_name industrial_sorting_joint_space_right_arm

if [ "$TASK" == "0" ]; then
    # CUDA_VISIBLE_DEVICES=0,1 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_industrial_sorting_jax
    CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_zjhumanoid_industrial_sorting_jax --exp-name=industrial_sorting_joint_space_right_arm_jax --overwrite

    # CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    #     pi05_zjhumanoid_industrial_sorting_jax --exp-name=industrial_sorting_joint_space_right_arm_jax --resume

elif [ "$TASK" == "1" ]; then
    # uv run python3 -c "import time; from tqdm import tqdm; total=2*3600; pbar=tqdm(total=total, desc='Remaining Sleep Time', unit='seconds', ncols=100); [time.sleep(1) or pbar.update(1) for _ in range(total)]; pbar.close();"

    CUDA_VISIBLE_DEVICES=1 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_industrial_sorting_tcp_raw_jax && \
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
        pi05_zjhumanoid_industrial_sorting_tcp_raw_jax --exp-name=industrial_sorting_ee_pose_raw_right_arm_jax --overwrite

    # CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    #     pi05_zjhumanoid_industrial_sorting_tcp_raw_jax --exp-name=industrial_sorting_ee_pose_raw_right_arm_jax --resume

elif [ "$TASK" == "2" ]; then
    # uv run python3 -c "import time; from tqdm import tqdm; total=14*3600; pbar=tqdm(total=total, desc='Remaining Sleep Time', unit='seconds', ncols=100); [time.sleep(1) or pbar.update(1) for _ in range(total)]; pbar.close();"

    # CUDA_VISIBLE_DEVICES=2 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_industrial_sorting_tcp_relative_chest_jax && \
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
        pi05_zjhumanoid_industrial_sorting_tcp_relative_chest_jax --exp-name=industrial_sorting_ee_pose_chest_right_arm_jax --overwrite

    # CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    #     pi05_zjhumanoid_industrial_sorting_tcp_relative_chest_jax --exp-name=industrial_sorting_ee_pose_chest_right_arm_jax --resume

elif [ "$TASK" == "3" ]; then
    # uv run python3 -c "import time; from tqdm import tqdm; total=16*3600; pbar=tqdm(total=total, desc='Remaining Sleep Time', unit='seconds', ncols=100); [time.sleep(1) or pbar.update(1) for _ in range(total)]; pbar.close();"

    CUDA_VISIBLE_DEVICES=0 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_industrial_sorting_tcp_relative_wrist_jax && \
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
        pi05_zjhumanoid_industrial_sorting_tcp_relative_wrist_jax --exp-name=industrial_sorting_ee_pose_wrist_right_arm_jax --overwrite

    # CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    #     pi05_zjhumanoid_industrial_sorting_tcp_relative_wrist_jax --exp-name=industrial_sorting_ee_pose_chest_right_arm_jax --resume

elif [ "$TASK" == "4" ]; then
    # uv run python3 -c "import time; from tqdm import tqdm; total=4*3600; pbar=tqdm(total=total, desc='Remaining Sleep Time', unit='seconds', ncols=100); [time.sleep(1) or pbar.update(1) for _ in range(total)]; pbar.close();"

    # CUDA_VISIBLE_DEVICES=1 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_cloth_joint_space && \
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
        pi05_zjhumanoid_cloth_joint_space --exp-name=cloth_joint --overwrite

    # CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    #     pi05_zjhumanoid_cloth_joint_space --exp-name=cloth_joint --resume

elif [ "$TASK" == "5" ]; then
    # CUDA_VISIBLE_DEVICES=2 uv run python3 scripts/compute_norm_stats.py --config-name pi05_zjhumanoid_cloth_absolute_tcp_pose && \
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_zjhumanoid_cloth_absolute_tcp_pose --exp-name=cloth_absolute_tcp_pose --overwrite

    # CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    #     pi05_zjhumanoid_cloth_absolute_tcp_pose --exp-name=cloth_absolute_tcp_pose --resume

elif [ "$TASK" == "6" ]; then
    # CUDA_VISIBLE_DEVICES=1 uv run python3 scripts/compute_norm_stats.py --config-name pi0_zjhumanoid_industrial_sorting_jax && \
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=1 uv run scripts/train.py \
        pi0_zjhumanoid_industrial_sorting_jax --exp-name=pi0_industrial_sorting_cleaned_waist_action --overwrite

    # CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_zjhumanoid_industrial_sorting_jax --exp-name=pi0_industrial_sorting_cleaned_waist_action --resume

    # bash /root/openpi/evaluate.sh \
    #     --cuda-visible-devices 1 \
    #     --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_manually_cleaned_20251210/pi0_zjhumanoid_industrial_sorting_jax \
    #     --model-root /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/pi0_industrial_sorting_cleaned_waist_action \
    #     --config-name pi0_zjhumanoid_industrial_sorting_jax \
    #     --use-arms "[False, True]" \
    #     --use-waist-angles True \
    #     --use-tcp-pose False \
    #     --total-steps 20000

elif [ "$TASK" == "7" ]; then
    CUDA_VISIBLE_DEVICES=0
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_waist && \
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist --exp-name=pi0_industrial_sorting_waist_action_1214data --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist --exp-name=pi0_industrial_sorting_waist_action_1214data --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214/pi0_industrial_sorting_joint_waist \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_waist/pi0_industrial_sorting_waist_action_1214data \
        --config-name pi0_industrial_sorting_joint_waist \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 20000

elif [ "$TASK" == "8" ]; then
    CUDA_VISIBLE_DEVICES=1
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi05_industrial_sorting_joint_waist && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_industrial_sorting_joint_waist --exp-name=pi05_industrial_sorting_waist_action_1214data_1 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_waist --exp-name=pi05_industrial_sorting_waist_action_1214data_1 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214/pi05_industrial_sorting_joint_waist \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_waist/pi05_industrial_sorting_waist_action_1214data_1 \
        --config-name pi05_industrial_sorting_joint_waist \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 30000

elif [ "$TASK" == "9" ]; then
    CUDA_VISIBLE_DEVICES=2
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_waist_change_prompt && \
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist_change_prompt --exp-name=pi0_industrial_sorting_waist_action_1214data_update_prompt --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist_change_prompt --exp-name=pi0_industrial_sorting_waist_action_1214data_update_prompt --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214/pi0_industrial_sorting_joint_waist_change_prompt \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_waist_change_prompt/pi0_industrial_sorting_waist_action_1214data_update_prompt \
        --config-name pi0_industrial_sorting_joint_waist_change_prompt \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 20000

elif [ "$TASK" == "10" ]; then
    CUDA_VISIBLE_DEVICES=1,2
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_waist && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi0_industrial_sorting_joint_waist --exp-name=pi0_industrial_sorting_waist_action_1214data_20251220 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist --exp-name=pi0_industrial_sorting_waist_action_1214data_20251220 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214/pi0_industrial_sorting_joint_waist \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_waist/pi0_industrial_sorting_waist_action_1214data_20251220 \
        --config-name pi0_industrial_sorting_joint_waist \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 20000

fi

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
#         --exp_name industrial_sorting_joint_space_right_arm --resume
#     # ps -ef | grep python | grep openpi | awk '{print $2}' | xargs -r kill -9

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
