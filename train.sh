#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"

TASK="$1"
if [ -z "$TASK" ]; then
    echo "[Error]Must give task index."
    exit 1
fi

if [ "$TASK" == "0" ]; then
    CUDA_VISIBLE_DEVICES=0
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260126 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi0_industrial_sorting_joint_20260126 --exp-name=pi0_industrial_sorting_20260128 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_20260126 --exp-name=pi0_industrial_sorting_20260128 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260125/pi0_joint_reshape_videos \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_20260126/pi0_industrial_sorting_20260128 \
        --config-name pi0_industrial_sorting_joint_20260126 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 35000 \
        --inference-res-freq 30

elif [ "$TASK" == "1" ]; then
    CUDA_VISIBLE_DEVICES=1
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi05_industrial_sorting_joint_20260126 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_industrial_sorting_joint_20260126 --exp-name=pi05_industrial_sorting_20260128 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_20260126 --exp-name=pi05_industrial_sorting_20260128 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260125/pi0_joint_reshape_videos \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260126/pi05_industrial_sorting_20260128 \
        --config-name pi05_industrial_sorting_joint_20260126 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 40000 \
        --inference-res-freq 30

elif [ "$TASK" == "2" ]; then
    uv run python3 -c "from tqdm import tqdm; import time; [time.sleep(1) for _ in tqdm(range(3600 * 25), desc='Sleeping 25 hours', unit='s')]"

    CUDA_VISIBLE_DEVICES=2
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260130_last_frames_still && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=1 uv run scripts/train.py \
        pi0_industrial_sorting_joint_20260130_last_frames_still --exp-name=pi0_industrial_sorting_last_frames_still_20260131 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_20260130_last_frames_still --exp-name=pi0_industrial_sorting_last_frames_still_20260131 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260125/pi0_joint_reshape_videos \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_20260130_last_frames_still/pi0_industrial_sorting_last_frames_still_20260131 \
        --config-name pi0_industrial_sorting_joint_20260130_last_frames_still \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 25000 \
        --inference-res-freq 30 \
        --evaluate-interval 1000

elif [ "$TASK" == "3" ]; then
    uv run python3 -c "from tqdm import tqdm; import time; [time.sleep(1) for _ in tqdm(range(3600 * 20), desc='Sleeping 20 hours', unit='s')]"

    CUDA_VISIBLE_DEVICES=0
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run scripts/compute_norm_stats.py --config-name pi05_industrial_sorting_joint_20260130_last_frames_still && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_industrial_sorting_joint_20260130_last_frames_still --exp-name=pi05_industrial_sorting_last_frames_still_20260131 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_20260130_last_frames_still --exp-name=pi05_industrial_sorting_last_frames_still_20260131 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260125/pi0_joint_reshape_videos \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260130_last_frames_still/pi05_industrial_sorting_last_frames_still_20260131 \
        --config-name pi05_industrial_sorting_joint_20260130_last_frames_still \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 25000 \
        --inference-res-freq 30 \
        --evaluate-interval 1000

elif [ "$TASK" == "4" ]; then
    uv run python3 -c "from tqdm import tqdm; import time; [time.sleep(1) for _ in tqdm(range(3600 * 10), desc='Sleeping 10 hours', unit='s')]"

    CUDA_VISIBLE_DEVICES=1
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=1 uv run scripts/train.py \
        pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images --exp-name=pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images --exp-name=pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260125/pi0_joint_reshape_videos \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images/pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images \
        --config-name pi0_industrial_sorting_joint_20260130_last_frames_still_grasp_noise_chest_images \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 25000 \
        --inference-res-freq 30 \
        --evaluate-interval 1000

elif [ "$TASK" == "5" ]; then
    CUDA_VISIBLE_DEVICES=2
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run scripts/compute_norm_stats.py --config-name pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images --exp-name=pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images --exp-name=pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260125/pi0_joint_reshape_videos \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images/pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images \
        --config-name pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 25000 \
        --inference-res-freq 30 \
        --evaluate-interval 1000

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
        --total-steps 30000 \
        --inference-res-freq 30

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
        --total-steps 20000 \
        --inference-res-freq 30

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
        --total-steps 20000 \
        --inference-res-freq 30

elif [ "$TASK" == "11" ]; then
    CUDA_VISIBLE_DEVICES=0
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_waist_manually_cleaned20251229 && \
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist_manually_cleaned20251229 --exp-name=pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist_manually_cleaned20251229 --exp-name=pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20251214/pi0_joint_clean_joint_jit \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_waist_manually_cleaned20251229/pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 \
        --config-name pi0_industrial_sorting_joint_waist_manually_cleaned20251229 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 30000 \
        --inference-res-freq 10

elif [ "$TASK" == "12" ]; then
    CUDA_VISIBLE_DEVICES=0
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi05_industrial_sorting_joint_waist_manually_cleaned20251229 && \
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_waist_manually_cleaned20251229 --exp-name=pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_waist_manually_cleaned20251229 --exp-name=pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20251214/pi0_joint_clean_joint_jit \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_waist_manually_cleaned20251229/pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 \
        --config-name pi05_industrial_sorting_joint_waist_manually_cleaned20251229 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 30000 \
        --inference-res-freq 10

elif [ "$TASK" == "13" ]; then
    CUDA_VISIBLE_DEVICES=1
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi05_industrial_sorting_joint_waist_manually_cleaned20251230 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_industrial_sorting_joint_waist_manually_cleaned20251230 --exp-name=pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251230_31rerun_lowerlr --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_waist_manually_cleaned20251230 --exp-name=pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251230_31rerun_lowerlr --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20251214/pi0_joint_clean_joint_jit \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_waist_manually_cleaned20251230/pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251230_31rerun_lowerlr \
        --config-name pi05_industrial_sorting_joint_waist_manually_cleaned20251230 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 30000 \
        --inference-res-freq 10

elif [ "$TASK" == "14" ]; then
    CUDA_VISIBLE_DEVICES=0
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_waist_manually_cleaned20251230 && \
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist_manually_cleaned20251230 --exp-name=pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20260108_30fps --overwrite

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi0_industrial_sorting_joint_waist_manually_cleaned20251230 --exp-name=pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20260105_30fps --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20251214/pi0_joint_clean_joint_jit \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_waist_manually_cleaned20251230/pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20260105_30fps \
        --config-name pi0_industrial_sorting_joint_waist_manually_cleaned20251230 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 30000 \
        --inference-res-freq 30

elif [ "$TASK" == "15" ]; then
    CUDA_VISIBLE_DEVICES=0
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi05_industrial_sorting_joint_waist_manually_cleaned20251229 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_industrial_sorting_joint_waist_manually_cleaned20251229 --exp-name=pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_waist_manually_cleaned20251229 --exp-name=pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20251214/pi0_joint_clean_joint_jit \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_waist_manually_cleaned20251229/pi05_industrial_sorting_waist_action_1214data_manually_cleaned_20251229 \
        --config-name pi05_industrial_sorting_joint_waist_manually_cleaned20251229 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 30000 \
        --inference-res-freq 30

elif [ "$TASK" == "16" ]; then
    CUDA_VISIBLE_DEVICES=1
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_waist_manually_cleaned20251230 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=2 uv run scripts/train.py \
        pi0_industrial_sorting_joint_waist_manually_cleaned20251230 --exp-name=pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20260105_30fps --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_waist_manually_cleaned20251230 --exp-name=pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20260105_30fps --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20251214/pi0_joint_clean_joint_jit \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_waist_manually_cleaned20251230/pi0_industrial_sorting_waist_action_1214data_manually_cleaned_20260105_30fps \
        --config-name pi0_industrial_sorting_joint_waist_manually_cleaned20251230 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 30000 \
        --inference-res-freq 30

elif [ "$TASK" == "17" ]; then
    CUDA_VISIBLE_DEVICES=1
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=1 uv run python3 scripts/compute_norm_stats.py --config-name pi0_breaker_placement_joint_20260108 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi0_breaker_placement_joint_20260108 --exp-name=pi0_breaker_placement_20260109 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_breaker_placement_joint_20260108 --exp-name=pi0_breaker_placement_20260109 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20251214/pi0_joint_clean_joint_jit \
        --model-root /root/openpi/checkpoints/pi0_breaker_placement_joint_20260108/pi0_breaker_placement_20260109 \
        --config-name pi0_breaker_placement_joint_20260108 \
        --use-arms "[False, True]" \
        --use-waist-angles False \
        --use-tcp-pose False \
        --total-steps 30000 \
        --inference-res-freq 30

elif [ "$TASK" == "18" ]; then
    CUDA_VISIBLE_DEVICES=0
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260112 && \
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_20260112 --exp-name=pi0_industrrial_sorting_20260114 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_20260112 --exp-name=pi0_industrrial_sorting_20260114 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260112/pi0_joint \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_20260112/pi0_industrrial_sorting_20260114 \
        --config-name pi0_industrial_sorting_joint_20260112 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 20000 \
        --inference-res-freq 30

elif [ "$TASK" == "19" ]; then
    CUDA_VISIBLE_DEVICES=0
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260112 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi0_industrial_sorting_joint_20260112 --exp-name=pi0_industrrial_sorting_20260120 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_20260112 --exp-name=pi0_industrrial_sorting_20260120 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260112/pi0_joint \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_20260112/pi0_industrrial_sorting_20260120 \
        --config-name pi0_industrial_sorting_joint_20260112 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 40000 \
        --inference-res-freq 30

elif [ "$TASK" == "20" ]; then
    CUDA_VISIBLE_DEVICES=2
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260112 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi0_industrial_sorting_joint_20260112 --exp-name=pi0_industrrial_sorting_20260118_2 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi0_industrial_sorting_joint_20260112 --exp-name=pi0_industrrial_sorting_20260118_2 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260112/pi0_joint \
        --model-root /root/openpi/checkpoints/pi0_industrial_sorting_joint_20260112/pi0_industrrial_sorting_20260118_2 \
        --config-name pi0_industrial_sorting_joint_20260112 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 40000 \
        --inference-res-freq 30

elif [ "$TASK" == "21" ]; then
    CUDA_VISIBLE_DEVICES=1
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260112 && \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
        pi05_industrial_sorting_joint_20260112 --exp-name=pi05_industrrial_sorting_20260122 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_20260112 --exp-name=pi05_industrrial_sorting_20260122 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260112/pi0_joint \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260112/pi05_industrrial_sorting_20260122 \
        --config-name pi05_industrial_sorting_joint_20260112 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 50000 \
        --inference-res-freq 30

elif [ "$TASK" == "22" ]; then
    CUDA_VISIBLE_DEVICES=2
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DEBUG_MODE=0 uv run python3 scripts/compute_norm_stats.py --config-name pi0_industrial_sorting_joint_20260112 && \
    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_20260112 --exp-name=pi05_industrrial_sorting_20260121 --overwrite

    # CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 DEBUG_MODE=0 uv run scripts/train.py \
    #     pi05_industrial_sorting_joint_20260112 --exp-name=pi05_industrrial_sorting_20260121 --resume

    bash /root/openpi/evaluate.sh \
        --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
        --dataset-dir /mnt/pfs/sorting_train_data/train_dataset_raw/industrial_sorting_clean_unzipped_20260112/pi0_joint \
        --model-root /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260112/pi05_industrrial_sorting_20260121 \
        --config-name pi05_industrial_sorting_joint_20260112 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False \
        --total-steps 40000 \
        --inference-res-freq 30

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
