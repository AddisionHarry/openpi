#! /bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"

TEST_DISPLAY=0
for arg in "$@"; do
    case $arg in
        --test-display-dataset)
            TEST_DISPLAY=1
            shift
            ;;
        *)
            echo "Unkown param: $arg"
            ;;
    esac
done

if [ $TEST_DISPLAY -eq 1 ]; then
    echo "[INFO] Run test_display_dataset.py"
    DEBUG_MODE=0 uv run python3 /root/openpi/src/evaluate/display_dataset.py \
        --host "0.0.0.0" \
        --port 10043 \
        --chunk-size 50 \
        --dataset-action-fps 30 \
        --dataset-dir "/root/openpi/assets/pi05_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20260125/pi05_industrial_sorting_joint_20260125" \
        --episode-index 50 \
        --use-arms "[False, True]" \
        --use-waist-angles True \
        --use-tcp-pose False
else
    echo "[INFO] Normal inference server.py"
    # CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 DEBUG_MODE=0 \
    #     uv run python3 ${SCRIPT_DIR}/scripts/inference.py \
    #     --model_path /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images/pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images/15000 \
    #     --device cuda:0 \
    #     --config_name pi05_industrial_sorting_joint_20260131_last_frames_still_grasp_noise_chest_images \
    #     --port 10043
    CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 DEBUG_MODE=0 \
        uv run python3 ${SCRIPT_DIR}/scripts/inference.py \
        --model_path /root/openpi/checkpoints/pi05_industrial_sorting_joint_20260126/pi05_industrial_sorting_20260128/13000 \
        --device cuda:0 \
        --config_name pi05_industrial_sorting_joint_20260126 \
        --port 10043
    # CUDA_VISIBLE_DEVICES=2 TORCHINDUCTOR_COMPILE_THREADS=1 DEBUG_MODE=0 \
    #     uv run python3 ${SCRIPT_DIR}/scripts/inference.py \
    #     --model_path /root/openpi/checkpoints/pi0_industrial_sorting_joint_20260126/pi0_industrial_sorting_20260128/20000 \
    #     --device cuda:0 \
    #     --config_name pi0_industrial_sorting_joint_20260126 \
    #     --port 10043
fi
