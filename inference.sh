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
    DEBUG_MODE=0 uv run python3 /root/openpi/scripts/test/test_display_dataset.py \
        --host "0.0.0.0" \
        --port 10043 \
        --chunk-size 50 \
        --dataset-action-fps 30 \
        --dataset-dir "/root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_cleaned_20251214/pi0_industrial_sorting_joint_waist" \
        --episode-index 10 \
        --use-arms "[False, True]" \
        --use-waist-angles False \
        --use-tcp-pose False
else
    echo "[INFO] Normal inference server.py"
    CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=1 DEBUG_MODE=0 \
        uv run python3 ${SCRIPT_DIR}/src/inference/server.py \
        --model_path /root/openpi/checkpoints/pi0_industrial_sorting_joint_waist/pi0_industrial_sorting_waist_action_1214data_20251220/30000 \
        --device cuda:0 \
        --config_name pi0_industrial_sorting_joint_waist \
        --port 10043
fi
