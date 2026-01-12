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
        --dataset-dir "/root/openpi/assets/pi0_zjhumanoid_breaker_placement/zj-humanoid/breaker_placement_20260108/pi0_breaker_placement_joint_20260108" \
        --episode-index 10 \
        --use-arms "[False, True]" \
        --use-waist-angles False \
        --use-tcp-pose False
else
    echo "[INFO] Normal inference server.py"
    CUDA_VISIBLE_DEVICES=2 TORCHINDUCTOR_COMPILE_THREADS=1 DEBUG_MODE=0 \
        uv run python3 ${SCRIPT_DIR}/scripts/inference.py \
        --model_path /root/openpi/checkpoints/pi0_breaker_placement_joint_20260108/pi0_breaker_placement_20260109/25000 \
        --device cuda:0 \
        --config_name pi0_breaker_placement_joint_20260108 \
        --port 10043
fi
