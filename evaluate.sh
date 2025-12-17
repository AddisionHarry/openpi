#! /bin/bash
# ==============================================================================
# Evaluation script for model checkpoints.
#
# This script evaluates multiple training checkpoints on a fixed dataset.
# Checkpoints are assumed to be stored under:
#   MODEL_ROOT/<step>/
#
# Evaluation steps:
#   - Every STEP_INTERVAL (default: 5000)
#   - Always include the final checkpoint: TOTAL_STEPS - 1
#
# Required arguments:
#   --cuda-visible-devices <id>   Visible CUDA devices (e.g. "0")
#   --dataset-dir <path>          Evaluation dataset directory
#   --model-root <path>           Root directory containing step-indexed checkpoints
#   --config-name <name>          Model config name
#   --use-arms "[False, True]"    Must be quoted
#   --use-waist-angles <True|False>
#   --use-tcp-pose <True|False>
#   --total-steps <int>           Total training steps
# Example Usage:
#   bash evaluate.sh \
#     --cuda-visible-devices 0 \
#     --dataset-dir /path/to/dataset \
#     --model-root /path/to/checkpoints \
#     --config-name config_name \
#     --use-arms "[False, True]" \
#     --use-waist-angles True \
#     --use-tcp-pose False \
#     --total-steps 20000
#
# Notes:
#   - --device cuda:0 refers to the first visible device after CUDA_VISIBLE_DEVICES
#   - Missing checkpoints are skipped with a warning
#   - set_envs.sh is sourced for shared environment setup
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_envs.sh"

STEP_INTERVAL=5000
DEVICE="cuda:0"   # Note: cuda:0 refers to the first *visible* device after CUDA_VISIBLE_DEVICES

while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_ARG="$2"; shift 2 ;;
    --dataset-dir)
      DATASET_DIR="$2"; shift 2 ;;
    --model-root)
      MODEL_ROOT="$2"; shift 2 ;;
    --config-name)
      CONFIG_NAME="$2"; shift 2 ;;
    --use-arms)
      USE_ARMS="$2"; shift 2 ;;
    --use-waist-angles)
      USE_WAIST_ANGLES="$2"; shift 2 ;;
    --use-tcp-pose)
      USE_TCP_POSE="$2"; shift 2 ;;
    --total-steps)
      TOTAL_STEPS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

: "${CUDA_VISIBLE_DEVICES_ARG:?Missing --cuda-visible-devices}"
: "${DATASET_DIR:?Missing --dataset-dir}"
: "${MODEL_ROOT:?Missing --model-root}"
: "${CONFIG_NAME:?Missing --config-name}"
: "${USE_ARMS:?Missing --use-arms}"
: "${USE_WAIST_ANGLES:?Missing --use-waist-angles}"
: "${USE_TCP_POSE:?Missing --use-tcp-pose}"
: "${TOTAL_STEPS:?Missing --total-steps}"

if [[ "$USE_ARMS" != "["*"]" ]]; then
  echo "[ERROR] --use-arms must be quoted, e.g. \"[False, True]\""
  exit 1
fi

STEPS=()
for ((s=STEP_INTERVAL; s<TOTAL_STEPS; s+=STEP_INTERVAL)); do
  STEPS+=("$s")
done

LAST_STEP=$((TOTAL_STEPS - 1))
if [[ ${#STEPS[@]} -eq 0 || "${STEPS[${#STEPS[@]}-1]}" != "$LAST_STEP" ]]; then
  STEPS+=("$LAST_STEP")
fi

echo "Resolved arguments:"
echo "  CUDA_VISIBLE_DEVICES_ARG = ${CUDA_VISIBLE_DEVICES_ARG}"
echo "  DATASET_DIR              = ${DATASET_DIR}"
echo "  MODEL_ROOT               = ${MODEL_ROOT}"
echo "  TOTAL_STEPS              = ${TOTAL_STEPS}"
echo "Will evaluate steps: ${STEPS[@]}"
echo

for STEP in "${STEPS[@]}"; do
  MODEL_PATH="${MODEL_ROOT}/${STEP}"
  OUTPUT_PATH="${MODEL_ROOT}/eval_results/${STEP}/evaluate.h5"
  mkdir -p "$(dirname "${OUTPUT_PATH}")"

  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[WARN] Skip missing checkpoint: ${MODEL_PATH}"
    continue
  fi

  echo "=============================================="
  echo "[INFO] Running evaluation for step ${STEP}"
  echo "Model path : ${MODEL_PATH}"
  echo "Output path: ${OUTPUT_PATH}"
  echo "=============================================="

  DEBUG_MODE=0 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_ARG}" uv run python3 /root/openpi/src/evaluate/eval_dataset.py \
    --dataset-dir "${DATASET_DIR}" \
    --episode-all \
    --model-path "${MODEL_PATH}" \
    --config-name "${CONFIG_NAME}" \
    --output-path "${OUTPUT_PATH}" \
    --device "${DEVICE}" \
    --use-arms "${USE_ARMS}" \
    --use-waist-angles "${USE_WAIST_ANGLES}" \
    --use-tcp-pose "${USE_TCP_POSE}"
done

# DEBUG_MODE=0 CUDA_VISIBLE_DEVICES=0 uv run python3 /root/openpi/scripts/test/test_eval_dataset.py \
#     --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_manually_cleaned_20251210/pi0_zjhumanoid_industrial_sorting_jax \
#     --episode-all \
#     --model-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/5000 \
#     --config-name pi0_zjhumanoid_industrial_sorting_jax \
#     --output-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/5000/evaluate.h5 \
#     --device cuda:0 \
#     --use-arms "[False, True]" --use-waist-angles True --use-tcp-pose False

# DEBUG_MODE=0 CUDA_VISIBLE_DEVICES=0 uv run python3 /root/openpi/scripts/test/test_eval_dataset.py \
#     --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_manually_cleaned_20251210/pi0_zjhumanoid_industrial_sorting_jax \
#     --episode-all \
#     --model-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/10000 \
#     --config-name pi0_zjhumanoid_industrial_sorting_jax \
#     --output-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/10000/evaluate.h5 \
#     --device cuda:0 \
#     --use-arms "[False, True]" --use-waist-angles True --use-tcp-pose False

# DEBUG_MODE=0 CUDA_VISIBLE_DEVICES=0 uv run python3 /root/openpi/scripts/test/test_eval_dataset.py \
#     --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_manually_cleaned_20251210/pi0_zjhumanoid_industrial_sorting_jax \
#     --episode-all \
#     --model-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/15000 \
#     --config-name pi0_zjhumanoid_industrial_sorting_jax \
#     --output-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/15000/evaluate.h5 \
#     --device cuda:0 \
#     --use-arms "[False, True]" --use-waist-angles True --use-tcp-pose False

# DEBUG_MODE=0 CUDA_VISIBLE_DEVICES=0 uv run python3 /root/openpi/scripts/test/test_eval_dataset.py \
#     --dataset-dir /root/openpi/assets/pi0_zjhumanoid_industrial_sorting/zj-humanoid/industrial_sorting_manually_cleaned_20251210/pi0_zjhumanoid_industrial_sorting_jax \
#     --episode-all \
#     --model-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/19999 \
#     --config-name pi0_zjhumanoid_industrial_sorting_jax \
#     --output-path /root/openpi/checkpoints/pi0_zjhumanoid_industrial_sorting_jax/industrial_sorting_cleaned_waist_action/19999/evaluate.h5 \
#     --device cuda:0 \
#     --use-arms "[False, True]" --use-waist-angles True --use-tcp-pose False
