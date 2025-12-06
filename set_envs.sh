#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")" || exit

export ROOT="/mnt/pfs/openpi_train_data/huggingface_root"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

export XDG_CACHE_HOME="$HF_HOME"

export OPENPI_DATA_HOME="$ROOT/.cache/openpi"

export TMPDIR="$ROOT/.tmp"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

export UV_PYTHON=/root/miniforge/bin/python3

echo "HuggingFace and LeRobot environment variables set."

