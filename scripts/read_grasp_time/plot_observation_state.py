#!/usr/bin/env python3
"""
Plot a specific dimension of 'observation.state' from a Parquet file.

Usage:
    python plot_observation_state.py --parquet-file data.parquet --dim 0 --output plot.png

Arguments:
    --parquet-file : path to the input Parquet file
    --dim          : which dimension of 'observation.state' to plot (0-based)
    --output       : output figure file path
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ast
from typing import List, Tuple

def read_observation_state(parquet_file: Path, dim: int) -> Tuple[List[int], List[float]]:
    """
    Read 'observation.state' dimension and frame_index from Parquet.

    Args:
        parquet_file: Path to the parquet file.
        dim: Dimension index of 'observation.state' to extract.

    Returns:
        Tuple of frame_index list and corresponding state values list.
    """
    df = pd.read_parquet(parquet_file)
    episodes = []
    state_vals = []
    for _, row in df.iterrows():
        # 'observation.state' could be a string, convert to list if necessary
        obs_state = row["observation.state"]
        if isinstance(obs_state, str):
            obs_state = ast.literal_eval(obs_state)
        if dim >= len(obs_state):
            continue
        episodes.append(row["frame_index"])
        state_vals.append(obs_state[dim])
    return episodes, state_vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-file", required=True, type=Path, help="Input Parquet file")
    parser.add_argument("--dim", required=True, type=int, help="Dimension index to plot")
    parser.add_argument("--output", required=True, type=Path, help="Output figure file path")
    args = parser.parse_args()

    episodes, state_vals = read_observation_state(args.parquet_file, args.dim)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, state_vals, marker='o', linestyle='-', color='blue')
    plt.xlabel("Frame Index")
    plt.ylabel(f"Observation State Dimension {args.dim}")
    plt.title(f"Observation State Dimension {args.dim} vs Frame Index")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved figure to {args.output}")

if __name__ == "__main__":
    main()
