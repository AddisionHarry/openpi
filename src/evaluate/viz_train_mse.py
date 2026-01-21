#!/usr/bin/env python3
"""
File: viz_train_mse.py
Description:
    This script visualizes the mean squared error (MSE) of OpenPI model evaluations
    across different training steps. It reads 'evaluate.h5' files under a given
    eval_results directory, computes mean and 25th/75th percentiles for MSE, and
    generates a line plot showing MSE trend with training steps.

Usage:
    python3 viz_train_mse.py --eval-dir /path/to/eval_results

Arguments:
    --eval-dir : str
        Path to the directory containing step folders, each with 'evaluate.h5'.
        The plot will be saved as 'mse_curve.png' in the same directory.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple

def extract_step_from_name(name: str) -> int:
    """Extract training step integer from folder name."""
    return int(name)

def load_mse_from_h5(h5_path: Path) -> np.ndarray:
    """
    Load MSE data from an evaluate.h5 file.

    Args:
        h5_path (Path): Path to the HDF5 file.

    Returns:
        np.ndarray: 1D array of MSE values.
    """
    with h5py.File(h5_path, "r") as f:
        if "episode_mse_summary/mean" in f:
            mse_data = f["episode_mse_summary/mean"][:]
        else:
            episode_keys = [k for k in f.keys() if k.startswith("episode_")]
            all_episode_mse = []
            for ep in episode_keys:
                if "action_mse" in f[ep]:
                    all_episode_mse.append(f[ep]["action_mse"][:])
            if all_episode_mse:
                mse_data = np.concatenate(all_episode_mse)
            else:
                mse_data = np.array([np.nan])
        return mse_data

def gather_mse_stats(eval_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gather mean and quantile MSE statistics from all steps.

    Args:
        eval_dir (Path): Directory containing step folders with 'evaluate.h5'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            steps, mean_mse, q25_mse, q75_mse
    """
    h5_files = sorted(
        [p / "evaluate.h5" for p in eval_dir.iterdir() if p.is_dir() and (p / "evaluate.h5").exists()],
        key=lambda x: extract_step_from_name(x.parent.name)
    )

    steps: List[int] = []
    mean_mse: List[float] = []
    q25_mse: List[float] = []
    q75_mse: List[float] = []

    for h5_file in h5_files:
        step = extract_step_from_name(h5_file.parent.name)
        steps.append(step)
        mse_data = load_mse_from_h5(h5_file)
        mean_mse.append(float(np.nanmean(mse_data)))
        q25_mse.append(float(np.nanpercentile(mse_data, 25)))
        q75_mse.append(float(np.nanpercentile(mse_data, 75)))

    return np.array(steps), np.array(mean_mse), np.array(q25_mse), np.array(q75_mse)

def plot_mse_curve(
    steps: np.ndarray,
    mean_mse: np.ndarray,
    q25_mse: np.ndarray,
    q75_mse: np.ndarray,
    save_path: Path,
    dpi: int = 300
) -> None:
    """
    Plot MSE curve with mean and 25-75 percentile shaded area, then save to file.

    Args:
        steps (np.ndarray): Training steps.
        mean_mse (np.ndarray): Mean MSE per step.
        q25_mse (np.ndarray): 25th percentile MSE.
        q75_mse (np.ndarray): 75th percentile MSE.
        save_path (Path): Output PNG file path.
        dpi (int): Resolution of saved image, default is 300.
    """
    plt.figure(figsize=(10,6))
    plt.plot(steps, mean_mse, label="Mean MSE", color="blue", marker='o')
    plt.fill_between(steps, q25_mse, q75_mse, color="blue", alpha=0.2, label="25-75 percentile", edgecolor='none')
    plt.xlabel("Training Step")
    plt.ylabel("MSE")
    plt.title("MSE vs Training Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MSE vs Training Step for OpenPI evaluations.")
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Path to the eval_results directory containing step folders with evaluate.h5")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {eval_dir}")

    steps, mean_mse, q25_mse, q75_mse = gather_mse_stats(eval_dir)

    output_file = eval_dir / "mse_curve.png"
    plot_mse_curve(steps, mean_mse, q25_mse, q75_mse, output_file, dpi=300)
    print(f"Saved MSE curve plot to {output_file}")

if __name__ == "__main__":
    main()
