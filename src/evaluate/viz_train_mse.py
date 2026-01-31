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


def extract_step_from_name(name: str) -> int:
    """Extract integer training step from folder name."""
    return int(name)

def load_episode_mse(h5_path: Path) -> np.ndarray | None:
    """Load episode MSE from evaluate.h5, skip if missing or invalid."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "episode_mse_summary/mean" not in f:
                print(f"[WARN] Missing episode_mse_summary/mean in {h5_path}")
                return None
            mse = np.asarray(f["episode_mse_summary/mean"][:], dtype=np.float64)
            mse = mse[np.isfinite(mse)]
            if mse.size == 0:
                print(f"[WARN] Empty or NaN-only MSE in {h5_path}")
                return None
            return mse
    except Exception as e:
        print(f"[WARN] Failed to read {h5_path}: {e}")
        return None


def gather_stats(eval_dir: Path):
    """Gather statistics: mean, median, quantiles, min, max for all valid steps."""
    step_dirs = sorted([p for p in eval_dir.iterdir() if p.is_dir() and (p / "evaluate.h5").exists()],
                       key=lambda p: extract_step_from_name(p.name))
    steps, mean, median, q25, q75, vmin, vmax = [], [], [], [], [], [], []
    for step_dir in step_dirs:
        step = extract_step_from_name(step_dir.name)
        mse = load_episode_mse(step_dir / "evaluate.h5")
        if mse is None:
            print(f"[SKIP] step {step}")
            continue
        steps.append(step)
        mean.append(float(np.mean(mse)))
        median.append(float(np.median(mse)))
        q25.append(float(np.percentile(mse, 25)))
        q75.append(float(np.percentile(mse, 75)))
        vmin.append(float(np.min(mse)))
        vmax.append(float(np.max(mse)))
    if not steps:
        raise RuntimeError("No valid evaluation data found.")
    return np.array(steps), np.array(mean), np.array(median), np.array(q25), np.array(q75), np.array(vmin), np.array(vmax)


def plot_mse_two_subplots(steps, mean, median, q25, q75, vmin, vmax, save_path: Path, dpi=300):
    """Plot two subplots: first with 25-75 percentile, median, mean; second with min-max included."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    # Subplot 1: 25-75 percentile + median + mean
    ax = axes[0]
    ax.fill_between(steps, q25, q75, alpha=0.3, color="#1f77b4", label="25–75 percentile")
    ax.plot(steps, median, linewidth=2, label="Median", color="#ff7f0e")
    ax.plot(steps, mean, linestyle="--", linewidth=2, label="Mean", color="#2ca02c")
    ax.set_title("25–75 percentile, Median & Mean", fontsize=14, fontweight='bold')
    ax.set_ylabel("Episode MSE", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    # Subplot 2: min-max + 25-75 percentile + median + mean
    ax = axes[1]
    ax.fill_between(steps, vmin, vmax, alpha=0.15, color="#d62728", label="Min–Max")
    ax.fill_between(steps, q25, q75, alpha=0.35, color="#1f77b4", label="25–75 percentile")
    ax.plot(steps, median, linewidth=2, label="Median", color="#ff7f0e")
    ax.plot(steps, mean, linestyle="--", linewidth=2, label="Mean", color="#2ca02c")
    ax.set_title("Min–Max + 25–75 percentile, Median & Mean", fontsize=14, fontweight='bold')
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Episode MSE", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot MSE vs training step with two subplots")
    parser.add_argument("--eval-dir", required=True, type=str, help="Directory with step folders containing evaluate.h5")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(eval_dir)
    steps, mean, median, q25, q75, vmin, vmax = gather_stats(eval_dir)
    out_file = eval_dir / "mse_curve.png"
    plot_mse_two_subplots(steps, mean, median, q25, q75, vmin, vmax, out_file, dpi=args.dpi)
    print(f"[OK] Saved plot to {out_file}")

if __name__ == "__main__":
    main()
