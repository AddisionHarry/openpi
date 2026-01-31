#!/usr/bin/env python3
"""
Batch process parquet files: plot observation.state slice per file with rising edges.

- One figure per parquet file: three subplots (original+filtered, diff, binary trigger)
- Mean rising edge across dimensions drawn
- Generate grid thumbnail images: each grid 5x5 showing all dimensions per file
  - Only plot the **mean rising edge**, not individual triggers
  - Multiple grids if files >25
  - High DPI for clarity

Usage:
python batch_plot_state_slice.py --parquet-dir /path/to/parquet_files --dims 58,60,63 --output-dir viz_results
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import math

# --------- Utility functions ---------
def read_state_dims(parquet_file: Path, dims: list[int]):
    """Read specific dimensions from observation.state into numpy array (N_frames, N_dims)."""
    df = pd.read_parquet(parquet_file)
    frame_idx, values_list = [], []
    for _, row in df.iterrows():
        obs = row["observation.state"]
        if isinstance(obs, str):
            obs = ast.literal_eval(obs)
        frame_idx.append(row["frame_index"])
        values_list.append([obs[d] for d in dims])
    return np.array(frame_idx), np.array(values_list)

def lowpass_iir_zero_phase(x, alpha=0.3):
    y_fwd = np.zeros_like(x)
    y_fwd[0] = x[0]
    for i in range(1, len(x)):
        y_fwd[i] = alpha*x[i] + (1-alpha)*y_fwd[i-1]
    y_bwd = np.zeros_like(x)
    y_bwd[-1] = x[-1]
    for i in range(len(x)-2, -1, -1):
        y_bwd[i] = alpha*x[i] + (1-alpha)*y_bwd[i+1]
    return 0.5*(y_fwd + y_bwd)

def schmitt_trigger(x, th_high=0.025, th_low=0.01):
    state = 0
    out = np.zeros_like(x, dtype=int)
    for i, val in enumerate(x):
        if state==0 and val>th_high:
            state=1
        elif state==1 and val<th_low:
            state=0
        out[i]=state
    return out

def compute_edges_per_dim(values, alpha=0.3, th_high=0.025, th_low=0.01):
    N_frames, N_dims = values.shape
    filt = np.zeros_like(values)
    diff_filt = np.zeros_like(values)
    trigger_bin = np.zeros_like(values, dtype=int)
    rising_idx_list = []

    for d in range(N_dims):
        x = values[:, d]
        x_filt = lowpass_iir_zero_phase(x, alpha)
        diff = np.diff(x_filt, prepend=x_filt[0])
        diff_f = lowpass_iir_zero_phase(diff, alpha)
        trigger = schmitt_trigger(diff_f, th_high=th_high, th_low=th_low)
        rising_idx = np.where(np.diff(trigger)==1)[0] + 1

        filt[:, d] = x_filt
        diff_filt[:, d] = diff_f
        trigger_bin[:, d] = trigger
        rising_idx_list.append(rising_idx)

    return filt, diff_filt, trigger_bin, rising_idx_list

# --------- Plot per file ---------
def plot_file(frame_idx, values, filt, diff_filt, trigger_bin, rising_idx_list, dims, output_dir, file_stem):
    N_dims = values.shape[1]
    colors = plt.cm.tab10.colors
    output_dir.mkdir(exist_ok=True, parents=True)

    all_rising = [r for r in rising_idx_list if len(r)>0]
    if all_rising:
        combined = np.concatenate(all_rising)
        median_idx = int(np.round(np.median(combined)))
        median_frame = frame_idx[median_idx]
    else:
        median_frame = None

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for d in range(N_dims):
        color = colors[d % 10]
        ax1.plot(frame_idx, values[:, d], color=color, alpha=0.5, label=f"Dim {dims[d]}")
        ax1.plot(frame_idx, filt[:, d], color=color, lw=2)
        ax2.plot(frame_idx, diff_filt[:, d], color=color, lw=2)
        ax3.step(frame_idx, trigger_bin[:, d], where='post', color=color, lw=2)

    if median_frame is not None:
        for ax in (ax1, ax2, ax3):
            ax.axvline(x=median_frame, color='black', linestyle=':', lw=2, label='Median Rising')

    ax1.set_ylabel("Value")
    ax1.set_title(f"{file_stem}: Original & Filtered")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=6)

    ax2.set_ylabel("Diff")
    ax2.set_title("Filtered Difference")
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax3.set_ylabel("Binary Trigger")
    ax3.set_title("Binary Trigger Output")
    ax3.set_xlabel("Frame Index")
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = output_dir / f"{file_stem}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    return frame_idx, values, median_frame

# --------- Thumbnail grid ---------
def plot_thumbnail_grid(summary_list, output_dir, grid_rows=5, grid_cols=5, dpi=300):
    output_dir.mkdir(exist_ok=True, parents=True)
    n_files = len(summary_list)
    n_per_grid = grid_rows * grid_cols
    n_grids = math.ceil(n_files / n_per_grid)

    for g in range(n_grids):
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*3, grid_rows*2.5), sharex=True, sharey=True)
        axes = axes.flatten()
        for i in range(n_per_grid):
            idx = g*n_per_grid + i
            if idx >= n_files:
                axes[i].axis('off')
                continue
            file_name, frame_idx, values, mean_frame = summary_list[idx]
            N_dims = values.shape[1]
            colors = plt.cm.tab10.colors
            for d in range(N_dims):
                color = colors[d % 10]
                axes[i].plot(frame_idx, values[:, d], color=color, lw=1.5, alpha=0.7)
            if mean_frame is not None:
                axes[i].axvline(x=mean_frame, color='black', linestyle=':', lw=2, label='Mean Rising')
            axes[i].set_title(file_name, fontsize=6)
            axes[i].grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        save_path = output_dir / f"thumbnail_grid_{g+1}.png"
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        print(f"Saved thumbnail grid: {save_path}")

# --------- Main ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--dims", required=True, type=str,
                        help="Comma-separated list of dimensions, e.g., 58,60,63")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--th-high", type=float, default=0.025)
    parser.add_argument("--th-low", type=float, default=0.01)
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(',')]
    args.output_dir.mkdir(exist_ok=True, parents=True)
    parquet_files = sorted(args.parquet_dir.glob("*.parquet"))

    summary_list = []
    all_rising_frames = []

    for pf in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            frame_idx, values = read_state_dims(pf, dims)
            filt, diff_filt, trigger_bin, rising_idx_list = compute_edges_per_dim(values, alpha=args.alpha,
                                                                                  th_high=args.th_high, th_low=args.th_low)
            frame_idx_all, values_all, mean_frame = plot_file(frame_idx, values, filt, diff_filt, trigger_bin,
                                                              rising_idx_list, dims, args.output_dir, pf.stem)
            for d, rising_idx in enumerate(rising_idx_list):
                if len(rising_idx) > 0:
                    all_rising_frames.extend(frame_idx[rising_idx])
            frame_idx_all, values_all, median_frame = plot_file(
                frame_idx, values, filt, diff_filt, trigger_bin,
                rising_idx_list, dims, args.output_dir, pf.stem
            )
            summary_list.append((pf.stem, frame_idx_all, values_all, mean_frame))
        except Exception as e:
            print(f"Failed to process {pf}: {e}")

    plot_thumbnail_grid(summary_list, args.output_dir, grid_rows=5, grid_cols=5, dpi=300)

    if len(all_rising_frames) == 0:
        print("No rising edges detected in the entire dataset.")
    else:
        all_rising_frames = np.asarray(all_rising_frames)
        mean_val = np.mean(all_rising_frames)
        var_val = np.var(all_rising_frames)
        median_val = np.median(all_rising_frames)
        q1 = np.percentile(all_rising_frames, 25)
        q3 = np.percentile(all_rising_frames, 75)
        print("\n=== Rising Edge Frame Statistics (Dataset-level) ===")
        print(f"Count        : {len(all_rising_frames)}")
        print(f"Mean         : {mean_val:.3f}")
        print(f"Variance     : {var_val:.3f}")
        print(f"Median       : {median_val:.3f}")
        print(f"25% Quantile : {q1:.3f}")
        print(f"75% Quantile : {q3:.3f}")

        # ===== Histogram of rising edge frames =====
        fig, ax = plt.subplots(figsize=(10, 6))
        # Histogram
        ax.hist(all_rising_frames, bins=50, density=False, alpha=0.75, edgecolor="black")
        # Vertical lines
        ax.axvline(mean_val, color="red", linestyle="--", lw=2, label="Mean")
        ax.axvline(median_val, color="black", linestyle=":", lw=2, label="Median")
        ax.axvline(q1, color="blue", linestyle="--", lw=1.5, label="Q1 (25%)")
        ax.axvline(q3, color="blue", linestyle="--", lw=1.5, label="Q3 (75%)")
        # Labels & title
        ax.set_xlabel("Frame Index of Rising Edge")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Rising Edge Frames (Dataset-level)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        # Statistics text box
        textstr = (
            f"Count   = {len(all_rising_frames)}\n"
            f"Mean    = {mean_val:.2f}\n"
            f"Var     = {var_val:.2f}\n"
            f"Median  = {median_val:.2f}\n"
            f"Q1      = {q1:.2f}\n"
            f"Q3      = {q3:.2f}"
        )
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        # Save
        hist_path = args.output_dir / "rising_edge_histogram.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"Saved rising edge histogram to {hist_path}")


if __name__ == "__main__":
    main()
