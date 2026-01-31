#!/usr/bin/env python3
"""
Plot observation.state for specific dimensions vs frame_index in one figure per dimension:

1. Original & filtered curves per dimension with rising edges
2. Filtered difference curves per dimension with rising edges
3. Binary Schmitt trigger outputs per dimension with rising edges

Also computes the mean rising edge index across dimensions that have triggers and
plots it as a vertical line on all subplots.

Usage:
python plot_state_slice_edges.py --parquet-file data.parquet --dims 2,4,5 --output-dir results
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# --------- Utility functions ---------
def read_state_dims(parquet_file: Path, dims: list[int]):
    """Read specific dimensions from observation.state as numpy array."""
    df = pd.read_parquet(parquet_file)
    frame_idx, values_list = [], []
    for _, row in df.iterrows():
        obs = row["observation.state"]
        if isinstance(obs, str):
            obs = ast.literal_eval(obs)
        frame_idx.append(row["frame_index"])
        values_list.append([obs[d] for d in dims])
    return np.array(frame_idx), np.array(values_list)

def lowpass_iir_zero_phase(x, alpha=0.2):
    y_fwd = np.zeros_like(x)
    y_fwd[0] = x[0]
    for i in range(1, len(x)):
        y_fwd[i] = alpha*x[i] + (1-alpha)*y_fwd[i-1]
    y_bwd = np.zeros_like(x)
    y_bwd[-1] = x[-1]
    for i in range(len(x)-2, -1, -1):
        y_bwd[i] = alpha*x[i] + (1-alpha)*y_bwd[i+1]
    return 0.5*(y_fwd + y_bwd)

def schmitt_trigger(x, th_high=0.01, th_low=0.001):
    state = 0
    out = np.zeros_like(x, dtype=int)
    for i, val in enumerate(x):
        if state==0 and val>th_high:
            state=1
        elif state==1 and val<th_low:
            state=0
        out[i]=state
    return out

def compute_edges_per_dim(values, alpha=0.2, th_high=0.01, th_low=0.001):
    N_frames, N_dims = values.shape
    filt = np.zeros_like(values)
    diff_filt = np.zeros_like(values)
    trigger_bin = np.zeros_like(values, dtype=int)
    rising_idx_list = []

    for d in range(N_dims):
        x = values[:, d]
        x_filt = lowpass_iir_zero_phase(x, alpha=alpha)
        diff = np.diff(x_filt, prepend=x_filt[0])
        diff_f = lowpass_iir_zero_phase(diff, alpha=alpha)
        trigger = schmitt_trigger(diff_f, th_high=th_high, th_low=th_low)
        rising_idx = np.where(np.diff(trigger)==1)[0] + 1

        filt[:, d] = x_filt
        diff_filt[:, d] = diff_f
        trigger_bin[:, d] = trigger
        rising_idx_list.append(rising_idx)

    return filt, diff_filt, trigger_bin, rising_idx_list

# --------- Main plotting ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-file", required=True, type=Path)
    parser.add_argument("--dims", required=True, type=str,
                        help="Comma-separated list of dimensions, e.g., 2,4,5")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--th-high", type=float, default=0.025)
    parser.add_argument("--th-low", type=float, default=0.01)
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(',')]
    frame_idx, values = read_state_dims(args.parquet_file, dims)
    filt, diff_filt, trigger_bin, rising_idx_list = compute_edges_per_dim(values,
                                                                         alpha=args.alpha,
                                                                         th_high=args.th_high,
                                                                         th_low=args.th_low)

    args.output_dir.mkdir(exist_ok=True, parents=True)

    # compute mean rising index across dimensions that have triggers
    all_rising = [r for r in rising_idx_list if len(r)>0]
    mean_frame = None
    if all_rising:
        combined = np.concatenate(all_rising)
        mean_idx = int(np.round(np.mean(combined)))
        mean_frame = frame_idx[mean_idx]

    print(f"Rising edge indices per dimension: {rising_idx_list}")
    if mean_frame is not None:
        print(f"Mean rising edge frame index across all dimensions: {mean_frame}")
    else:
        print("No rising edges detected in any dimension.")

    N_dims = values.shape[1]
    colors = plt.cm.tab10.colors

    for i, d in enumerate(dims):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

        # Original & filtered
        ax1.plot(frame_idx, values[:, i], color=colors[i % 10], alpha=0.5, label=f"Dim {d} Original")
        ax1.plot(frame_idx, filt[:, i], color=colors[i % 10], lw=2, label="Filtered")
        for idx in rising_idx_list[i]:
            ax1.axvline(x=frame_idx[idx], color=colors[i % 10], linestyle='--', alpha=0.7)
        if mean_frame is not None:
            ax1.axvline(x=mean_frame, color='black', linestyle=':', lw=2, label='Mean Rising')
        ax1.set_ylabel("Value")
        ax1.set_title(f"Dim {d} Original & Filtered with Rising Edges")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='best')

        # Filtered difference
        ax2.plot(frame_idx, diff_filt[:, i], color=colors[i % 10], lw=2, label="Filtered Diff")
        for idx in rising_idx_list[i]:
            ax2.axvline(x=frame_idx[idx], color=colors[i % 10], linestyle='--', alpha=0.7)
        if mean_frame is not None:
            ax2.axvline(x=mean_frame, color='black', linestyle=':', lw=2, label='Mean Rising')
        ax2.set_ylabel("Filtered Diff")
        ax2.set_title(f"Dim {d} Filtered Difference with Rising Edges")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='best')

        # Binary trigger
        ax3.step(frame_idx, trigger_bin[:, i], where='post', color=colors[i % 10], lw=2, label='Binary Trigger')
        for idx in rising_idx_list[i]:
            ax3.axvline(x=frame_idx[idx], color=colors[i % 10], linestyle='--', alpha=0.7)
        if mean_frame is not None:
            ax3.axvline(x=mean_frame, color='black', linestyle=':', lw=2, label='Mean Rising')
        ax3.set_xlabel("Frame Index")
        ax3.set_ylabel("Binary Trigger")
        ax3.set_title(f"Dim {d} Binary Trigger Output")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='best')

        plt.tight_layout()
        save_path = args.output_dir / f"dim_{d}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
