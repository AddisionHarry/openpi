#!/usr/bin/env python3
"""
Plot observation.state dimension vs frame_index in one figure with three subplots:
1. Original and filtered curve with rising edges
2. Filtered difference curve with rising edges
3. Binary Schmitt trigger output (after absolute value) with rising edges

Usage:
python plot_state_edges_binary.py --parquet-file data.parquet --dim 0 --output result.png
"""

import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def read_state_slice(parquet_file: Path, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read specific dimension from 'observation.state' column."""
    df = pd.read_parquet(parquet_file)
    frame_indices, values = [], []
    for _, row in df.iterrows():
        obs = row["observation.state"]
        if isinstance(obs, str):
            obs = ast.literal_eval(obs)
        if dim >= len(obs):
            continue
        frame_indices.append(row["frame_index"])
        values.append(obs[dim])
    return np.array(frame_indices), np.array(values)

def lowpass_iir_zero_phase(x: np.ndarray, alpha: float=0.2) -> np.ndarray:
    """One-pole IIR low-pass zero-phase filter (forward-backward)."""
    y_fwd = np.zeros_like(x)
    y_fwd[0] = x[0]
    for i in range(1, len(x)):
        y_fwd[i] = alpha * x[i] + (1 - alpha) * y_fwd[i-1]
    y_bwd = np.zeros_like(x)
    y_bwd[-1] = x[-1]
    for i in range(len(x)-2, -1, -1):
        y_bwd[i] = alpha * x[i] + (1 - alpha) * y_bwd[i+1]
    return 0.5 * (y_fwd + y_bwd)

def schmitt_trigger(x: np.ndarray, th_high: float, th_low: float) -> np.ndarray:
    """Dual-threshold Schmitt trigger to 1D array. Returns 0/1."""
    state = 0
    out = np.zeros_like(x, dtype=int)
    for i, val in enumerate(x):
        if state == 0 and val > th_high:
            state = 1
        elif state == 1 and val < th_low:
            state = 0
        out[i] = state
    return out

def compute_edges(values: np.ndarray,
                  alpha: float=0.2, th_high: float=0.01, th_low: float=0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute filtered curve, filtered difference, rising edge indices, and binary trigger output.
    Returns:
        filt: zero-phase filtered values
        diff_filt: filtered difference
        rising_idx: indices of rising edges
        trigger_bin: Schmitt trigger binary output (0/1)
    """
    filt = lowpass_iir_zero_phase(values, alpha=alpha)
    diff = np.diff(filt, prepend=filt[0])
    diff_filt = lowpass_iir_zero_phase(diff, alpha=alpha)
    trigger_bin = schmitt_trigger(np.abs(diff_filt), th_high=th_high, th_low=th_low)
    rising_idx = np.where(np.diff(trigger_bin) == 1)[0] + 1
    return filt, diff_filt, rising_idx, trigger_bin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-file", required=True, type=Path)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--th-high", type=float, default=0.01)
    parser.add_argument("--th-low", type=float, default=0.001)
    args = parser.parse_args()

    frame_idx, values = read_state_slice(args.parquet_file, args.dim)
    filt, diff_filt, rising_idx, trigger_bin = compute_edges(values, alpha=args.alpha,
                                                            th_high=args.th_high, th_low=args.th_low)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Top subplot: original and filtered
    ax1.plot(frame_idx, values, color="blue", label="Original")
    ax1.plot(frame_idx, filt, color="green", lw=2, label="Filtered")
    for idx in rising_idx:
        ax1.axvline(x=frame_idx[idx], color="red", linestyle="--", alpha=0.7)
    ax1.set_ylabel(f"Dim {args.dim} Value")
    ax1.set_title(f"Dim {args.dim} Original & Filtered with Rising Edges")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="best")

    # Middle subplot: filtered difference
    ax2.plot(frame_idx, diff_filt, color="purple", lw=2, label="Filtered Difference")
    for idx in rising_idx:
        ax2.axvline(x=frame_idx[idx], color="red", linestyle="--", alpha=0.7)
    ax2.set_ylabel(f"Dim {args.dim} Difference")
    ax2.set_title(f"Dim {args.dim} Filtered Difference with Rising Edges")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="best")

    # Bottom subplot: binary trigger output
    ax3.step(frame_idx, trigger_bin, where='post', color="orange", lw=2, label="Binary Trigger")
    for idx in rising_idx:
        ax3.axvline(x=frame_idx[idx], color="red", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Binary Output")
    ax3.set_title(f"Dim {args.dim} Binary Schmitt Trigger Output")
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc="best")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved combined figure: {args.output}")

if __name__ == "__main__":
    main()
