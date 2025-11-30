#!/usr/bin/env python3
"""
check_compute_episodes_stats.py

Script to compute per-episode statistics from parquet files for LeRobot v2.1 format.
Generates `episodes_stats.jsonl` in the specified output directory.

Command-line usage:
    python check_compute_episodes_stats.py --input_dir </path/to/parquets> --output_dir </path/to/output> [--force]

External interface usage:
    from check_compute_episodes_stats import check_compute_episodes_stats
    success = check_compute_episodes_stats_func(
        input_dir="/path/to/parquets",
        output_dir="/path/to/output",
        force=False
    )
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def flatten_dict(d):
    """Flatten nested dicts into a 1D array preserving key order."""
    if not isinstance(d, dict):
        return np.array(d)
    keys = sorted(d.keys())
    return np.array([d[k] for k in keys])

def safe_stack(series):
    """Safely stack a pandas Series into a numeric numpy array."""
    arrs = []
    for item in series:
        if isinstance(item, dict):
            arrs.append(flatten_dict(item))
        else:
            arrs.append(np.array(item))
    return np.stack(arrs)

def compute_stats(arr: np.ndarray):
    """Compute mean/std/min/max for numeric numpy array."""
    if len(arr.shape) == 1:
        arr = arr[:, None]
    return {
        "min": np.nanmin(arr, axis=0).tolist(),
        "max": np.nanmax(arr, axis=0).tolist(),
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std": np.nanstd(arr, axis=0).tolist(),
        "count": [arr.shape[0]],
    }

def process_episode(parquet_path: str):
    """Process a single parquet episode and compute stats for numeric columns."""
    df = pd.read_parquet(parquet_path)
    stats = {}

    valid_cols = [
        c for c in df.columns
        if not any(x in c.lower() for x in ["index", "timestamp", "frame", "episode"])
    ]

    for col in valid_cols:
        try:
            arr = safe_stack(df[col])
            if not np.issubdtype(arr.dtype, np.number):
                continue
            stats[col] = compute_stats(arr)
        except Exception as e:
            print(f"Skipping {col}: {e}")

    return stats

def check_compute_episodes_stats_func(input_dir: str, output_dir: str, force: bool = False) -> bool:
    """
    Compute episode stats from parquet files.

    Parameters
    ----------
    input_dir : str
        Directory containing episode_*.parquet files
    output_dir : str
        Directory to write episodes_stats.jsonl
    force : bool
        If True, overwrite existing stats file if present

    Returns
    -------
    bool
        True if stats file was created successfully (or already exists and no errors),
        False if output exists and force=False (check failed)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "episodes_stats.jsonl")

    if os.path.exists(output_file) and not force:
        print(f"   Output file already exists: {output_file}")
        print("    Use --force to overwrite.")
        return True
    else:
        if force:
            print(f"   Force enabled: overwriting {output_file}")
        else:
            print(f"Will write new stats file to: {output_file}")

    parquet_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ])
    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    with open(output_file, "w") as fout:
        for idx, fpath in enumerate(tqdm(parquet_files, desc="Processing episodes")):
            episode_stats = process_episode(fpath)
            fout.write(json.dumps({
                "episode_index": idx,
                "stats": episode_stats
            }) + "\n")

    print(f"Done. Wrote {len(parquet_files)} episodes to {output_file}")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute LeRobot v2.1-compatible stats file (flat keys).")
    parser.add_argument("--input_dir", required=True, help="Path to directory with episode_*.parquet")
    parser.add_argument("--output_dir", required=True, help="Directory to save episodes_stats.jsonl")
    parser.add_argument("--force", action="store_true", help="Force overwrite if stats file exists")
    args = parser.parse_args()

    check_compute_episodes_stats_func(args.input_dir, args.output_dir, force=args.force)

if __name__ == "__main__":
    main()
