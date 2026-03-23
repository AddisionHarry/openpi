#!/usr/bin/env python3
"""
check_compute_episodes_stats.py

Compute per-episode statistics from LeRobot v2.1 episode parquet files.

This script recursively scans the given input directory for
`episode_*.parquet` files (including nested chunk directories such as
`chunk-000`, `chunk-001`, etc.), computes statistics for numeric columns
in each episode, and writes the results to a JSONL file:

    episodes_stats.jsonl

Each line in the output file corresponds to one episode and contains:
    {
        "episode_index": <int>,
        "stats": { ... }
    }

Command-line usage:
    python check_compute_episodes_stats.py \
        --input_dir </path/to/dataset/data> \
        --output_dir </path/to/dataset/meta> \
        [--force]

External interface usage:
    from check_compute_episodes_stats import check_compute_episodes_stats_func

    success = check_compute_episodes_stats_func(
        input_dir="/path/to/dataset/data",
        output_dir="/path/to/dataset/meta",
        force=False
    )
"""


import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import re

def extract_episode_index(path: str) -> int:
    m = re.search(r"episode_(\d+)\.parquet", path)
    return int(m.group(1)) if m else -1

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

def collect_parquet_files(input_dir: str) -> list:
    """Recursively collect all episode_*.parquet files."""
    parquet_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.startswith("episode_") and f.endswith(".parquet"):
                parquet_files.append(os.path.join(root, f))
    parquet_files.sort(key=extract_episode_index)
    return parquet_files

def check_compute_episodes_stats_func(input_dir: str, output_dir: str, force: bool = False) -> bool:
    """
    Compute per-episode statistics from recursively collected
    `episode_*.parquet` files.

    The function scans `input_dir` recursively (e.g., across multiple
    `chunk-*` subdirectories), extracts numeric columns from each episode
    file, computes min/max/mean/std/count statistics, and writes results
    to `episodes_stats.jsonl` in `output_dir`.

    Parameters
    ----------
    input_dir : str
        Root directory containing episode parquet files.
        The directory may include nested chunk subdirectories
        (e.g., chunk-000, chunk-001, ...).

    output_dir : str
        Directory where `episodes_stats.jsonl` will be written.

    force : bool
        If True, overwrite the existing stats file.
        If False and the file already exists, no computation is performed.

    Returns
    -------
    bool
        True if:
            - The stats file was successfully written, or
            - The file already exists and force=False.

        False only if:
            - No valid episode files were found, or
            - An unrecoverable error occurred.
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

    parquet_files = collect_parquet_files(input_dir)
    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    with open(output_file, "w") as fout:
        for fpath in tqdm(parquet_files, desc="Processing episodes"):
            idx = extract_episode_index(fpath)
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
