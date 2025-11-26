#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
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
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std": np.nanstd(arr, axis=0).tolist(),
        "min": np.nanmin(arr, axis=0).tolist(),
        "max": np.nanmax(arr, axis=0).tolist(),
        "count": [arr.shape[0]],
    }


def process_episode(parquet_path: str):
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



def main():
    parser = argparse.ArgumentParser(description="Compute LeRobot v2.1-compatible stats file (flat keys).")
    parser.add_argument("--input_dir", required=True, help="Path to directory with episode_*.parquet")
    parser.add_argument("--output_path", default="episodes_stats.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    parquet_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".parquet")
    ])

    print(f"Found {len(parquet_files)} parquet files in {args.input_dir}")

    with open(args.output_path, "w") as fout:
        for idx, fpath in enumerate(tqdm(parquet_files, desc="Processing episodes")):
            episode_stats = process_episode(fpath)
            fout.write(json.dumps({
                "episode_index": idx,
                "stats": episode_stats
            }) + "\n")

    print(f"Done. Wrote {len(parquet_files)} episodes to {args.output_path}")


if __name__ == "__main__":
    main()
