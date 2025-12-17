#!/usr/bin/env python3
"""
check_task_index_consistency.py

Checks all parquet episode files in data/chunk-* directories to verify:

1. 'task_index' column exists.
2. All rows have the same task_index.
3. task_index matches the tasks described in meta/tasks.jsonl and meta/episodes.jsonl.

Optional fix mode:
- Missing task_index: auto-fill from episodes.jsonl -> tasks.jsonl.
- Prints errors if cannot fix automatically.

Command-line usage:
    python check_task_index_consistency.py --data-root </path/to/parquet> --meta-root </path/to/meta> [--fix]

External interface usage:
    from check_task_index_consistency import check_task_index_consistency_func
    success = check_task_index_consistency_func("/path/to/parquet", "/path/to/meta", fix=True)
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def check_task_index_consistency_func(data_root: str, meta_root: str, fix: bool = False) -> bool:
    """
    Checks and optionally fixes task_index for all parquet episode files.

    Parameters
    ----------
    data_root : str
        Root directory containing chunk-* subfolders with parquet files.
    meta_root : str
        Directory containing meta/tasks.jsonl and meta/episodes.jsonl.
    fix : bool
        Whether to attempt automatic fix.

    Returns
    -------
    bool
        True if all files pass check or are successfully fixed. False otherwise.
    """
    # Load meta info
    tasks_meta = load_jsonl(os.path.join(meta_root, "tasks.jsonl"))
    episodes_meta = load_jsonl(os.path.join(meta_root, "episodes.jsonl"))
    task_map = {t["task"]: t["task_index"] for t in tasks_meta}
    episode_map = {e["episode_index"]: e["tasks"] for e in episodes_meta}

    # Collect all parquet files
    data_root = Path(data_root)
    parquet_files = sorted(data_root.glob("chunk-*/*.parquet"))
    print(f"Found {len(parquet_files)} parquet files under {data_root}")

    error_count = 0

    for fpath in tqdm(parquet_files, desc="Checking parquet files"):
        df = pd.read_parquet(fpath)
        basename = fpath.name

        # Determine episode_index
        if "episode_index" not in df.columns:
            tqdm.write(f"[ERROR] {basename} missing 'episode_index' column")
            error_count += 1
            continue

        episode_idx_set = set(df["episode_index"].unique())
        if len(episode_idx_set) != 1:
            tqdm.write(f"[ERROR] {basename} has multiple episode_index values: {episode_idx_set}")
            error_count += 1
            continue
        episode_idx = episode_idx_set.pop()

        # Get expected task from episodes_meta
        if episode_idx not in episode_map:
            tqdm.write(f"[ERROR] Episode {episode_idx} not found in episodes.jsonl")
            error_count += 1
            continue
        expected_task = episode_map[episode_idx]

        # Get expected task_index from tasks_meta
        if expected_task not in task_map:
            tqdm.write(f"[ERROR] Task '{expected_task}' not found in tasks.jsonl for episode {episode_idx}")
            error_count += 1
            continue
        expected_task_index = task_map[expected_task]

        # Check task_index column
        if "task_index" not in df.columns:
            tqdm.write(f"[MISSING] {basename} missing task_index column")
            if fix:
                df["task_index"] = expected_task_index
                df.to_parquet(fpath, index=False)
                tqdm.write(f"  Fixed: added task_index={expected_task_index}")
            else:
                error_count += 1
            continue

        # Verify all rows have same task_index
        parquet_task_idx_set = set(df["task_index"].unique())
        if len(parquet_task_idx_set) != 1:
            tqdm.write(f"[ERROR] {basename} has multiple task_index values: {parquet_task_idx_set}")
            error_count += 1
            continue
        parquet_task_index = parquet_task_idx_set.pop()

        # Verify consistency with meta
        if parquet_task_index != expected_task_index:
            tqdm.write(f"[ERROR] {basename} task_index mismatch: parquet={parquet_task_index}, expected={expected_task_index}")
            if fix:
                df["task_index"] = expected_task_index
                df.to_parquet(fpath, index=False)
                tqdm.write(f"  Fixed: updated task_index to {expected_task_index}")
            else:
                error_count += 1

    print("\n========== Summary ==========")
    print(f"Total parquet files checked: {len(parquet_files)}")
    print(f"Total errors found: {error_count}")
    print(f"Fix mode: {'ON' if fix else 'OFF'}")

    return error_count == 0

def main():
    parser = argparse.ArgumentParser(description="Check/fix task_index consistency in parquet episodes")
    parser.add_argument("--data-root", required=True, help="Root folder containing chunk-* subfolders")
    parser.add_argument("--meta-root", required=True, help="Folder containing meta/tasks.jsonl and meta/episodes.jsonl")
    parser.add_argument("--fix", action="store_true", help="Automatically fix missing or inconsistent task_index")
    args = parser.parse_args()

    success = check_task_index_consistency_func(args.data_root, args.meta_root, fix=args.fix)
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
