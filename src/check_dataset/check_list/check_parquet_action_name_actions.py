#!/usr/bin/env python3
"""
check_parquet_action_name_actions.py

Script to verify and optionally rename 'action' column to 'actions' in parquet files.

Command-line usage:
    python check_parquet_action_name_actions.py --input_dir </path/to/parquet> [--fix]

External interface usage:
    from check_parquet_action_name_actions import check_parquet_action_name_actions
    success = check_parquet_action_name_actions_func("/path/to/parquet", fix=True)
"""

import os
import pandas as pd
from tqdm import tqdm


def process_file(fpath, fix=False):
    """
    Check if a parquet file contains 'action' column.
    If fix=True, rename it to 'actions' in-place.
    Returns True if file needed modification, False otherwise.
    """
    df = pd.read_parquet(fpath)

    # 如果没有 action 列，静默通过
    if "action" not in df.columns:
        return False

    # 需要修改的情况才输出
    tqdm.write(f"[WARN]  {os.path.basename(fpath)} contains 'action', expected 'actions'")

    if fix:
        df = df.rename(columns={"action": "actions"})
        df.to_parquet(fpath, index=False)
        tqdm.write(f"[FIXED] {os.path.basename(fpath)} updated in-place")

    return True


def check_parquet_action_name_actions_func(input_dir: str, fix: bool = False) -> bool:
    """
    External interface for checking parquet action column names.
    Returns True if all files are correct, False if any file needs change.
    Printing behavior matches command-line output.
    """
    parquet_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ])

    tqdm.write(f"Found {len(parquet_files)} parquet files in {input_dir}")
    changed = 0

    for fpath in tqdm(parquet_files, desc="Checking parquet files"):
        changed += process_file(fpath, fix=fix)

    tqdm.write("\n===== SUMMARY =====")
    tqdm.write(f"Files needing change: {changed}")
    tqdm.write(f"Fix mode: {'ON (files were modified)' if fix else 'OFF (no changes written)'}")
    tqdm.write("===================\n")

    return changed == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check or fix parquet action column names")
    parser.add_argument("--input_dir", required=True, help="Directory containing parquet files")
    parser.add_argument("--fix", action="store_true", help="Apply in-place fix (rename 'action' → 'actions')")
    args = parser.parse_args()

    check_parquet_action_name_actions_func(args.input_dir, fix=args.fix)
