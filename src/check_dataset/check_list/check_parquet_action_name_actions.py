#!/usr/bin/env python3
"""
check_parquet_action_name_actions.py

Script to verify and optionally rename 'action' column to 'actions' in parquet files.

Command-line usage:
    python check_parquet_action_name_actions.py --input-dir </path/to/parquet> --info-path </path/to/info.json> [--fix]

External interface usage:
    from check_parquet_action_name_actions import check_parquet_action_name_actions
    success = check_parquet_action_name_actions_func("/path/to/parquet", "/path/to/info.json", fix=True)
"""

import os
import json
import pandas as pd
from tqdm import tqdm


def process_parquet_file(fpath, fix=False):
    """
    Check if a parquet file contains 'action' column.
    If fix=True, rename it to 'actions' in-place.
    Returns True if file needed modification, False otherwise.
    """
    df = pd.read_parquet(fpath)

    if "action" not in df.columns:
        return False

    tqdm.write(f"[WARN]  {os.path.basename(fpath)} contains 'action', expected 'actions'")

    if fix:
        df = df.rename(columns={"action": "actions"})
        df.to_parquet(fpath, index=False)
        tqdm.write(f"[FIXED] {os.path.basename(fpath)} updated in-place")

    return True

def process_info_json(info_path: str, fix=False) -> bool:
    """
    Check if info.json features contain 'action' instead of 'actions'.
    If fix=True, rename feature key.
    Returns True if modification was needed.
    """
    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})
    changed = False

    if "action" in features:
        tqdm.write(
            "[ERROR] info.json features contains 'action', expected 'actions'"
        )
        changed = True

        if fix:
            if "actions" in features:
                tqdm.write(
                    "[WARN]  info.json already has 'actions'; "
                    "overwriting with content from 'action'"
                )

            features["actions"] = features.pop("action")
            tqdm.write("[FIXED] info.json feature 'action' -> 'actions'")

    if fix and changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    return changed

def check_parquet_action_name_actions_func(
    input_dir: str,
    info_path: str,
    fix: bool = False
) -> bool:
    """
    External interface.
    Returns True if everything is correct, False if any issue is found.
    """
    parquet_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    )

    tqdm.write(f"Found {len(parquet_files)} parquet files in {input_dir}")

    parquet_changed = 0
    for fpath in tqdm(parquet_files, desc="Checking parquet files"):
        parquet_changed += process_parquet_file(fpath, fix=fix)

    info_changed = process_info_json(info_path, fix=fix)

    tqdm.write("\n===== SUMMARY =====")
    tqdm.write(f"Parquet files needing change: {parquet_changed}")
    tqdm.write(f"info.json feature errors: {1 if info_changed else 0}")
    tqdm.write(
        f"Fix mode: {'ON (files were modified)' if fix else 'OFF (check only)'}"
    )
    tqdm.write("===================\n")

    return (parquet_changed == 0) and (not info_changed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check or fix parquet action column names")
    parser.add_argument("--input-dir", required=True, help="Directory containing parquet files")
    parser.add_argument("--info-path", required=True, help="Path to info.json file")
    parser.add_argument("--fix", action="store_true", help="Apply in-place fix (rename 'action' → 'actions')")
    args = parser.parse_args()

    check_parquet_action_name_actions_func(args.input_dir, args.info_path, fix=args.fix)
