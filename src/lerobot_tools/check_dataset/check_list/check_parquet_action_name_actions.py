#!/usr/bin/env python3
"""
check_parquet_action_name_actions.py

Verify consistency of the action column naming in a LeRobot-style dataset.

This script recursively scans the provided input directory for
`episode_*.parquet` files (including nested chunk directories such as
`chunk-000`, `chunk-001`, etc.) and checks whether any file contains
a column named 'action' instead of the expected 'actions'.

It also validates the "features" section in the corresponding
info.json file to ensure the key name is 'actions'.

If --fix is enabled, the script performs in-place corrections:
    - Rename parquet column: 'action' -> 'actions'
    - Rename info.json feature key: 'action' -> 'actions'

Command-line usage:
    python check_parquet_action_name_actions.py \
        --input-dir </path/to/dataset/data> \
        --info-path </path/to/dataset/meta/info.json> \
        [--fix]

External interface usage:
    from check_parquet_action_name_actions import \
        check_parquet_action_name_actions_func

    success = check_parquet_action_name_actions_func(
        input_dir="/path/to/dataset/data",
        info_path="/path/to/dataset/meta/info.json",
        fix=True
    )

Return value:
    True  -> No inconsistency found (or successfully fixed)
    False -> One or more inconsistencies detected (check mode)
"""

import os
import json
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import re

def extract_episode_index(path: str) -> int:
    m = re.search(r"episode_(\d+)\.parquet", path)
    return int(m.group(1)) if m else -1

def collect_parquet_files(input_dir: str) -> list:
    """Recursively collect all episode_*.parquet files."""
    parquet_files = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.startswith("episode_") and f.endswith(".parquet"):
                parquet_files.append(os.path.join(root, f))

    parquet_files.sort(key=extract_episode_index)
    return parquet_files

def process_parquet_file(fpath, fix=False):
    """
    Check if a parquet file contains 'action' column.
    If fix=True, rename it to 'actions' in-place.
    Returns True if file needed modification, False otherwise.
    """
    schema = pq.read_schema(fpath)
    columns = schema.names

    if "action" not in columns:
        return False

    tqdm.write(f"[WARN]  {os.path.basename(fpath)} contains 'action', expected 'actions'")

    if fix:
        df = pd.read_parquet(fpath)
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

def check_parquet_action_name_actions_func(input_dir: str, info_path: str, fix: bool = False) -> bool:
    """
    External interface.
    Returns True if everything is correct, False if any issue is found.
    """
    parquet_files = collect_parquet_files(input_dir)
    if not parquet_files:
        tqdm.write("No episode parquet files found.")
        return False
    tqdm.write(f"Found {len(parquet_files)} episode parquet files (recursive)")

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
    parser.add_argument("--fix", action="store_true", help="Apply in-place fix (rename 'action' -> 'actions')")
    args = parser.parse_args()

    check_parquet_action_name_actions_func(args.input_dir, args.info_path, fix=args.fix)
