#!/usr/bin/env python3
"""
check_parquet_episode_index.py

Check and optionally fix `episode_index` in parquet files.

This script recursively scans all parquet files under `data/chunk-*`
and verifies that the `episode_index` column matches the index in the filename.
If `--fix` is specified, mismatched values will be corrected in-place.

Command-line usage:
    python check_parquet_episode_index.py --dataset-root </path/to/dataset> [--fix]

External interface usage:
    from check_parquet_episode_index import check_parquet_episode_index_func
    success = check_parquet_episode_index_func("/path/to/dataset", fix=True)
"""


import os
import re
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


def extract_index_from_filename(filename: str):
    """Extract integer episode index from filename like episode_000063.parquet"""
    match = re.search(r"episode_(\d+)\.parquet$", filename)
    if match:
        return int(match.group(1))
    return None


def check_parquet_episode_index_func(dataset_root: str, fix: bool = False) -> bool:
    """
    Check all parquet files under `data/chunk-*` to ensure `episode_index` matches filename.
    Supports multiple chunks automatically.
    """
    root = Path(dataset_root) / "data"
    parquet_files = sorted(root.glob("chunk-*/episode_*.parquet"))

    if not parquet_files:
        print("No parquet files found.")
        return True

    mode = "DRY RUN" if not fix else "FIX MODE"
    print(f"Checking {len(parquet_files)} parquet files... [{mode}]\n")

    error_count = 0

    for file_path in tqdm(parquet_files, desc="Checking"):
        filename = file_path.name
        expected_idx = extract_index_from_filename(filename)
        if expected_idx is None:
            tqdm.write(f"[Skip] Unrecognized filename: {filename}")
            continue

        table = pq.read_table(file_path)
        if "episode_index" not in table.column_names:
            tqdm.write(f"[Error] Missing `episode_index` in {filename}")
            error_count += 1
            continue

        ep_idx = table["episode_index"][0].as_py()
        if ep_idx != expected_idx:
            error_count += 1
            tqdm.write(
                f"\n[ERROR] Mismatch in {filename}\n"
                f"        Filename index : {expected_idx}\n"
                f"        Parquet index  : {ep_idx}"
            )

            if fix:
                corrected_col = pa.array([expected_idx] * len(table))
                table = table.set_column(
                    table.column_names.index("episode_index"),
                    "episode_index",
                    corrected_col
                )
                tmp_path = file_path.with_suffix(".tmp.parquet")
                pq.write_table(table, tmp_path)
                os.replace(tmp_path, file_path)
                tqdm.write("        Fixed.\n")

    print("\n===== SUMMARY =====")
    print(f"Total parquet files checked : {len(parquet_files)}")
    print(f"Total mismatched episodes   : {error_count}")
    print(f"Fix mode                    : {fix}")
    print("========================\n")

    return error_count == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check and optionally fix episode_index in parquet files.")
    parser.add_argument("--dataset-root", required=True, help="Root directory of the dataset containing 'data/chunk-*' subdirectories")
    parser.add_argument("--fix", action="store_true", help="Apply fixes in-place. Default is dry-run.")
    args = parser.parse_args()

    check_parquet_episode_index_func(args.dataset_root, fix=args.fix)
