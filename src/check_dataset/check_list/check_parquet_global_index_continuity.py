#!/usr/bin/env python3
"""
check_parquet_global_index_continuity.py

Verify and optionally repair global `index` continuity across
episode parquet files in a dataset directory.

The script recursively scans the provided root directory for
files matching the pattern `episode_*.parquet` (including nested
subdirectories such as chunk folders), sorts them by episode index,
and performs the following checks:

1. Intra-file continuity:
   Ensures that the `index` column within each parquet file increases
   strictly by 1 without gaps.

2. Inter-file continuity:
   Ensures that the first index of each episode file matches the
   expected global index based on the previous file, and that the
   overall dataset starts from index 0.

If --fix is enabled, the script rewrites each parquet file in-place,
reconstructing a continuous global `index` column while preserving
the original column type.

Command-line usage:
    python check_parquet_global_index_continuity.py \
        --dir </path/to/dataset/root> \
        [--fix]

External interface usage:
    from check_parquet_global_index_continuity import \
        check_parquet_global_index_continuity_func

    success = check_parquet_global_index_continuity_func(
        directory="/path/to/dataset/root",
        fix=True
    )

Return value:
    True  -> No continuity errors detected (or successfully fixed)
    False -> One or more continuity errors detected (check mode)
"""


import os
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def extract_episode_index(filename: str):
    """Extract integer episode index from episode_000123.parquet"""
    m = re.search(r"episode_(\d+)\.parquet$", filename)
    return int(m.group(1)) if m else None

def collect_parquet_files(root_dir: str):
    """
    Recursively collect (episode_index, absolute_path)
    for all episode_*.parquet files.
    """
    files = []

    for root, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.startswith("episode_") or not fname.endswith(".parquet"):
                continue

            ep = extract_episode_index(fname)
            if ep is not None:
                full_path = os.path.join(root, fname)
                files.append((ep, full_path))

    files.sort(key=lambda x: x[0])
    return files

def check_parquet_global_index_continuity_func(directory: str, fix: bool = False) -> bool:
    """
    Recursively check global continuity of `index` across
    episode_*.parquet files ordered by episode index.

    If fix=True, rewrite parquet files in-place with continuous global index.
    """
    files = collect_parquet_files(directory)
    if not files:
        print(f"[Error]No valid episode parquet files found in {directory}.")
        return False
    mode = "FIX MODE (FILES WILL BE MODIFIED)" if fix else "DRY RUN (NO MODIFICATION)"
    print(f"Checking {len(files)} parquet files... [{mode}]\n")
    error_count = 0
    global_expected_index = None
    rewrite_tables = []
    # for _, fname in tqdm(files, desc="Checking"):
    for _, path in tqdm(files, desc="Checking"):
        fname = os.path.basename(path)
        table = pq.read_table(path)
        if "index" not in table.column_names:
            tqdm.write(f"[ERROR] Missing `index` column in {fname}")
            error_count += 1
            continue
        index_col = table["index"].to_numpy()
        if len(index_col) == 0:
            continue
        # check intra-file continuity
        if not np.all(index_col[1:] == index_col[:-1] + 1):
            tqdm.write(f"[ERROR] Non-continuous index inside {fname}")
            error_count += 1
            continue
        # check inter-file continuity
        first_idx = index_col[0]
        last_idx = index_col[-1]
        if global_expected_index is None:
            # check global start
            if first_idx != 0:
                tqdm.write(
                    f"[ERROR] Global index does not start from 0 in {fname}\n"
                    f"        Actual start index: {first_idx}\n"
                    f"        Expected          : 0"
                )
                error_count += 1
            # Regardless of fix or dry-run, define expected start
            global_expected_index = 0 if fix else first_idx
        else:
            if first_idx != global_expected_index:
                tqdm.write(
                    f"[ERROR] Global index break at {fname}\n"
                    f"        Expected start index: {global_expected_index}\n"
                    f"        Actual start index  : {first_idx}"
                )
                error_count += 1
        if fix:
            new_indices = np.arange(global_expected_index, global_expected_index + len(index_col))
            new_col = pa.array(new_indices)
            table = table.set_column(table.column_names.index("index"), "index",new_col)
            tmp_path = path + ".tmp"
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, path)
            global_expected_index += len(index_col)
        else:
            global_expected_index = last_idx + 1

    print("\n================ SUMMARY ================")
    print(f"Total parquet files checked : {len(files)}")
    print(f"Index continuity errors     : {error_count}")
    print(f"Fix mode                    : {fix}")
    print("========================================\n")

    return error_count == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and optionally fix global index continuity in parquet files."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing episode_XXXXX.parquet files"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply in-place fix (rewrite index column)"
    )

    args = parser.parse_args()
    check_parquet_global_index_continuity_func(args.dir, fix=args.fix)
