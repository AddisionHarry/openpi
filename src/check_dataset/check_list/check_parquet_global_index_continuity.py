#!/usr/bin/env python3
"""
check_parquet_global_index_continuity.py

Check and optionally fix global `index` continuity across parquet files,
ordered by episode index.

Command-line usage:
    python check_parquet_global_index_continuity.py --dir </path/to/parquet> [--fix]

External interface usage:
    from check_parquet_global_index_continuity import check_parquet_global_index_continuity_func
    success = check_parquet_global_index_continuity_func("/path/to/parquet", fix=True)
"""

import os
import re
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def extract_episode_index(filename: str):
    """Extract integer episode index from episode_000123.parquet"""
    m = re.search(r"episode_(\d+)\.parquet$", filename)
    return int(m.group(1)) if m else None


def check_parquet_global_index_continuity_func(directory: str, fix: bool = False) -> bool:
    """
    Check global continuity of `index` across parquet files ordered by episode_index.

    If fix=True, rewrite parquet files in-place with continuous global index.

    Returns True if no errors found, False otherwise.
    """
    files = []
    for f in os.listdir(directory):
        if not f.endswith(".parquet"):
            continue
        ep = extract_episode_index(f)
        if ep is not None:
            files.append((ep, f))
    if not files:
        print(f"[Error]No valid episode parquet files found in {directory}.")
        return False
    files.sort(key=lambda x: x[0])
    mode = "FIX MODE (FILES WILL BE MODIFIED)" if fix else "DRY RUN (NO MODIFICATION)"
    print(f"Checking {len(files)} parquet files... [{mode}]\n")
    error_count = 0
    global_expected_index = None
    rewrite_tables = []
    for _, fname in tqdm(files, desc="Checking"):
        path = os.path.join(directory, fname)
        table = pq.read_table(path)
        if "index" not in table.column_names:
            tqdm.write(f"[ERROR] Missing `index` column in {fname}")
            error_count += 1
            continue
        index_col = table["index"].to_pylist()
        if not index_col:
            continue
        # check intra-file continuity
        for i in range(1, len(index_col)):
            if index_col[i] != index_col[i - 1] + 1:
                tqdm.write(
                    f"[ERROR] Non-continuous index inside {fname}: "
                    f"{index_col[i - 1]} -> {index_col[i]}"
                )
                error_count += 1
                break
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
            new_indices = list(
                range(global_expected_index, global_expected_index + len(index_col))
            )
            new_col = pa.array(new_indices)
            table = table.set_column(
                table.column_names.index("index"),
                "index",
                new_col
            )
            rewrite_tables.append((path, table))
        global_expected_index = last_idx + 1 if not fix else global_expected_index + len(index_col)

    # Apply fixes
    if fix:
        tqdm.write("\nApplying fixes...")
        for path, table in rewrite_tables:
            tmp_path = path + ".tmp"
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, path)
        tqdm.write("All parquet files rewritten with continuous global index.")

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
