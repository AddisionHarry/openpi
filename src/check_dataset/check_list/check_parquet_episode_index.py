#!/usr/bin/env python3
"""
check_parquet_episode_index.py

Script to check and optionally fix `episode_index` in parquet files.

Command-line usage:
    python check_parquet_episode_index.py --dir </path/to/parquet> [--fix]

External interface usage:
    from check_parquet_episode_index import check_parquet_episode_index
    success = check_parquet_episode_index_func("/path/to/parquet", fix=True)
"""

import os
import re
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def extract_index_from_filename(filename: str):
    """Extract integer episode index from filename like episode_000063.parquet"""
    match = re.search(r"episode_(\d+)\.parquet$", filename)
    if match:
        return int(match.group(1))
    return None


def check_parquet_episode_index_func(directory: str, fix: bool = False) -> bool:
    """
    Check all parquet files in `directory` to ensure `episode_index` matches filename.
    If fix=True, correct mismatched `episode_index` in-place.

    Returns True if all files are correct, False if any mismatch found.
    Printing behavior matches command-line output.
    """
    files = sorted(f for f in os.listdir(directory) if f.endswith(".parquet"))
    if not files:
        print("No parquet files found.")
        return True

    mode = "DRY RUN (NO MODIFICATION)" if not fix else "FIX MODE (FILES WILL BE MODIFIED)"
    print(f"Checking {len(files)} parquet files... [{mode}]\n")

    error_count = 0

    for filename in tqdm(files, desc="Checking"):
        file_path = os.path.join(directory, filename)
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

            if not fix:
                tqdm.write("        (Dry-run: NOT modifying file)\n")
                continue

            tqdm.write("        Fixing...")

            # Fix column
            corrected_col = pa.array([expected_idx] * len(table))
            table = table.set_column(
                table.column_names.index("episode_index"),
                "episode_index",
                corrected_col
            )

            # Safe write
            tmp_path = file_path + ".tmp"
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, file_path)

            tqdm.write("        Fixed and overwritten.\n")

    print("\n================ SUMMARY ================")
    print(f"Total parquet files checked : {len(files)}")
    print(f"Total mismatched episodes   : {error_count}")
    print(f"Fix mode                    : {fix}")
    print("==========================================\n")

    return error_count == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check and optionally fix episode_index in parquet files.")
    parser.add_argument("--dir", required=True, help="Directory containing episode_XXXXX.parquet files")
    parser.add_argument("--fix", action="store_true", help="Apply fixes in-place. Default is dry-run.")
    args = parser.parse_args()

    check_parquet_episode_index_func(args.dir, fix=args.fix)
