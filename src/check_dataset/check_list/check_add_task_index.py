#!/usr/bin/env python3
"""
check_add_task_index.py

This script checks parquet episode files in a directory and verifies whether
they contain a `task_index` column. It can also fix missing columns by inserting
a specified task_index value.

Two usage modes:
1. CLI usage (normal command-line execution):
        python check_add_task_index.py --input_dir </path/to/parquets> [--fix] [--value 0]
2. Python API usage via the function `check_add_task_index`:
        from check_add_task_index import check_add_task_index
        success = check_add_task_index_func("/path/to/parquets", fix=False, value=0)

The output printed during API usage is identical to CLI logging.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm


def check_add_task_index_func(input_dir: str, fix: bool = False, value: int = 0) -> bool:
    """
    External API function to check or fix task_index in all parquet files.

    Parameters
    ----------
    input_dir : str
        Directory containing episode_*.parquet files.
    fix : bool
        If True, insert missing task_index columns using given `value`.
    value : int
        Value to assign to task_index when fixing missing files.

    Returns
    -------
    bool
        True    |   all parquet files already have task_index OR were successfully fixed.
        False   |   at least one parquet file is missing task_index (and fix=False).
    """
    parquet_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ])

    print(f"Found {len(parquet_files)} parquet files in {input_dir}\n")

    missing_count = 0
    missing_files = []

    for fpath in tqdm(parquet_files, desc="Checking task_index"):
        df = pd.read_parquet(fpath)

        if "task_index" not in df.columns:
            missing_count += 1
            missing_files.append(fpath)
            tqdm.write(f"[Missing] {os.path.basename(fpath)} is missing task_index")

            if fix:
                df["task_index"] = value
                df.to_parquet(fpath, index=False)
                tqdm.write(f"    Fixed: wrote task_index={value}\n")

    # Summary block
    print("\n========== Summary ==========")
    print(f"Total parquet files: {len(parquet_files)}")
    print(f"Files missing task_index: {missing_count}")

    if missing_count > 0:
        print("\nMissing files:")
        for f in missing_files:
            print("  -", os.path.basename(f))


    if fix:
        print(f"\nFix mode was ON, Missing files were updated with task_index={value}")
    else:
        print("\nDry-run mode, No files were modified.")

    # Return boolean result
    return missing_count == 0


def main():
    """
    CLI entry point for command-line use.
    Mirrors behavior of API function but parses arguments directly.
    """

    parser = argparse.ArgumentParser(
        description="Check (and optionally add) task_index column to parquet episodes."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to directory containing episode_*.parquet files"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="If set, automatically add task_index to files missing it."
    )
    parser.add_argument(
        "--value",
        type=int,
        default=0,
        help="Value to assign to task_index when using --fix. Default = 0."
    )

    args = parser.parse_args()

    # Call the API function and ignore result for CLI use
    check_add_task_index_func(args.input_dir, fix=args.fix, value=args.value)


if __name__ == "__main__":
    main()
