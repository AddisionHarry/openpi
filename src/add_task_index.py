#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Add task_index column to parquet episodes")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to directory containing episode_*.parquet files"
    )
    args = parser.parse_args()
    input_dir = args.input_dir

    parquet_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ])

    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    for fpath in tqdm(parquet_files, desc="Adding task_index"):
        df = pd.read_parquet(fpath)

        if "task_index" not in df.columns:
            df["task_index"] = 0

        df.to_parquet(fpath, index=False)

    print("All episodes updated with task_index.")

if __name__ == "__main__":
    main()
