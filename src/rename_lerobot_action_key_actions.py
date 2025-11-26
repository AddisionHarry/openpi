#!/usr/bin/env python3
import os
import pandas as pd
from tqdm import tqdm

def main(input_dir):
    parquet_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ])

    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    for fpath in tqdm(parquet_files, desc="Processing parquet files"):
        df = pd.read_parquet(fpath)
        if "action" in df.columns:
            df = df.rename(columns={"action": "actions"})
            df.to_parquet(fpath, index=False)

    print("Done. All parquet files updated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to directory containing parquet files")
    args = parser.parse_args()
    main(args.input_dir)
