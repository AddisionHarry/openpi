#!/usr/bin/env python3
"""
Fast Parquet -> JSONL converter with multiprocessing + progress bar.

Features:
- Streaming read (no OOM)
- Multi-file parallel
- Real-time row progress bar
- Very fast (CPU bound)

Usage:
    python test_parquet_to_jsonl.py input.parquet output.jsonl
    python test_parquet_to_jsonl.py parquet_dir jsonl_dir
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


# -----------------------
# Worker
# -----------------------

def convert_one(args):
    parquet_path, jsonl_path, batch_size = args

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    with open(jsonl_path, "w", encoding="utf-8") as f:
        with tqdm(
            total=total_rows,
            desc=parquet_path.name,
            unit="rows",
            position=0,
            leave=True,
        ) as pbar:

            for batch in pf.iter_batches(batch_size=batch_size):
                table = batch.to_pydict()
                keys = list(table.keys())
                n = len(table[keys[0]])

                for i in range(n):
                    row = {k: table[k][i] for k in keys}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                pbar.update(n)

    return str(parquet_path)


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(mp.cpu_count() - 1, 1),
        help="Number of parallel processes",
    )

    args = parser.parse_args()

    # Single file
    if args.input.is_file():
        convert_one((args.input, args.output, args.batch_size))
        return

    # Directory mode
    args.output.mkdir(parents=True, exist_ok=True)

    tasks = []
    for p in sorted(args.input.glob("*.parquet")):
        out = args.output / (p.stem + ".jsonl")
        tasks.append((p, out, args.batch_size))

    print(f"[INFO] Found {len(tasks)} parquet files")
    print(f"[INFO] Using {args.workers} workers")

    with mp.Pool(args.workers) as pool:
        list(pool.imap_unordered(convert_one, tasks))

    print("[DONE]")


if __name__ == "__main__":
    main()
