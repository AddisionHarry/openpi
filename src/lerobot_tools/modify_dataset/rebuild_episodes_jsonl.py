#!/usr/bin/env python3
"""
rebuild_episodes_jsonl.py

Rebuild meta/episodes.jsonl from an existing dataset-root.

Expected dataset structure:

    dataset-root/
        data/
            chunk-*/episode_XXXXXX.parquet
        meta/
            tasks.jsonl

The script will:
    1. Recursively scan data/chunk-*/*.parquet
    2. Extract:
        - episode_index
        - row count (length)
        - task_index
    3. Map task_index -> task name using meta/tasks.jsonl
    4. Generate meta/episodes.jsonl

Supports episodes distributed across multiple chunks.

----------------------------------------------------------------------
Command-line usage:

    python rebuild_episodes_jsonl.py \
        --dataset-root /path/to/dataset-root \
        [--force]

Arguments:
    --dataset-root   Root directory of dataset.
                     Must contain:
                        data/chunk-*/*.parquet
                        meta/tasks.jsonl

    --force          Overwrite existing meta/episodes.jsonl if present.
----------------------------------------------------------------------
"""

import os
import re
import json
import argparse
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm


def extract_episode_index(filename: str):
    m = re.search(r"episode_(\d+)\.parquet$", filename)
    return int(m.group(1)) if m else None


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records):
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def rebuild_episodes_jsonl(dataset_root: str, force: bool = False) -> bool:
    dataset_root = Path(dataset_root)

    data_root = dataset_root / "data"
    meta_root = dataset_root / "meta"
    tasks_path = meta_root / "tasks.jsonl"
    output_path = meta_root / "episodes.jsonl"

    if not data_root.exists():
        raise RuntimeError(f"Missing data directory: {data_root}")

    if not tasks_path.exists():
        raise RuntimeError(f"Missing tasks.jsonl: {tasks_path}")

    if output_path.exists() and not force:
        raise RuntimeError(
            f"{output_path} already exists. Use --force to overwrite."
        )

    tasks_meta = load_jsonl(tasks_path)
    task_index_to_name = {
        t["task_index"]: t["task"]
        for t in tasks_meta
    }

    parquet_files = sorted(data_root.glob("chunk-*/*.parquet"))

    if not parquet_files:
        raise RuntimeError(
            f"No parquet files found under {data_root}/chunk-*"
        )

    print(f"Found {len(parquet_files)} parquet files.")

    episode_records = {}
    duplicates = []

    for fpath in tqdm(parquet_files, desc="Scanning episodes"):
        ep_idx = extract_episode_index(fpath.name)
        if ep_idx is None:
            continue

        if ep_idx in episode_records:
            duplicates.append((ep_idx, fpath))
            continue

        metadata = pq.read_metadata(fpath)
        length = metadata.num_rows

        df = pd.read_parquet(fpath, columns=["task_index"])
        task_indices = df["task_index"].unique()

        if len(task_indices) != 1:
            raise ValueError(
                f"Episode {ep_idx:06d} has multiple task_index values: {task_indices}"
            )

        task_index = int(task_indices[0])

        if task_index not in task_index_to_name:
            raise ValueError(
                f"task_index {task_index} not found in tasks.jsonl"
            )

        episode_records[ep_idx] = {
            "episode_index": ep_idx,
            "length": length,
            "tasks": task_index_to_name[task_index],
        }

    if duplicates:
        raise ValueError(
            f"Duplicate episode indices detected: {duplicates}"
        )

    episodes = [
        episode_records[k]
        for k in sorted(episode_records.keys())
    ]

    write_jsonl(output_path, episodes)

    print("\n=====================================")
    print(f"Rebuilt {len(episodes)} episodes.")
    print(f"Saved to: {output_path}")
    print("=====================================\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild meta/episodes.jsonl from dataset-root"
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Root directory of the dataset.\n"
            "Must contain:\n"
            "  data/chunk-*/*.parquet\n"
            "  meta/tasks.jsonl"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing meta/episodes.jsonl if it already exists.",
    )

    args = parser.parse_args()

    rebuild_episodes_jsonl(
        dataset_root=args.dataset_root,
        force=args.force,
    )
