#!/usr/bin/env python3
"""
Update task prompt in dataset metadata.

This script:
- Updates a task prompt in meta/tasks.jsonl
- Finds affected episodes via parquet task_index
- Updates the corresponding `tasks` field in meta/episodes.jsonl

Parquet files are read-only; only metadata files are modified.

Usage:
    python update_task_prompt.py --dataset-root <path> --old-task "<old>" --new-task "<new>"
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True, help="dataset root path")
    parser.add_argument("--old-task", type=str, required=True, help="current task prompt")
    parser.add_argument("--new-task", type=str, required=True, help="new task prompt")
    return parser.parse_args()

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    args = parse_args()
    root = Path(args.dataset_root).resolve()

    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"

    # Find task_index for old_task and update to new_task
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = load_jsonl(tasks_path)

    target_task_index = None
    for t in tasks:
        if t["task"] == args.old_task:
            target_task_index = t["task_index"]
            t["task"] = args.new_task
            break

    if target_task_index is None:
        raise RuntimeError(
            f"Task '{args.old_task}' not found in meta/tasks.jsonl"
        )

    write_jsonl(tasks_path, tasks)

    # Find all episodes with the target_task_index
    affected_episodes = set()

    for parquet_file in sorted(data_dir.glob("episode_*.parquet")):
        table = pq.read_table(
            parquet_file,
            columns=["episode_index", "task_index"]
        )
        df = table.to_pandas()

        if (df["task_index"] == target_task_index).any():
            episode_idx = int(df["episode_index"].iloc[0])
            affected_episodes.add(episode_idx)

    if not affected_episodes:
        print(
            f"Warning: no episodes found for task_index={target_task_index}"
        )

    # Update episodes.jsonl tasks
    episodes_path = meta_dir / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)

    updated = 0
    for e in episodes:
        if e["episode_index"] in affected_episodes:
            if e["tasks"] == args.old_task:
                e["tasks"] = args.new_task
                updated += 1

    write_jsonl(episodes_path, episodes)

    print("Task prompt updated successfully")
    print(f"task_index      : {target_task_index}")
    print(f"episodes updated: {updated}")

if __name__ == "__main__":
    import os
    if os.environ.get("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for VS Code debugger to attach on port 5678...")
        debugpy.wait_for_client()
        print("Debugger attached, resuming execution...")
    main()
