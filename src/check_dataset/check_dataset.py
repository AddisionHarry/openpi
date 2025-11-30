#!/usr/bin/env python3
"""
Top-level checker for LeRobot dataset.

This script runs all six check scripts sequentially on a given dataset directory,
maintaining each script's tqdm progress bar and printing outputs.
After all checks are done, it prints a summary of which checks passed or failed.
"""

import os
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(__file__))
import check_list

# Each entry: (name, function, args as tuple)
CHECKS = [
    (
        "Check Task Index",
        check_list.check_add_task_index_func,
        lambda dataset_dir, fix: (dataset_dir, fix, 0)
    ),
    (
        "Compute Episode Stats",
        check_list.check_compute_episodes_stats_func,
        lambda dataset_dir, fix: (str(os.path.join(dataset_dir, "data", "chunk-000")), str(os.path.join(dataset_dir, "meta")), fix)
    ),
    (
        "Delete Depth Info",
        check_list.check_delete_depth_info_func,
        lambda dataset_dir, fix: (os.path.join(dataset_dir, "meta", "info.json"), fix)
    ),
    (
        "Check Parquet Action Name",
        check_list.check_parquet_action_name_actions_func,
        lambda dataset_dir, fix: (dataset_dir,)
    ),
    (
        "Check Parquet Episode Index",
        check_list.check_parquet_episode_index_func,
        lambda dataset_dir, fix: (str(os.path.join(dataset_dir, "data", "chunk-000")), fix)
    ),
    (
        "Check Video Frames",
        check_list.check_lerobot_video_frames_func,
        lambda dataset_dir, fix: (dataset_dir, fix)
    ),
]

def main(dataset_dir: str, fix: bool = False):
    if not os.path.exists(dataset_dir):
        print(f"Error: dataset directory does not exist: {dataset_dir}")
        return

    results = {}

    for i, (check_name, check_func, arg_func) in enumerate(tqdm(CHECKS, desc="Running dataset checks")):
        tqdm.write(f"\n[{i+1}/{len(CHECKS)}] ================== Running {check_name} ==================\n")
        try:
            args = arg_func(dataset_dir, fix)
            passed = check_func(*args)
            results[check_name] = passed
        except Exception as e:
            tqdm.write(f"[ERROR] {check_name} failed: {e}")
            results[check_name] = False

    tqdm.write("\n========== CHECK SUMMARY ==========")
    for name, passed in results.items():
        status = "PASSED " if passed else "FAILED "
        tqdm.write(f"{name:30s} : {status}")
    tqdm.write("==================================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Top-level dataset checker for LeRobot")
    parser.add_argument("--dataset-dir", required=True, help="Root path of dataset to check")
    parser.add_argument("--fix", action="store_true", help="Apply fixes in-place. Default is dry-run.")
    args = parser.parse_args()

    main(args.dataset_dir, args.fix)
