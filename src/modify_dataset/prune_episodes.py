#!/usr/bin/env python3
"""
Remove specified episodes from a dataset.

This script:
- Deletes parquet and video files for given episode indices
- Updates meta/episodes.jsonl and meta/episodes_stats.jsonl with reindexed episodes
- Rewrites parquet files with updated episode_index and globally continuous frame indices
- Updates meta/info.json accordingly
- Backs up original data and video directories to avoid accidental overwrite

Usage:
    python remove_episodes.py --dataset-root <path> --remove-indices "[1,2,3]"
"""

import ast
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

def parse_args():
    def parse_int_list(value: str):
        try:
            parsed = ast.literal_eval(value)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"Invalid list format: {value}"
            ) from e
        if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
            raise argparse.ArgumentTypeError(
                "remove-indices must be a list of integers, e.g. [1, 2, 3]"
            )
        return parsed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--remove-indices",
        type=parse_int_list,
        required=True,
        help='Python-style list, e.g. "[4, 5, 14]"'
    )
    return parser.parse_args()

def load_jsonl(path: Path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def remove_episode_files(data_dir: Path, video_dir: Path, remove_set):
    for idx in remove_set:
        name = f"episode_{idx:06d}"
        parquet = data_dir / f"{name}.parquet"
        if parquet.exists():
            parquet.unlink()
        if video_dir.exists():
            for cam_dir in video_dir.iterdir():
                mp4 = cam_dir / f"{name}.mp4"
                if mp4.exists():
                    mp4.unlink()

def update_episode_metadata(meta_dir: Path, remove_set):
    def reindex_by_episode_index(records, remove_set):
        """
        Filter records by remove_set, sort by episode_index,
        and reindex episode_index to 0..N-1

        Returns:
            new_records: reindexed records
            old_to_new: mapping from old episode_index to new episode_index
        """
        records = [r for r in records if r["episode_index"] not in remove_set]
        records.sort(key=lambda x: x["episode_index"])
        old_to_new = {}
        for new_idx, r in enumerate(records):
            old_to_new[r["episode_index"]] = new_idx
            r["episode_index"] = new_idx
        return records, old_to_new

    episodes_jsonl = meta_dir / "episodes.jsonl"
    stats_jsonl = meta_dir / "episodes_stats.jsonl"
    episodes = load_jsonl(episodes_jsonl)
    stats = load_jsonl(stats_jsonl)
    episodes, old_to_new = reindex_by_episode_index(episodes, remove_set)
    stats = [s for s in stats if s["episode_index"] in old_to_new]
    for s in stats:
        s["episode_index"] = old_to_new[s["episode_index"]]
    write_jsonl(episodes_jsonl, episodes)
    write_jsonl(stats_jsonl, stats)
    return old_to_new, episodes

def backup_and_recreate_dir(dir_path: Path) -> Path:
    """
    Rename dir_path to dir_path_bak_YYYYMMDDHHMMSS
    and recreate an empty dir_path.

    Returns:
        backup_dir: Path to backup directory
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"{dir_path} does not exist")
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_dir = dir_path.parent / f"{dir_path.name}_bak_{ts}"
    dir_path.rename(backup_dir)
    dir_path.mkdir(parents=True, exist_ok=False)
    return backup_dir

def update_parquet_files(
    data_dir: Path,
    video_dir: Path,
    old_to_new: dict
):
    """
    Safely rewrite parquet + video files with:
    - reindexed episode_index
    - globally continuous index
    - directory-level backup to avoid name collisions
    """
    data_bak = backup_and_recreate_dir(data_dir)
    video_bak = backup_and_recreate_dir(video_dir)

    # Rewrite parquet files
    global_index = 0
    for old_idx, new_idx in sorted(old_to_new.items(), key=lambda x: x[1]):
        old_name = f"episode_{old_idx:06d}"
        new_name = f"episode_{new_idx:06d}"
        src_parquet = data_bak / f"{old_name}.parquet"
        dst_parquet = data_dir / f"{new_name}.parquet"
        if not src_parquet.exists():
            raise FileNotFoundError(f"Missing parquet file: {src_parquet}")
        df = pd.read_parquet(src_parquet)
        if "episode_index" not in df.columns:
            raise KeyError(f"'episode_index' not found in {src_parquet}")
        if "index" not in df.columns:
            raise KeyError(f"'index' not found in {src_parquet}")
        frame_count = len(df)
        df["episode_index"] = new_idx
        df["index"] = range(global_index, global_index + frame_count)
        global_index += frame_count
        df.to_parquet(dst_parquet, index=False)

    # Rewrite video files
    if video_bak is not None:
        for cam_dir in video_bak.iterdir():
            if not cam_dir.is_dir():
                continue
            dst_cam_dir = video_dir / cam_dir.name
            dst_cam_dir.mkdir(parents=True, exist_ok=True)
            for old_idx, new_idx in old_to_new.items():
                old_name = f"episode_{old_idx:06d}.mp4"
                new_name = f"episode_{new_idx:06d}.mp4"
                src_mp4 = cam_dir / old_name
                dst_mp4 = dst_cam_dir / new_name
                if src_mp4.exists():
                    shutil.copy2(src_mp4, dst_mp4)
    return global_index

def update_info_json(meta_dir: Path, episodes, total_frames):
    info_json = meta_dir / "info.json"
    with open(info_json, "r") as f:
        info = json.load(f)
    total_episodes = len(episodes)
    info["total_episodes"] = total_episodes
    info["total_videos"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"]["train"] = f"0:{total_episodes}"
    with open(info_json, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    root = Path(args.dataset_root)
    remove_set = set(args.remove_indices)

    data_dir = root / "data/chunk-000"
    video_dir = root / "videos/chunk-000"
    meta_dir = root / "meta"

    remove_episode_files(data_dir, video_dir, remove_set)
    old_to_new, episodes = update_episode_metadata(meta_dir, remove_set)
    total_frames = update_parquet_files(
        data_dir,
        video_dir,
        old_to_new
    )
    update_info_json(meta_dir, episodes, total_frames)

    print("Done.")
    print(f"Removed episodes: {sorted(remove_set)}")
    print(f"Remaining episodes: {len(episodes)}")
    print(f"Total frames: {total_frames}")

if __name__ == "__main__":
    main()
