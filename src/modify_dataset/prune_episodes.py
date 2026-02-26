#!/usr/bin/env python3
"""
Remove specified episodes from a dataset and re-pack chunks.

This script:
- Deletes parquet and video files for given episode indices
- Updates meta/episodes.jsonl and meta/episodes_stats.jsonl with reindexed episodes
- Rewrites parquet files with updated episode_index and globally continuous frame indices
- Re-packs episodes into chunks to preserve original chunk size
- Updates meta/info.json accordingly
- Backs up original data and video directories

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
            raise argparse.ArgumentTypeError(f"Invalid list format: {value}") from e
        if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
            raise argparse.ArgumentTypeError("remove-indices must be a list of integers, e.g. [1, 2, 3]")
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


def remove_episode_files(data_root: Path, videos_root: Path, remove_set: set):
    print("Step 2/5: Removing episode files (parquet + mp4)...")
    data_chunks = sorted(data_root.glob("chunk-*"))
    video_chunks = sorted(videos_root.glob("chunk-*"))

    for chunk_dir in data_chunks:
        for idx in remove_set:
            parquet = chunk_dir / f"episode_{idx:06d}.parquet"
            if parquet.exists():
                parquet.unlink()

    for chunk_dir in video_chunks:
        for cam_dir in chunk_dir.iterdir():
            if not cam_dir.is_dir():
                continue
            for idx in remove_set:
                mp4 = cam_dir / f"episode_{idx:06d}.mp4"
                if mp4.exists():
                    mp4.unlink()

    print("Step 2/5: Episode files removed successfully\n")


def update_episode_metadata(meta_dir: Path, remove_set):
    print("Step 3/5: Updating episode metadata...")

    def reindex_by_episode_index(records, remove_set):
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

    print(f"Step 3/5: Metadata updated (remaining episodes: {len(episodes)})\n")
    return old_to_new, episodes


def backup_and_recreate_dir(dir_path: Path) -> Path:
    if not dir_path.exists():
        raise FileNotFoundError(f"{dir_path} does not exist")
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_dir = dir_path.parent / f"{dir_path.name}_bak_{ts}"
    print(f"Backing up {dir_path} to {backup_dir}...")
    dir_path.rename(backup_dir)
    dir_path.mkdir(parents=True, exist_ok=False)
    return backup_dir


def build_episode_list(data_root: Path):
    """
    Return sorted list of (episode_index, parquet_path) for all remaining episodes
    """
    episode_list = []
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for parquet in chunk_dir.glob("episode_*.parquet"):
            ep_idx = int(parquet.stem.split("_")[1])
            episode_list.append((ep_idx, parquet))
    episode_list.sort(key=lambda x: x[0])
    return episode_list


def build_video_map(videos_root: Path):
    """
    Build mapping: episode_index -> list of video files across all chunks/cameras
    """
    mapping = {}
    for chunk_dir in videos_root.glob("chunk-*"):
        for cam_dir in chunk_dir.iterdir():
            if not cam_dir.is_dir():
                continue
            for video_file in cam_dir.glob("episode_*.mp4"):
                ep_idx = int(video_file.stem.split("_")[1])
                mapping.setdefault(ep_idx, []).append(video_file)
    return mapping


def update_files(data_dir: Path, video_dir: Path, data_bak: Path, video_bak: Path, old_to_new: dict):
    # collect remaining episodes from backup
    episode_list = []
    for chunk_dir in sorted(data_bak.glob("chunk-*")):
        for pq_file in chunk_dir.glob("episode_*.parquet"):
            ep_idx = int(pq_file.stem.split("_")[1])
            if ep_idx in old_to_new:
                episode_list.append((ep_idx, pq_file))
    episode_list.sort(key=lambda x: x[0])

    video_map = build_video_map(video_bak)

    # determine original chunk size
    original_chunk_dirs = sorted(data_bak.glob("chunk-*"))
    original_chunk_size = max(len(list(d.glob("episode_*.parquet"))) for d in original_chunk_dirs)

    print("Step 4/5: Rewriting parquet and video files into new chunks...")
    global_index = 0
    new_chunk_idx = 0
    for i in range(0, len(episode_list), original_chunk_size):
        chunk_episodes = episode_list[i:i + original_chunk_size]
        dst_chunk_dir = data_dir / f"chunk-{new_chunk_idx:03d}"
        dst_chunk_dir.mkdir(parents=True, exist_ok=True)

        for old_idx, src_parquet in chunk_episodes:
            new_idx = old_to_new[old_idx]
            df = pd.read_parquet(src_parquet)
            df["episode_index"] = new_idx
            df["index"] = range(global_index, global_index + len(df))
            global_index += len(df)
            dst_parquet = dst_chunk_dir / f"episode_{new_idx:06d}.parquet"
            df.to_parquet(dst_parquet, index=False)

            # copy videos
            for src_mp4 in video_map.get(old_idx, []):
                dst_mp4 = video_dir / dst_chunk_dir.name / src_mp4.parent.name / f"episode_{new_idx:06d}.mp4"
                dst_mp4.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_mp4, dst_mp4)

        new_chunk_idx += 1

    return global_index



def update_info_json(meta_dir: Path, episodes, total_frames):
    print("Step 5/5: Updating meta/info.json...")
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
    print("Step 5/5: info.json updated successfully\n")


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    remove_set = set(args.remove_indices)

    data_root = root / "data"
    video_root = root / "videos"
    meta_dir = root / "meta"

    print("Step 1/5: Backing up original data and video directories...")
    data_bak = backup_and_recreate_dir(data_root)
    video_bak = backup_and_recreate_dir(video_root)
    print("Step 1/5: Directories backed up successfully\n")

    remove_episode_files(data_root, video_root, remove_set)
    old_to_new, episodes = update_episode_metadata(meta_dir, remove_set)
    total_frames = update_files(data_root, video_root,
    data_bak, video_bak, old_to_new)
    update_info_json(meta_dir, episodes, total_frames)

    print("Done.")
    print(f"Removed episodes: {sorted(remove_set)}")
    print(f"Remaining episodes: {len(episodes)}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()
