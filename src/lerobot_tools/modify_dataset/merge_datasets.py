#!/usr/bin/env python3
"""
Merge multiple datasets into one dataset.

Usage:
python merge_datasets.py \
  --dataset-roots /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
  --output-root /path/to/merged_dataset --chunk-size 500
"""

import os
import ast
import argparse
import json
import re
import shutil
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


INFO_IGNORE_KEYS = {
    "total_episodes",
    "total_frames",
    "total_videos",
    "total_chunks",
    "splits",
    "total_tasks"
}

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def extract_episode_index(filename: str):
    m = re.search(r"episode_(\d+)\.parquet$", filename)
    return int(m.group(1)) if m else None

def diff_dict(d1, d2, prefix=""):
    diffs = []
    keys = set(d1.keys()) | set(d2.keys())
    for k in keys:
        p = f"{prefix}.{k}" if prefix else k
        if k not in d1:
            diffs.append(f"{p} missing in dataset1")
            continue
        if k not in d2:
            diffs.append(f"{p} missing in dataset2")
            continue
        v1 = d1[k]
        v2 = d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            diffs.extend(diff_dict(v1, v2, p))
        else:
            if v1 != v2:
                diffs.append(f"{p} differs: {v1} != {v2}")
    return diffs

def check_info_compatibility(dataset_roots):
    infos = []
    for root in dataset_roots:
        info_path = root / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        for k in INFO_IGNORE_KEYS:
            info.pop(k, None)
        infos.append((root, info))
    base_root, base_info = infos[0]
    for root, info in infos[1:]:
        diffs = diff_dict(base_info, info)
        if diffs:
            msg = "\n".join(diffs[:20])
            raise RuntimeError(
                f"\ninfo.json mismatch between datasets:\n"
                f"{base_root}\n"
                f"{root}\n\n"
                f"Differences:\n{msg}\n"
            )
    print("info.json compatibility check passed")
    return base_info

def collect_episodes(dataset_roots):
    """
    collect (dataset_root, episode_index, parquet_path)
    """
    episodes = []

    for root in dataset_roots:
        data_root = root / "data"

        for chunk_dir in sorted(data_root.glob("chunk-*")):
            for pq in chunk_dir.glob("episode_*.parquet"):
                ep_idx = int(pq.stem.split("_")[1])
                episodes.append((root, ep_idx, pq))

    episodes.sort(key=lambda x: (str(x[0]), x[1]))
    return episodes


def collect_videos(dataset_roots: list[Path]):
    mapping = {}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    for root in dataset_roots:
        video_root = root / "videos"
        for chunk_dir in video_root.glob("chunk-*"):
            for cam_dir in chunk_dir.iterdir():
                if not cam_dir.is_dir():
                    continue
                for video_file in cam_dir.glob("episode_*"):
                    if video_file.suffix.lower() not in video_exts:
                        continue
                    ep_idx = int(video_file.stem.split("_")[1])
                    mapping.setdefault((root, ep_idx), []).append(video_file)
    return mapping


def merge_tasks(dataset_roots: Path):
    merged_tasks = []
    task_to_new_index = {}
    task_index_map = {}
    next_index = 0
    for root in dataset_roots:
        tasks_path = root / "meta" / "tasks.jsonl"
        tasks = load_jsonl(tasks_path)
        for t in tasks:
            task_str = t["task"]
            old_idx = t["task_index"]
            if task_str not in task_to_new_index:
                task_to_new_index[task_str] = next_index
                merged_tasks.append({
                    "task_index": next_index,
                    "task": task_str
                })
                next_index += 1
            new_idx = task_to_new_index[task_str]
            task_index_map[(root, old_idx)] = new_idx
    return merged_tasks, task_index_map


def merge_meta(dataset_roots, new_episode_map, episode_length_map):
    episodes_all = []
    stats_all = []

    for root in dataset_roots:
        meta = root / "meta"
        episodes = load_jsonl(meta / "episodes.jsonl")
        stats = load_jsonl(meta / "episodes_stats.jsonl")
        for e in episodes:
            old_idx = e["episode_index"]
            key = (root, old_idx)
            if key not in new_episode_map:
                continue
            e_new = dict(e)
            e_new["episode_index"] = new_episode_map[key]
            e_new["length"] = episode_length_map[key]
            episodes_all.append(e_new)
        for s in stats:
            old_idx = s["episode_index"]
            key = (root, old_idx)
            if key not in new_episode_map:
                continue
            s_new = dict(s)
            s_new["episode_index"] = new_episode_map[key]
            stats_all.append(s_new)

    episodes_all.sort(key=lambda x: x["episode_index"])
    stats_all.sort(key=lambda x: x["episode_index"])

    return episodes_all, stats_all


def write_final_info(meta_out, base_info, total_episodes, total_frames, total_videos, chunk_size, merged_tasks_num):
    info = dict(base_info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["total_chunks"] = (total_episodes + chunk_size - 1) // chunk_size
    info["chunks_size"] = chunk_size
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["total_tasks"] = merged_tasks_num
    with open(meta_out / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def rebuild_episodes_jsonl(dataset_root: str, force: bool = False):
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
        raise RuntimeError(f"{output_path} already exists. Use --force to overwrite.")

    tasks_meta = load_jsonl(tasks_path)
    task_index_to_name = {t["task_index"]: t["task"] for t in tasks_meta}

    parquet_files = sorted(data_root.glob("chunk-*/*.parquet"))

    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {data_root}/chunk-*")

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
            raise ValueError(f"Episode {ep_idx:06d} has multiple task_index values: {task_indices}")
        task_index = int(task_indices[0])
        if task_index not in task_index_to_name:
            raise ValueError(f"task_index {task_index} not found in tasks.jsonl")
        episode_records[ep_idx] = {
            "episode_index": ep_idx,
            "length": length,
            "tasks": task_index_to_name[task_index],
        }
    if duplicates:
        raise ValueError(f"Duplicate episode indices detected: {duplicates}")
    episodes = [episode_records[k] for k in sorted(episode_records.keys())]
    write_jsonl(output_path, episodes)


def main(args):
    dataset_roots = [Path(p) for p in args.dataset_roots]
    out_root = Path(args.output_root)

    data_out = out_root / "data"
    video_out = out_root / "videos"
    meta_out = out_root / "meta"

    data_out.mkdir(parents=True, exist_ok=True)
    video_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    print("Collecting episodes...")
    episodes = collect_episodes(dataset_roots)
    video_map = collect_videos(dataset_roots)
    total_videos = sum(len(v) for v in video_map.values())

    chunk_size = args.chunk_size
    print("Using chunk size:", chunk_size)

    new_episode_map = {}
    global_frame = 0

    print("Checking info.json compatibility...")
    base_info = check_info_compatibility(dataset_roots)

    print("Merging tasks...")
    merged_tasks, task_index_map = merge_tasks(dataset_roots)
    write_jsonl(meta_out / "tasks.jsonl", merged_tasks)
    print("Unique tasks after merge:", len(merged_tasks))

    print("Rebuilding dataset...")
    new_idx = 0
    new_chunk = 0
    episode_length_map = {}
    pbar = tqdm(total=len(episodes), desc="Merging episodes")
    for i in range(0, len(episodes), chunk_size):
        chunk_eps = episodes[i:i+chunk_size]
        dst_chunk = data_out / f"chunk-{new_chunk:03d}"
        dst_chunk.mkdir(parents=True, exist_ok=True)
        for root, old_ep, pq in chunk_eps:
            new_episode_map[(root, old_ep)] = new_idx
            df = pd.read_parquet(pq, engine="pyarrow")
            episode_len = len(df)
            df["episode_index"] = new_idx
            df["index"] = range(global_frame, global_frame + len(df))
            df["task_index"] = df["task_index"].map(lambda x: task_index_map.get((root, int(x))))
            episode_length_map[(root, old_ep)] = episode_len
            if df["task_index"].isnull().any():
                raise RuntimeError(f"Unknown task_index in {pq}")
            df["task_index"] = df["task_index"].astype("int32")
            global_frame += len(df)
            dst_parquet = dst_chunk / f"episode_{new_idx:06d}.parquet"
            df.to_parquet(dst_parquet, index=False, engine="pyarrow")
            for src_video in video_map.get((root, old_ep), []):
                dst_video = video_out / dst_chunk.name / src_video.parent.name / f"episode_{new_idx:06d}{src_video.suffix}"
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src_video, dst_video)
                except OSError:
                    shutil.copy2(src_video, dst_video)
            new_idx += 1
            pbar.update(1)
        new_chunk += 1
    pbar.close()

    print("Merging meta files...")
    episodes_meta, stats_meta = merge_meta(dataset_roots, new_episode_map, episode_length_map)
    write_jsonl(meta_out / "episodes.jsonl", episodes_meta)
    write_jsonl(meta_out / "episodes_stats.jsonl", stats_meta)
    write_final_info(meta_out, base_info, len(episodes_meta), global_frame, total_videos, chunk_size, len(merged_tasks))

    print("Rebuilding episodes.jsonl...")
    rebuild_episodes_jsonl(str(out_root), True)

    print("\nDone")
    print("Total episodes:", len(episodes_meta))
    print("Total frames:", global_frame)


if __name__ == "__main__":
    def parse_list(value: str):
        try:
            parsed = ast.literal_eval(value)
        except Exception as e:
            raise argparse.ArgumentTypeError("Invalid list") from e
        if not isinstance(parsed, list):
            raise argparse.ArgumentTypeError("dataset-roots must be list")
        return parsed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-roots", nargs="+", required=True, help="list of dataset roots")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=1000, help="force chunk size (default: auto detect)")
    args = parser.parse_args()

    main(args)
