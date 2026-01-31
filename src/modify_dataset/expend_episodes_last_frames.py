#!/usr/bin/env python3
"""
Pad each episode by duplicating its last frame N times, then recompute episodes_stats.jsonl (LeRobot v2.1 compatible).

Operations:
- Parquet: duplicate last row N times, update indices and timestamps
- Video: freeze last frame N frames using FFmpeg tpad
- Meta: update info.json and episodes.jsonl
- Stats: regenerate episodes_stats.jsonl

Arguments:
--src-root: Source dataset root
--dst-root: Output dataset root
--n: Number of frames to duplicate
--force-stats: Overwrite existing stats
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("Pad episode tail and recompute stats")
    parser.add_argument("--src-root", required=True, help="Source dataset root")
    parser.add_argument("--dst-root", required=True, help="Output dataset root")
    parser.add_argument("--n", type=int, required=True, help="Number of frames to duplicate")
    parser.add_argument("--force-stats", action="store_true", help="Overwrite existing stats")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    """Save a dictionary as a JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(items: list[dict], path: Path) -> None:
    """Save a list of dictionaries as a JSONL file."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def pad_parquet(parquet_path: Path, n: int, global_index_start: int) -> int:
    """Duplicate the last row of a Parquet episode N times and update indices and timestamps."""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    last = df.iloc[-1].copy()
    dt = df["timestamp"].diff().median() if len(df) > 1 else 0.0
    dt = dt if np.isfinite(dt) and dt > 0 else 0.0
    pad_rows = []
    for i in range(n):
        row = last.copy()
        row["timestamp"] = last["timestamp"] + (i + 1) * dt
        pad_rows.append(row)
    df = pd.concat([df, pd.DataFrame(pad_rows)], ignore_index=True)
    df["frame_index"] = np.arange(len(df), dtype=np.int64)
    df["index"] = np.arange(global_index_start, global_index_start + len(df), dtype=np.int64)
    pq.write_table(pa.Table.from_pandas(df), parquet_path)
    return len(df)


def pad_video_tpad(video_path: Path, n: int, fps: int = 30) -> None:
    """Pad the last frame of a video using FFmpeg tpad filter. Execution is silent."""
    final_video = video_path.with_suffix(".pad.mp4")
    duration_sec = n / fps
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"tpad=stop_mode=clone:stop_duration={duration_sec}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(final_video)
    ], check=True, **kwargs)
    final_video.replace(video_path)


def flatten_dict(d) -> np.ndarray:
    """Flatten a dictionary into a numeric numpy array."""
    if not isinstance(d, dict):
        return np.array(d)
    keys = sorted(d.keys())
    return np.array([d[k] for k in keys])


def safe_stack(series: list) -> np.ndarray:
    """Stack a series of arrays or dicts into a single numpy array."""
    arrs = []
    for item in series:
        if isinstance(item, dict):
            arrs.append(flatten_dict(item))
        else:
            arrs.append(np.array(item))
    return np.stack(arrs)


def compute_stats(arr: np.ndarray) -> dict:
    """Compute min, max, mean, std, count statistics for a numpy array."""
    if len(arr.shape) == 1:
        arr = arr[:, None]
    return {
        "min": np.nanmin(arr, axis=0).tolist(),
        "max": np.nanmax(arr, axis=0).tolist(),
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std": np.nanstd(arr, axis=0).tolist(),
        "count": [arr.shape[0]]
    }


def process_episode(parquet_path: Path) -> dict:
    """Compute stats for one episode parquet file."""
    df = pd.read_parquet(parquet_path)
    stats = {}
    valid_cols = [c for c in df.columns if not any(x in c.lower() for x in ["index", "timestamp", "frame", "episode"])]
    for col in valid_cols:
        try:
            arr = safe_stack(df[col])
            if not np.issubdtype(arr.dtype, np.number):
                continue
            stats[col] = compute_stats(arr)
        except Exception as e:
            print(f"[stats] Skipping {col}: {e}")
    return stats


def main() -> None:
    """Main entry point to pad episodes and recompute stats."""
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    n = args.n
    if n <= 0:
        raise ValueError("n must be positive")
    if dst_root.exists():
        raise RuntimeError("Output directory already exists")
    shutil.copytree(src_root, dst_root)

    info_path = dst_root / "meta" / "info.json"
    episodes_path = dst_root / "meta" / "episodes.jsonl"
    stats_path = dst_root / "meta" / "episodes_stats.jsonl"
    info = load_json(info_path)
    episodes = load_jsonl(episodes_path)
    episode_meta = {ep["episode_index"]: ep for ep in episodes}
    fps = info["fps"]
    data_root = dst_root / "data"
    video_root = dst_root / "videos"
    global_index = 0
    total_frames = 0

    # Pad episodes
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        chunk_id = chunk_dir.name.split("-")[-1]
        for parquet_path in tqdm(sorted(chunk_dir.glob("episode_*.parquet")), desc=f"Padding chunk-{chunk_id}", unit="episode"):
            ep_idx = int(parquet_path.stem.split("_")[-1])
            frames = pad_parquet(parquet_path, n, global_index)
            episode_meta[ep_idx]["length"] = frames
            global_index += frames
            total_frames += frames
            # Pad videos
            video_chunk = video_root / f"chunk-{chunk_id}"
            if video_chunk.exists():
                for cam in video_chunk.iterdir():
                    video_path = cam / f"episode_{ep_idx:06d}.mp4"
                    if video_path.exists():
                        pad_video_tpad(video_path, n, fps)
                    else:
                        print(f"Missing video: {video_path}")

    info["total_frames"] = total_frames
    save_json(info, info_path)
    save_jsonl(episodes, episodes_path)

    # Recompute stats
    if stats_path.exists() and not args.force_stats:
        print(f"[stats] {stats_path} exists, skipping (use --force-stats to overwrite)")
    else:
        print("[stats] Recomputing episodes_stats.jsonl")
        with open(stats_path, "w") as fout:
            for chunk_dir in sorted(data_root.glob("chunk-*")):
                for parquet_path in sorted(chunk_dir.glob("episode_*.parquet")):
                    ep_idx = int(parquet_path.stem.split("_")[-1])
                    stats = process_episode(parquet_path)
                    fout.write(json.dumps({"episode_index": ep_idx, "stats": stats}) + "\n")

    print("[Done] Tail padding + stats recomputation finished")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()
