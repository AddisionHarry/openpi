#!/usr/bin/env python3
"""
Downsample LeRobot dataset with optional video and parquet frame skipping.

This script takes a LeRobot dataset, applies frame downsampling with a given
factor k, updates parquet files and video files accordingly, adjusts timestamps
to start from 0 for each episode, and updates info.json accordingly.
"""

import argparse
import json
import random
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Downsample LeRobot dataset")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root path of input dataset")
    parser.add_argument("--k", type=int, required=True, help="Downsampling factor (positive integer)")
    parser.add_argument("--output-root", type=str, required=True, help="Root path of output dataset")
    return parser.parse_args()


def load_info(info_path: Path) -> dict:
    """Load dataset info.json file."""
    with open(info_path, "r") as f:
        return json.load(f)


def save_info(info: dict, path: Path) -> None:
    """Save dataset info.json file."""
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def downsample_parquet(src_path: Path, dst_path: Path, n: int, k: int, global_index_start: int) -> int:
    """
    Downsample parquet file by keeping every k-th frame starting from n.

    Adjust frame_index, global index, and timestamps to start from 0.

    Args:
        src_path: Path to source parquet file
        dst_path: Path to destination parquet file
        n: Starting offset (0 <= n < k)
        k: Downsampling factor
        global_index_start: Starting global index for this episode

    Returns:
        Number of frames after downsampling
    """
    table = pq.read_table(src_path)
    df = table.to_pandas()

    keep_idx = np.arange(n, len(df), k)
    df = df.iloc[keep_idx].reset_index(drop=True)

    # Re-index frame_index and global index
    df["frame_index"] = np.arange(len(df), dtype=np.int64)
    df["index"] = np.arange(global_index_start, global_index_start + len(df), dtype=np.int64)

    # Adjust timestamps so first frame starts at 0
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]

    table = pa.Table.from_pandas(df)
    pq.write_table(table, dst_path)

    return len(df)


def downsample_video_ffmpeg(src_video: Path, dst_video: Path, n: int, k: int, old_fps: int, new_fps: int) -> None:
    """
    Downsample video using FFmpeg (silent mode).

    Args:
        src_video: Path to source video file
        dst_video: Path to destination video file
        n: Starting offset (0 <= n < k)
        k: Downsampling factor
        old_fps: Original FPS of video
        new_fps: New FPS after downsampling
    """
    filter_expr = f"select='not(mod(n-{n},{k}))',setpts=N/({new_fps}*TB)"
    cmd = [
        "ffmpeg",
        "-y",                   # Overwrite output file
        "-hide_banner",         # Hide FFmpeg banner
        "-loglevel", "error",   # Only show errors
        "-i", str(src_video),
        "-vf", filter_expr,
        "-r", str(new_fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(dst_video)
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    """Main function to downsample dataset parquet files and videos."""
    args = parse_args()
    root: Path = Path(args.dataset_root)
    out_root: Path = Path(args.output_root)
    k: int = args.k

    if k <= 0:
        raise ValueError("Downsampling factor k must be a positive integer")

    if out_root.exists():
        raise RuntimeError("Output directory already exists")

    shutil.copytree(root, out_root)

    info_path: Path = out_root / "meta" / "info.json"
    info: dict = load_info(info_path)

    old_fps: int = info["fps"]
    new_fps: int = old_fps // k
    info["fps"] = new_fps

    # Update video FPS for all features containing videos
    for feat in info["features"].values():
        if isinstance(feat, dict) and "info" in feat and "video.fps" in feat["info"]:
            feat["info"]["video.fps"] = new_fps

    global_index: int = 0
    total_frames: int = 0
    log_lines: list[str] = []

    data_root: Path = out_root / "data"
    video_root: Path = out_root / "videos"

    for chunk_dir in sorted(data_root.glob("chunk-*")):
        chunk_id: str = chunk_dir.name.split("-")[-1]
        parquet_paths = sorted(chunk_dir.glob("episode_*.parquet"))

        for parquet_path in tqdm(parquet_paths, desc=f"Chunk {chunk_id} episodes", unit="episode"):
            episode_id: str = parquet_path.stem
            n: int = random.randint(0, k - 1)
            log_lines.append(f"{episode_id}: n={n}")

            # ---- Parquet ----
            dst_parquet: Path = parquet_path
            frames: int = downsample_parquet(parquet_path, dst_parquet, n, k, global_index)
            global_index += frames
            total_frames += frames

            # ---- Videos ----
            video_chunk: Path = video_root / f"chunk-{chunk_id}"
            for video_key in tqdm(list(video_chunk.iterdir()), desc=f"{episode_id} videos", leave=False, unit="video"):
                src_video: Path = video_key / f"{episode_id}.mp4"
                if not src_video.exists():
                    continue
                tmp_video: Path = src_video.with_suffix(".tmp.mp4")
                downsample_video_ffmpeg(src_video, tmp_video, n, k, old_fps, new_fps)
                tmp_video.replace(src_video)

    # Update total frames in info.json
    info["total_frames"] = total_frames
    save_info(info, info_path)

    # Save downsample start offsets
    with open(out_root / "downsample_start_idx.log", "w") as f:
        f.write("\n".join(log_lines))

    print("[Done] Downsampling finished")
    print(f"Total frames after downsampling: {total_frames}")


if __name__ == "__main__":
    main()
