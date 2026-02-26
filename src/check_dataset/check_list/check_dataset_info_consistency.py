#!/usr/bin/env python3
"""
check_dataset_info_consistency.py

Check and optionally fix consistency of a dataset by comparing:
- meta/info.json metadata
- all parquet files under data/chunk-*/episode_*.parquet
- all videos under videos/chunk-*/<camera>/episode_*.mp4 or .avi

Features verified include:
- total episodes, frames, videos
- non-video feature shapes
- video feature resolutions

Command-line usage:
    python check_dataset_info_consistency.py --dataset-root <path> [--fix]

External interface usage:
    from check_dataset_info_consistency import check_dataset_info_consistency_func
    success = check_dataset_info_consistency_func(
        dataset_root="/path/to/dataset",
        fix=False
    )

Notes:
- Supports multiple data/video chunks (chunk-000, chunk-001, etc.)
- In fix mode (--fix), updates info.json if mismatches are found
"""


import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_video_shape(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read video frame: {video_path}")
    h, w, c = frame.shape
    return [h, w, c]


def check_dataset_info_consistency_func(dataset_root: str, fix: bool = False) -> bool:
    root = Path(dataset_root)

    info_path = root / "meta/info.json"

    data_chunks = sorted((root / "data").glob("chunk-*"))
    video_chunks = sorted((root / "videos").glob("chunk-*"))

    info = load_json(info_path)
    features = info.get("features", {})

    total_episodes_real = 0
    total_frames_real = 0
    parquet_files_all = []

    for chunk_dir in data_chunks:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        total_episodes_real += len(parquet_files)
        parquet_files_all.extend(parquet_files)
        for p in parquet_files:
            df = pd.read_parquet(p, columns=["frame_index"])
            total_frames_real += len(df)

    errors = 0

    total_videos_real = 0
    for chunk_dir in video_chunks:
        for feat_name, feat_info in features.items():
            if feat_info.get("dtype") != "video":
                continue
            cam_dir = chunk_dir / feat_name
            if not cam_dir.exists():
                continue
            mp4s = sorted(list(cam_dir.glob("episode_*.mp4")) + list(cam_dir.glob("episode_*.avi")))
            total_videos_real += len(mp4s)


    def check_field(name: str, real: int, info, errors: int):
        recorded = info.get(name)
        if recorded != real:
            tqdm.write(
                f"[ERROR] info.json {name}={recorded}, actual={real}"
            )
            errors += 1
            if fix:
                info[name] = real
                tqdm.write(f"[FIXED] info.json {name} -> {real}")
        return errors

    errors = check_field("total_episodes", total_episodes_real, info, errors)
    errors = check_field("total_frames", total_frames_real, info, errors)
    errors = check_field("total_videos", total_videos_real, info, errors)

    # Feature shape check (non-image) with progress
    tqdm.write("Checking non-image features...")
    sample_df = pd.read_parquet(parquet_files_all[0])

    for feat_name, feat_info in tqdm(features.items(), desc="Checking features"):
        if feat_info.get("dtype") == "video":
            continue

        if feat_name not in sample_df.columns:
            tqdm.write(f"[ERROR] feature '{feat_name}' not found in parquet")
            errors += 1
            continue

        value = sample_df.iloc[0][feat_name]
        arr = np.asarray(value)
        if arr.ndim == 0:
            real_shape = [1]
        elif arr.ndim == 1:
            real_shape = [arr.shape[0]]
        else:
            real_shape = list(arr.shape)

        recorded_shape = feat_info.get("shape", [])

        if real_shape != recorded_shape:
            tqdm.write(
                f"[ERROR] feature '{feat_name}' shape mismatch: "
                f"info={recorded_shape}, parquet={real_shape}"
            )
            errors += 1
            if fix:
                feat_info["shape"] = real_shape
                tqdm.write(
                    f"[FIXED] feature '{feat_name}' shape -> {real_shape}"
                )

    # Image feature vs video shape with progress
    tqdm.write("Checking video features...")
    for feat_name, feat_info in tqdm(features.items(), desc="Checking video features"):
        if feat_info.get("dtype") != "video":
            continue

        for chunk_dir in video_chunks:
            cam_dir = chunk_dir / feat_name
            if not cam_dir.exists():
                tqdm.write(f"[ERROR] video dir missing: {cam_dir}")
                errors += 1
                continue

            mp4s = sorted(list(cam_dir.glob("episode_*.mp4")) + list(cam_dir.glob("episode_*.avi")))
            if not mp4s:
                tqdm.write(f"[ERROR] no videos in {cam_dir}")
                errors += 1
                continue

            real_shape = get_video_shape(mp4s[0])
            recorded_shape = feat_info["shape"]

            if real_shape[:-1] != recorded_shape[:-1]:
                tqdm.write(
                    f"[ERROR] image feature '{feat_name}' shape mismatch: "
                    f"info={recorded_shape}, video={real_shape}"
                )
                errors += 1
                if fix:
                    feat_info["shape"] = real_shape[:-1] + feat_info["shape"][-1]
                    tqdm.write(
                        f"[FIXED] image feature '{feat_name}' shape -> {real_shape}"
                    )

    # Save if fixed
    if fix and (errors > 0):
        tqdm.write("Saving updated info.json...")
        save_json(info_path, info)

    tqdm.write("\n===== SUMMARY =====")
    tqdm.write(f"Errors found: {errors}")
    tqdm.write(f"Fix mode: {'ON (info.json updated)' if fix else 'OFF (check only)'}")
    tqdm.write("===================\n")

    return errors == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and optionally fix dataset info.json consistency"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--fix", action="store_true")

    args = parser.parse_args()
    check_dataset_info_consistency_func(args.dataset_root, fix=args.fix)
