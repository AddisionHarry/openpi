#!/usr/bin/env python3
"""
check_dataset_info_consistency.py

Check and optionally fix consistency between:
- meta/info.json
- data/chunk-000/*.parquet
- videos/chunk-000/*/*.mp4

Command-line usage:
    python check_dataset_info_consistency.py --dataset-root <path> [--fix]

External interface:
    from check_dataset_info_consistency import check_dataset_info_consistency
    success = check_dataset_info_consistency(
        dataset_root="/path/to/parquets",
        fix=False
    )
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
    data_dir = root / "data/chunk-000"
    video_root = root / "videos/chunk-000"

    info = load_json(info_path)
    features = info.get("features", {})

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))

    errors = 0

    # Get Episode / frame / video counts
    total_episodes_real = len(parquet_files)

    total_frames_real = 0
    for p in parquet_files:
        df = pd.read_parquet(p, columns=["frame_index"])
        total_frames_real += len(df)

    total_videos_real = total_episodes_real

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

    # Feature shape check (non-image)
    sample_df = pd.read_parquet(parquet_files[0])

    for feat_name, feat_info in features.items():
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

    # Image feature vs video shape
    for feat_name, feat_info in features.items():
        if feat_info.get("dtype") != "video":
            continue

        cam_dir = video_root / feat_name
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
