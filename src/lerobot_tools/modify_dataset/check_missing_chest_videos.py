#!/usr/bin/env python3

"""
check_missing_episode_videos.py

Check for missing episode video files by comparing:
- meta/episodes.jsonl
- a specified video directory (episode_XXXXXX.mp4)

Each episode in episodes.jsonl is expected to have a corresponding video file. Missing videos are reported and written to a log file.

This is a read-only diagnostic script; no dataset files are modified.

Usage:
    python check_missing_episode_videos.py --dataset-root <path> \
        [--video-subdir videos/chunk-000/observation.images.chest_rgb] \
        [--log-path manually_clean.log]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check missing episode videos according to episodes.jsonl"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root path containing meta/ and videos/"
    )
    parser.add_argument(
        "--video-subdir",
        type=str,
        default="videos/chunk-000/observation.images.chest_rgb",
        help="Relative path to video folder (e.g. videos/chunk-000/observation.images.chest_rgb)"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="manually_clean.log",
        help="Output log file path (relative to dataset-root if not absolute)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    episodes_jsonl = dataset_root / "meta/episodes.jsonl"
    video_dir = dataset_root / args.video_subdir

    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = dataset_root / log_path

    assert episodes_jsonl.exists(), f"Missing {episodes_jsonl}"
    assert video_dir.exists(), f"Missing {video_dir}"

    # Collect existing video filenames
    existing_videos = {
        p.name for p in video_dir.glob("episode_*.mp4")
    }

    missing = []

    with open(episodes_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            info = json.loads(line)
            ep_idx = info["episode_index"]
            expected_name = f"episode_{ep_idx:06d}.mp4"

            if expected_name not in existing_videos:
                missing.append({
                    "episode_index": ep_idx,
                    "expected_file": expected_name,
                    "task": info.get("tasks"),
                    "length": info.get("length")
                })

    # Write log
    with open(log_path, "w") as log:
        log.write("[Missing Episode Video Check]\n")
        log.write(f"Time: {datetime.now().isoformat()}\n")
        log.write(f"Video directory: {video_dir}\n")
        log.write(f"Episodes jsonl: {episodes_jsonl}\n")
        log.write(f"Total episodes: {sum(1 for _ in open(episodes_jsonl))}\n")
        log.write(f"Missing count: {len(missing)}\n\n")

        for item in missing:
            log.write(
                f"episode_index={item['episode_index']:06d}, "
                f"task={item['task']}, "
                f"length={item['length']}, "
                f"missing_file={item['expected_file']}\n"
            )

        log.write(f"\nMissing episodes: {len(missing)}, index: {[episode['episode_index'] for episode in missing]}")

    print(f"[Done] Missing episodes: {len(missing)}, index: {[episode['episode_index'] for episode in missing]}")
    print(f"[Log saved to] {log_path}")

if __name__ == "__main__":
    main()
