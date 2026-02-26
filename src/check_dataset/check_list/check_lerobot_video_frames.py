#!/usr/bin/env python3
"""
check_lerobot_video_frames.py

Verify video frame counts in LeRobot datasets against JSONL episode records.

This script automatically scans all 'chunk-*' directories under 'videos/' and compares
the actual frame counts of each perspective video against the expected length recorded
in 'meta/episodes.jsonl'. It reports any discrepancies and can optionally be used
as an external function.

Command-line usage:
    python check_lerobot_video_frames.py --dataset-root </path/to/dataset> [--frame-threshold N]

    --dataset-root      Path to the root of the LeRobot dataset (must contain meta/episodes.jsonl
                        and videos/chunk-* directories)
    --frame-threshold   Maximum allowed difference between actual and expected frames (default: 5)

External interface usage:
    from check_lerobot_video_frames import check_lerobot_video_frames_func
    success = check_lerobot_video_frames_func("/path/to/dataset", frame_diff_threshold=1)

Returns True if all videos match the expected frame counts within the threshold,
False if any abnormalities are found.
"""


import os
import json
import glob
from typing import Dict, List
from tqdm import tqdm
import ffmpeg


def get_video_actual_frames(video_path: str) -> int:
    """Return actual frame count of a video; -1 if parsing fails or file missing."""
    if not os.path.exists(video_path):
        tqdm.write(f"  Video file does not exist: {video_path}")
        return -1
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"),
            None
        )
        if not video_stream:
            tqdm.write(f" No video stream found: {video_path}")
            return -1
        frames = int(video_stream.get("nb_frames", -1))
        return frames if frames > 0 else -1
    except Exception as e:
        tqdm.write(f" Failed to parse video {video_path}: {str(e)}")
        return -1


def load_jsonl_episodes(jsonl_path: str) -> Dict[int, int]:
    """Load episode_index -> expected frame count mapping from JSONL."""
    episodes = {}
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f" JSONL file does not exist: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ep_idx = data.get("episode_index")
                ep_length = data.get("length")
                if ep_idx is None or ep_length is None:
                    tqdm.write(f"  Missing key fields in line {line_num}: {line}")
                    continue
                episodes[int(ep_idx)] = int(ep_length)
            except json.JSONDecodeError as e:
                tqdm.write(f"  JSON format error in line {line_num}: {line}, error: {str(e)}")
    return episodes


def check_episode_video_mismatch(dataset_root: str, frame_diff_threshold: int = 5) -> List[Dict]:
    """
    Compare actual video frame counts with JSONL records.
    Returns list of abnormal episodes.
    """
    jsonl_path = os.path.join(dataset_root, "meta", "episodes.jsonl")
    video_root_pattern = os.path.join(dataset_root, "videos", "chunk-*")
    video_chunks = sorted(glob.glob(video_root_pattern))

    camera_views = [
        "observation.images.chest_rgb",
        "observation.images.head_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb"
    ]

    tqdm.write(f"Loading JSONL file: {jsonl_path}")
    ep_expected_length = load_jsonl_episodes(jsonl_path)
    if not ep_expected_length:
        raise ValueError("No episode data loaded from JSONL")
    tqdm.write(f"Successfully loaded expected lengths for {len(ep_expected_length)} episodes")

    abnormal_episodes = []
    sorted_ep_indices = sorted(ep_expected_length.keys())
    tqdm.write(f"\nStarting verification of video frame counts for {len(sorted_ep_indices)} episodes "
            f"(error threshold: {frame_diff_threshold} frames)")

    episode_to_chunk = {}
    for chunk_path in video_chunks:
        chest_dir = os.path.join(chunk_path, "observation.images.chest_rgb")
        if not os.path.exists(chest_dir):
            continue
        for mp4 in glob.glob(os.path.join(chest_dir, "episode_*.mp4")):
            ep_idx = int(os.path.basename(mp4).split("_")[1].split(".")[0])
            episode_to_chunk[ep_idx] = chunk_path

    for ep_idx in tqdm(sorted_ep_indices, desc="Verification progress"):
        expected_frames = ep_expected_length[ep_idx]
        if ep_idx not in episode_to_chunk:
            tqdm.write(f"[WARN] Episode {ep_idx:06d} not found in any chunk")
            continue

        chunk_path = episode_to_chunk[ep_idx]
        video_filename = f"episode_{ep_idx:06d}.mp4"
        view_actual_frames = {}
        is_abnormal = False

        for view in camera_views:
            view_path = os.path.join(chunk_path, view)
            video_path = os.path.join(view_path, video_filename)
            actual_frames = get_video_actual_frames(video_path)
            view_actual_frames[f"{os.path.basename(chunk_path)}/{view}"] = actual_frames
            if actual_frames == -1 or abs(actual_frames - expected_frames) > frame_diff_threshold:
                is_abnormal = True

        if is_abnormal:
            view_frame_diffs = {}
            for view, actual in view_actual_frames.items():
                if actual == -1:
                    view_frame_diffs[view] = "Unknown (file does not exist/parsing failed)"
                else:
                    view_frame_diffs[view] = f"{actual} (difference: {abs(actual - expected_frames)} frames)"
            abnormal_episodes.append({
                "chunk": os.path.basename(chunk_path),
                "episode_index": ep_idx,
                "expected_frames": expected_frames,
                "actual_frames_per_view": view_frame_diffs,
                "is_abnormal": True
            })
    return abnormal_episodes


def print_abnormal_summary(abnormal_episodes: List[Dict]):
    """Print summary report of abnormal episodes."""
    if not abnormal_episodes:
        tqdm.write("\nAll episode video frame counts match JSONL records, no abnormalities!")
        return
    tqdm.write(f"\nFound {len(abnormal_episodes)} abnormal episodes in total:")
    tqdm.write("-" * 120)
    for idx, ep in enumerate(abnormal_episodes, 1):
        tqdm.write(f"\n[Abnormality {idx}] Episode index: {ep['episode_index']}")
        tqdm.write(f"  Expected frame count (JSONL record): {ep['expected_frames']}")
        tqdm.write("  Actual frame counts per perspective:")
        for view, diff_info in ep["actual_frames_per_view"].items():
            simple_view = view.replace("observation.images.", "")
            tqdm.write(f"    - {simple_view:16s}: {diff_info}")
    tqdm.write("-" * 120)


def check_lerobot_video_frames_func(dataset_root: str, frame_diff_threshold: int = 1) -> bool:
    """
    External interface to check video frames vs JSONL.
    Returns True if no abnormalities, False otherwise.
    Printing behavior matches command-line output.
    """
    try:
        abnormal_eps = check_episode_video_mismatch(
            dataset_root=dataset_root,
            frame_diff_threshold=frame_diff_threshold
        )
        print_abnormal_summary(abnormal_eps)
        return len(abnormal_eps) == 0
    except Exception as e:
        tqdm.write(f"\nScript execution failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lerobot dataset video length and JSONL verification script")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root path (must contain meta/episodes.jsonl and videos/chunk-*)"
    )
    parser.add_argument(
        "--frame-threshold",
        type=int,
        default=5,
        help="Frame count error threshold (exceed to determine abnormality, default 5 frames)"
    )
    args = parser.parse_args()

    check_lerobot_video_frames_func(dataset_root=args.dataset_root, frame_diff_threshold=args.frame_threshold)
