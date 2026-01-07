#!/usr/bin/env python3
"""
Video downscaling script using FFmpeg

Traverses a folder of video chunks, rescales videos to target resolution while maintaining aspect ratio,
and updates meta/info.json with the new resolution.

Usage:
    python reshape_video_images.py --folder <path> --resolution 128 128
"""

import argparse
import json
from pathlib import Path
from subprocess import run, DEVNULL

from tqdm import tqdm

def get_new_dimensions(orig_h: int, orig_w: int, target_h: int) -> tuple[int, int]:
    """Compute new width maintaining aspect ratio for given target height."""
    scale = target_h / orig_h
    return target_h, int(orig_w * scale)

def process_video_ffmpeg(video_path: Path, target_h: int) -> tuple[int, int]:
    """
    Resize video using FFmpeg while keeping aspect ratio.

    Args:
        video_path: path to video file
        target_h: target height

    Returns:
        new height and width
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return 0, 0
    orig_h, orig_w = frame.shape[:2]
    new_h, new_w = get_new_dimensions(orig_h, orig_w, target_h)
    temp_file = video_path.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"scale={new_w}:{new_h}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an", str(temp_file)
    ]
    run(cmd, stdout=DEVNULL, stderr=DEVNULL)
    video_path.unlink()
    temp_file.rename(video_path)
    return new_h, new_w

def update_meta(meta_path: Path, target_h: int):
    """Update meta/info.json with resized video dimensions."""
    with meta_path.open("r") as f:
        meta = json.load(f)
    for key, val in meta.get("features", {}).items():
        if val.get("dtype") == "video" and "shape" in val:
            orig_h, orig_w = val["shape"][:2]
            new_h, new_w = get_new_dimensions(orig_h, orig_w, target_h)
            val["shape"][0] = new_h
            val["shape"][1] = new_w
            if "info" in val:
                val["info"]["video.height"] = new_h
                val["info"]["video.width"] = new_w
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Resize videos and update meta info using FFmpeg.")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--resolution", type=int, nargs=2, required=True)
    args = parser.parse_args()

    folder = Path(args.folder)
    target_h = args.resolution[0]

    video_files = list(folder.glob("videos/chunk-*/**/episode_*.mp4")) + \
                  list(folder.glob("videos/chunk-*/**/episode_*.avi"))

    print(f"Found {len(video_files)} videos. Resizing to height {target_h}...")
    for vf in tqdm(video_files, desc="Processing videos", unit="video"):
        process_video_ffmpeg(vf, target_h)

    meta_path = folder / "meta/info.json"
    if meta_path.exists():
        update_meta(meta_path, target_h)
        print(f"Updated meta info at {meta_path}")

if __name__ == "__main__":
    main()
