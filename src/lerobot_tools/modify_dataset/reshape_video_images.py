#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced video reshape script using FFmpeg

Supports:
  1. Force resize (change aspect ratio)
  2. Keep ratio + minimal loss (no crop, may exceed target)
  3. Keep ratio + zero padding (letterbox)

Usage:
    python reshape_video_images.py \
        --folder <path> \
        --resolution 128 128 \
        [--force-resize] \
        [--zero-pad]
"""

import argparse
import json
from pathlib import Path
from subprocess import run, DEVNULL

from tqdm import tqdm


# =========================
# Dimension helpers
# =========================

def compute_scale_dims(orig_h, orig_w, target_h, target_w, mode):
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    if mode == "force":
        return target_h, target_w

    elif mode == "pad":
        scale = min(scale_h, scale_w)

    else:  # minimal loss (no pad)
        scale = max(scale_h, scale_w)

    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    return new_h, new_w


# =========================
# FFmpeg processing
# =========================

def build_ffmpeg_filter(orig_h, orig_w, target_h, target_w, mode):
    new_h, new_w = compute_scale_dims(orig_h, orig_w, target_h, target_w, mode)

    if mode == "force":
        return f"scale={target_w}:{target_h}"

    elif mode == "pad":
        # scale + pad to center
        return (
            f"scale={new_w}:{new_h},"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
        )

    else:
        # minimal loss (no pad)
        return f"scale={new_w}:{new_h}"


def process_video_ffmpeg(video_path: Path, target_h: int, target_w: int, mode: str):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"WARNING: Failed to read {video_path}")
        return 0, 0

    orig_h, orig_w = frame.shape[:2]

    vf_filter = build_ffmpeg_filter(orig_h, orig_w, target_h, target_w, mode)

    temp_file = video_path.with_suffix(".tmp.mp4")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(temp_file)
    ]

    run(cmd, stdout=DEVNULL, stderr=DEVNULL)

    video_path.unlink()
    temp_file.rename(video_path)

    new_h, new_w = compute_scale_dims(orig_h, orig_w, target_h, target_w, mode)

    if mode == "pad":
        return target_h, target_w

    if mode == "force":
        return target_h, target_w

    return new_h, new_w


# =========================
# Meta update
# =========================

def update_meta(meta_path: Path, target_h, target_w, mode):
    with meta_path.open("r") as f:
        meta = json.load(f)

    for key, val in meta.get("features", {}).items():
        if val.get("dtype") == "video" and "shape" in val:
            orig_h, orig_w = val["shape"][:2]

            new_h, new_w = compute_scale_dims(orig_h, orig_w, target_h, target_w, mode)

            if mode == "pad":
                val["shape"][0] = target_h
                val["shape"][1] = target_w
            elif mode == "force":
                val["shape"][0] = target_h
                val["shape"][1] = target_w
            else:
                val["shape"][0] = new_h
                val["shape"][1] = new_w

            if "info" in val:
                val["info"]["video.height"] = val["shape"][0]
                val["info"]["video.width"] = val["shape"][1]

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Advanced video reshape tool")

    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--resolution", type=int, nargs=2, required=True)

    parser.add_argument("--force-resize", action="store_true")
    parser.add_argument("--zero-pad", action="store_true")

    args = parser.parse_args()

    folder = Path(args.folder)
    target_h, target_w = args.resolution

    # determine mode
    if args.force_resize:
        mode = "force"
    elif args.zero_pad:
        mode = "pad"
    else:
        mode = "minimal"

    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Target: {target_h} x {target_w}")
    print("=" * 60)

    video_files = list(folder.glob("videos/chunk-*/**/episode_*.mp4")) + \
                  list(folder.glob("videos/chunk-*/**/episode_*.avi"))

    print(f"Found {len(video_files)} videos")

    for vf in tqdm(video_files, desc="Processing", unit="video"):
        process_video_ffmpeg(vf, target_h, target_w, mode)

    meta_path = folder / "meta/info.json"
    if meta_path.exists():
        update_meta(meta_path, target_h, target_w, mode)
        print(f"Updated meta: {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()
