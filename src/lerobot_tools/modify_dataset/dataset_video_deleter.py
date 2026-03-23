#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_video_deleter.py

Unified dataset cleaning tool for OpenPI-style datasets.

This script performs two main operations:
1. Remove keys from JSON / JSONL files that match given patterns
2. Remove corresponding video directories under videos/

Features:
- Fuzzy key matching (e.g. "depth", "chest_depth")
- Supports both JSON and JSONL formats
- Works on full dataset root recursively
- Safe mode by default (dry-run), use --fix to apply changes

Command line usage:

Required arguments:
    dataset_root            Root directory of dataset
    --patterns              Patterns to match for deletion

Optional arguments:
    --fix                   Enable actual deletion (default is dry-run)

Examples:

Dry-run (recommended first):
    python dataset_video_deleter.py /data/my_dataset --patterns depth

Apply deletion:
    python dataset_video_deleter.py /data/my_dataset --patterns depth chest_depth --fix
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =========================
# Matching logic
# =========================

def match_key(key: str, patterns: List[str]) -> bool:
    """Check if a key matches any pattern using case-insensitive substring matching."""
    key_lower = key.lower()
    for p in patterns:
        if p.lower() in key_lower:
            return True
    return False


def find_bad_keys(obj: Any, patterns: List[str], path: str = "") -> List[Tuple[str, Any]]:
    """Recursively find keys matching patterns in nested JSON-like structures."""
    bad: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if match_key(k, patterns):
                bad.append((path + "/" + k, v))
            bad.extend(find_bad_keys(v, patterns, path + "/" + k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad.extend(find_bad_keys(v, patterns, path + f"[{i}]"))
    return bad


def remove_bad_keys(obj: Any, patterns: List[str]) -> Any:
    """Recursively remove keys matching patterns from JSON-like structures."""
    if isinstance(obj, dict):
        new_obj: Dict[str, Any] = {}
        for k, v in obj.items():
            if not match_key(k, patterns):
                new_obj[k] = remove_bad_keys(v, patterns)
        return new_obj
    if isinstance(obj, list):
        return [remove_bad_keys(v, patterns) for v in obj]
    return obj


# =========================
# JSON / JSONL cleaner
# =========================

def clean_json_file(path: Path, patterns: List[str], fix: bool = False) -> bool:
    """Check and optionally clean a JSON or JSONL file."""
    print(f"\nChecking JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    try:
        data = json.loads("".join(raw_lines))
        is_jsonl = False
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw_lines if line.strip()]
        is_jsonl = True
    had_issue = False
    items = enumerate(data, start=1) if is_jsonl else [(1, data)]
    for idx, item in items:
        bad = find_bad_keys(item, patterns)
        if bad:
            had_issue = True
            print(f"  Found in entry {idx}:")
            for k, _ in bad:
                print(f"    - {k}")
    if not had_issue:
        print("  OK")
        return True
    if not fix:
        print("  Use --fix to remove")
        return False
    cleaned = remove_bad_keys(data, patterns)
    with open(path, "w", encoding="utf-8") as f:
        if is_jsonl:
            for obj in cleaned:
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print("  Cleaned")
    return True


def count_total_videos(dataset_root: Path) -> int:
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        return 0
    total = 0
    for path in videos_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in video_exts:
            total += 1
    return total

# =========================
# Video directory cleaner
# =========================

def clean_video_dirs(dataset_root: Path, patterns: List[str], fix: bool = False) -> None:
    """Remove video subdirectories under videos/ that match patterns."""
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        print("No videos directory found, skip.")
        return
    print(f"\nScanning video dirs: {videos_root}")
    for chunk in sorted(videos_root.glob("chunk-*")):
        if not chunk.is_dir():
            continue
        print(f"\nChunk: {chunk.name}")

        for sub in chunk.iterdir():
            if not sub.is_dir():
                continue
            if match_key(sub.name, patterns):
                print(f"  MATCH: {sub}")
                if fix:
                    shutil.rmtree(sub)
                    print("    Deleted")
                else:
                    print("    (dry-run)")


def update_info_json(dataset_root: Path, fix=False):
    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        print("No info.json found, skip.")
        return
    print("\nUpdating info.json total_videos...")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    new_total = count_total_videos(dataset_root)
    old_total = info.get("total_videos", None)
    print(f"  old total_videos: {old_total}")
    print(f"  new total_videos: {new_total}")
    if not fix:
        print("  (dry-run)")
        return
    info["total_videos"] = new_total
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("  info.json updated")

# =========================
# Main pipeline
# =========================

def run_cleaner(dataset_root: str, patterns: List[str], fix: bool = False) -> None:
    """Run full dataset cleaning pipeline."""
    root = Path(dataset_root)

    if not root.exists():
        print(f"ERROR: dataset root not found: {root}")
        return

    print("=" * 60)
    print("Dataset Video Deleter")
    print(f"Root: {root}")
    print(f"Patterns: {patterns}")
    print(f"Fix mode: {fix}")
    print("=" * 60)

    for path in root.rglob("*"):
        if path.suffix in [".json", ".jsonl"]:
            clean_json_file(path, patterns, fix=fix)

    clean_video_dirs(root, patterns, fix=fix)
    update_info_json(root, fix)

    print("\nDone.")


# =========================
# CLI
# =========================

def main() -> None:
    """Command line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset cleaner for JSON and video directories")
    parser.add_argument("dataset_root", type=str, help="Dataset root path")
    parser.add_argument("--patterns", nargs="+", required=True, help="Patterns to match for deletion")
    parser.add_argument("--fix", action="store_true", help="Enable actual deletion")

    args = parser.parse_args()
    run_cleaner(args.dataset_root, args.patterns, fix=args.fix)


if __name__ == "__main__":
    main()
