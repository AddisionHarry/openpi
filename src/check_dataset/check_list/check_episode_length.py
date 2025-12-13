#!/usr/bin/env python3
"""
check_episode_length.py

Script to check and optionally fix episode length recorded in
episodes.jsonl by comparing against actual parquet file lengths.

Command-line usage:
    python check_episode_length.py \
        --parquet-dir </path/to/parquet> \
        --episode-jsonl-path </path/to/episodes.jsonl> \
        [--fix]

External interface usage:
    from check_episode_length import check_episode_length_func
    success = check_episode_length_func("/path/to/parquet", "/path/to/episodes.jsonl", fix=True)
"""

import os
import re
import json
import pyarrow.parquet as pq
from tqdm import tqdm


def extract_episode_index(filename: str):
    """Extract integer episode index from filename like episode_000063.parquet"""
    m = re.search(r"episode_(\d+)\.parquet$", filename)
    return int(m.group(1)) if m else None


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def check_episode_length_func(
    parquet_dir: str,
    episode_jsonl_path: str,
    fix: bool = False
) -> bool:
    """
    Check episode length consistency between parquet files and episodes.jsonl.

    If fix=True, update incorrect `length` fields in episodes.jsonl.

    Returns True if no mismatches found, False otherwise.
    """
    # Collect parquet episode lengths
    parquet_lengths = {}
    for fname in os.listdir(parquet_dir):
        if not fname.endswith(".parquet"):
            continue
        ep = extract_episode_index(fname)
        if ep is None:
            raise ValueError(f"Unrecognized parquet filename: {fname}")
        path = os.path.join(parquet_dir, fname)
        parquet_lengths[ep] = pq.read_metadata(path).num_rows
    if not parquet_lengths:
        print("[Error] No valid episode parquet files found.")
        return False

    # Load stats
    stats = load_jsonl(episode_jsonl_path)
    stats_by_ep = {s["episode_index"]: s for s in stats}
    mode = "FIX MODE (STATS FILE WILL BE MODIFIED)" if fix else "DRY RUN (NO MODIFICATION)"
    print(f"Checking {len(parquet_lengths)} episodes... [{mode}]\n")
    error_count = 0
    for ep_idx, parquet_len in tqdm(sorted(parquet_lengths.items()), desc="Checking"):
        stat = stats_by_ep.get(ep_idx)
        if stat is None:
            tqdm.write(f"[ERROR] Missing episode {ep_idx:06d} in episodes.jsonl")
            error_count += 1
            continue
        recorded_len = stat.get("length")
        if recorded_len != parquet_len:
            error_count += 1
            tqdm.write(
                f"\n[ERROR] Length mismatch in episode_{ep_idx:06d}\n"
                f"        Parquet length : {parquet_len}\n"
                f"        Stats length   : {recorded_len}"
            )
            if not fix:
                tqdm.write("        (Dry-run: NOT modifying stats)\n")
                continue
            stat["length"] = parquet_len
            tqdm.write("        Fixed in episodes.jsonl\n")

    # Apply fixes
    if fix and error_count > 0:
        write_jsonl(episode_jsonl_path, stats)
        tqdm.write("episodes.jsonl updated successfully.")

    print("\n================ SUMMARY ================")
    print(f"Total episodes checked : {len(parquet_lengths)}")
    print(f"Length mismatches      : {error_count}")
    print(f"Fix mode               : {fix}")
    print("========================================\n")

    return error_count == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and optionally fix episode length in episodes.jsonl"
    )
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory containing episode_XXXXXX.parquet files"
    )
    parser.add_argument(
        "--episode-jsonl-path",
        required=True,
        help="Path to episodes.jsonl"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes to episodes.jsonl (default: dry-run)"
    )
    args = parser.parse_args()

    check_episode_length_func(args.parquet_dir, args.episode_jsonl_path, fix=args.fix)
