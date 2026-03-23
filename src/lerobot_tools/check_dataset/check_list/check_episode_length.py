#!/usr/bin/env python3
"""
check_episode_length.py

Validate the consistency between episode lengths recorded in `episodes.jsonl`
and the actual number of rows in corresponding `episode_XXXXXX.parquet` files.

For each episode:
    - Read row count from parquet metadata.
    - Compare against the `length` field in episodes.jsonl.
    - Report mismatches.
    - Optionally overwrite incorrect `length` values when --fix is enabled.

This script also checks:
    - Missing episodes in episodes.jsonl
    - Episodes present in jsonl but missing corresponding parquet files
    - Duplicate episode indices in parquet directory (raises error)

----------------------------------------------------------------------
Command-line usage:

    python check_episode_length.py \
        --parquet-dir </path/to/parquet_dir> \
        --episode-jsonl-path </path/to/episodes.jsonl> \
        [--fix]

Options:
    --fix    Apply in-place correction to episodes.jsonl.
             If not specified, runs in dry-run mode.

----------------------------------------------------------------------
Programmatic usage:

    from check_episode_length import check_episode_length_func

    success = check_episode_length_func(
        parquet_dir="/path/to/parquet_dir",
        episode_jsonl_path="/path/to/episodes.jsonl",
        fix=True
    )

Return value:
    True  -> No inconsistencies found (or fix mode executed successfully)
    False -> Inconsistencies detected (in dry-run mode)
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

def collect_parquet_files(root_dir: str):
    files = {}
    for root, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.startswith("episode_") or not fname.endswith(".parquet"):
                continue
            ep = extract_episode_index(fname)
            if ep is None:
                continue
            full_path = os.path.join(root, fname)
            if ep in files:
                raise ValueError(
                    f"Duplicate episode index {ep:06d} found:\n"
                    f"  {files[ep]}\n"
                    f"  {full_path}"
                )
            files[ep] = full_path
    return files

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def check_episode_length_func(parquet_dir: str, episode_jsonl_path: str, fix: bool = False) -> bool:
    """
    Validate episode length consistency between parquet files and episodes.jsonl.

    Parameters
    ----------
    parquet_dir : str
        Root directory containing episode_XXXXXX.parquet files.
        Files are recursively discovered.

    episode_jsonl_path : str
        Path to episodes.jsonl containing per-episode metadata.
        Each record must contain:
            - "episode_index" (int)
            - "length" (int)

    fix : bool, default=False
        If True, overwrite incorrect `length` fields in episodes.jsonl.
        If False, perform validation only (dry-run mode).

    Behavior
    --------
    - Reads parquet row count using metadata (no full file scan).
    - Compares against recorded length in episodes.jsonl.
    - Reports:
        * Missing episodes in jsonl
        * Length mismatches
        * Episodes present in jsonl but missing parquet file
    - Raises ValueError if duplicate parquet episode indices are found.

    Returns
    -------
    bool
        True  -> No mismatches found (dry-run), or fix mode completed.
        False -> Mismatches detected in dry-run mode.
    """
    # Collect parquet episode lengths
    parquet_lengths = {
        ep: pq.read_metadata(path).num_rows
        for ep, path in collect_parquet_files(parquet_dir).items()
    }
    if not parquet_lengths:
        print("[ERROR] No valid episode parquet files found.")
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

    json_episode_indices = set(stats_by_ep.keys())
    parquet_episode_indices = set(parquet_lengths.keys())
    extra_in_json = json_episode_indices - parquet_episode_indices
    for ep in sorted(extra_in_json):
        tqdm.write(f"[ERROR] Episode {ep:06d} exists in jsonl but missing parquet")
        error_count += 1

    # Apply fixes
    if fix and error_count > 0:
        write_jsonl(episode_jsonl_path, stats)
        tqdm.write("episodes.jsonl updated successfully.")

    print("\n================ SUMMARY ================")
    print(f"Total episodes checked : {len(parquet_lengths)}")
    print(f"Length mismatches      : {error_count}")
    print(f"Fix mode               : {fix}")
    print("========================================\n")

    if fix:
        return True
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
