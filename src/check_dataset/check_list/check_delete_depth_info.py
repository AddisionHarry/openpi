#!/usr/bin/env python3
"""
check_delete_depth_info.py

Script to check and optionally remove fields containing both "observation" and "depth" in info.json file.

Command-line usage:
    python check_delete_depth_info.py </path/to/info.json> [--fix]

External interface usage:
    from check_delete_depth_info import check_delete_depth_info
    success = check_delete_depth_info_func("/path/to/info.json", fix=False)
"""

import os
import json

def find_bad_keys(obj, path=""):
    """Return list of (full_key_path, value) where key contains both 'observation' and 'depth'."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lower = k.lower()
            if "observation" in lower and "depth" in lower:
                bad.append((path + "/" + k, v))
            bad += find_bad_keys(v, path + "/" + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += find_bad_keys(v, path + f"[{i}]")
    return bad

def remove_bad_keys(obj):
    """Recursively remove keys containing both 'observation' & 'depth'."""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            lower = k.lower()
            if not ("observation" in lower and "depth" in lower):
                new[k] = remove_bad_keys(v)
        return new
    elif isinstance(obj, list):
        return [remove_bad_keys(v) for v in obj]
    return obj

def show_context(lines, target_line, window=3):
    """Print lines around a target line."""
    start = max(0, target_line - window - 1)
    end = min(len(lines), target_line + window)
    ctx = lines[start:end]
    print("\n---- Context ----")
    for idx, l in enumerate(ctx, start=start + 1):
        mark = ">>" if idx == target_line else "  "
        print(f"{mark} {idx}: {l.rstrip()}")
    print("-----------------\n")

def check_delete_depth_info_func(input_path, fix=False):
    """
    Check and optionally remove fields containing 'observation' and 'depth'.

    Parameters
    ----------
    input_path : str
        Path to JSON or JSONL file to check.
    fix : bool
        If True, remove illegal fields in-place.

    Returns
    -------
    bool
        True if file has no illegal keys or illegal keys were successfully removed,
        False if illegal keys were found but fix=False.
    """
    print(f"Checking: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: File not found - {input_path}")
        return False

    # Read raw lines for context printing
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Try JSON / JSONL
    try:
        data = json.loads("".join(raw_lines))
        is_list = False
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw_lines if line.strip()]
        is_list = True

    had_issue = False

    if is_list:
        iterable = enumerate(data)
    else:
        iterable = [(1, data)]

    for idx, item in iterable:
        bad = find_bad_keys(item)
        if not bad:
            continue

        had_issue = True
        print(f"\nFound illegal keys in entry at line {idx}:")
        for key_path, _ in bad:
            print(f"  - {key_path}")

        for key_path, _ in bad:
            key = key_path.split("/")[-1].strip()
            for lineno, line in enumerate(raw_lines, start=1):
                if f'"{key}"' in line.replace(" ", ""):
                    show_context(raw_lines, lineno)
                    break

    if not had_issue:
        print("No illegal depth fields found. File is correct.")
        return True

    if not fix:
        print("Detected illegal fields. Use --fix to remove them.")
        return False

    # Perform deletion
    print("🔧 Removing keys (in-place)...")
    cleaned = remove_bad_keys(data)

    with open(input_path, "w", encoding="utf-8") as f:
        if is_list:
            for obj in cleaned:
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("Finished cleaning file (in-place modified).")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Check and optionally remove "observation"+"depth" fields from JSON/JSONL'
    )
    parser.add_argument("input_path", help="JSON or JSONL file to check")
    parser.add_argument("--fix", action="store_true", help="Delete illegal keys in-place")

    args = parser.parse_args()
    check_delete_depth_info_func(args.input_path, fix=args.fix)
