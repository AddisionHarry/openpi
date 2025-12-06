#!/usr/bin/env python3
"""
Unified cleaner for:
  - episodes_stats.jsonl (JSON Lines)

Capabilities:
  Detect and remove keys where name contains both "observation" and "depth"
  Auto-detect JSON / JSONL
  Safe in-place fix (--fix)
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
    """Recursively remove keys containing both 'observation' and 'depth'."""
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


def check_delete_depth_jsondata_func(input_path, fix=False):
    """
    Check and optionally remove fields containing 'observation' and 'depth'.
    Supports both JSON and JSONL transparently.

    Returns True if:
      - No illegal keys found, OR
      - Illegal keys found and removed (fix=True)

    Returns False if illegal fields found while fix=False.
    """
    print(f"Checking: {input_path}")

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        return False

    # Load raw lines for context
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    try:
        data = json.loads("".join(raw_lines))
        is_list = False
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw_lines if line.strip()]
        is_list = True

    had_issue = False

    # JSONL -> enumerate lines
    if is_list:
        items = enumerate(data, start=1)
    else:
        items = [(1, data)]

    for entry_line, item in items:
        bad_keys = find_bad_keys(item)
        if not bad_keys:
            continue

        had_issue = True
        print(f"\nFound illegal keys in entry at line {entry_line}:")
        for key_path, _ in bad_keys:
            print(f"  - {key_path}")

        # show context for each offending key
        for key_path, _ in bad_keys:
            key = key_path.split("/")[-1].strip()
            for ln, line in enumerate(raw_lines, start=1):
                if f'"{key}"' in line.replace(" ", ""):
                    break

    if not had_issue:
        print("No illegal depth fields found. File is correct.")
        return True

    if not fix:
        print("Detected illegal fields. Use --fix to remove them.")
        return False

    print("Removing illegal depth-observation keys...")
    cleaned = remove_bad_keys(data)

    with open(input_path, "w", encoding="utf-8") as f:
        if is_list:  # JSONL
            for obj in cleaned:
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")
        else:  # normal JSON
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("Finished cleaning. File updated in-place.")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Check and optionally remove "observation"+"depth" keys from JSON/JSONL files.'
    )
    parser.add_argument("input_path", help="JSON / JSONL file path")
    parser.add_argument("--fix", action="store_true", help="Fix illegal keys")

    args = parser.parse_args()
    check_delete_depth_jsondata_func(args.input_path, fix=args.fix)
