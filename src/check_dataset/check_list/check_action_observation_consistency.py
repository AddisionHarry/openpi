#!/usr/bin/env python3
"""
check_action_observation_consistency_func.py

Checks consistency between actions and observation.state
ONLY for joints whose action names end with "_pos".

Optional exclusion:
    --exclude-neck : ignore joints whose name contains 'neck'

Violation condition:
    abs(action - observation.state) * fps >= position_threshold

Where:
    - fps is read from meta/info.json['fps']
    - position_threshold is a CLI argument (default = 20)

Directory layout:
    dataset_root/
        ├── data/
        │   └── chunk-*/episode_*.parquet
        └── meta/
            └── info.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Meta loading
# ============================================================

def load_fps_and_pos_joints(
    dataset_root: Path,
    exclude_neck: bool,
) -> Tuple[float, List[int], List[str]]:
    """
    Load fps and filtered *_pos joint indices/names from meta/info.json
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found: {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    try:
        fps = float(info["fps"])
        action_names = info["features"]["actions"]["names"]
    except KeyError as e:
        raise KeyError(f"Missing key in meta/info.json: {e}")

    pos_indices = []
    pos_names = []

    for i, name in enumerate(action_names):
        lname = name.lower()

        if not lname.endswith("_pos"):
            continue

        if exclude_neck and "neck" in lname:
            continue

        pos_indices.append(i)
        pos_names.append(name)

    if not pos_indices:
        raise RuntimeError(
            "No valid *_pos joints found after applying filters "
            f"(exclude_neck={exclude_neck})"
        )

    return fps, pos_indices, pos_names


# ============================================================
# Core check
# ============================================================

def check_action_observation_consistency_func(
    dataset_root: str,
    position_threshold: float = 20.0,
    exclude_neck: bool = False,
) -> bool:
    """
    Returns True if all frames pass the check.
    """
    dataset_root = Path(dataset_root)
    data_root = dataset_root / "data"

    if not data_root.exists():
        raise FileNotFoundError(f"data directory not found: {data_root}")

    fps, pos_indices, pos_names = load_fps_and_pos_joints(
        dataset_root,
        exclude_neck=exclude_neck,
    )

    parquet_files = sorted(data_root.glob("chunk-*/*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    print(f"FPS = {fps}")
    print(f"Position threshold = {position_threshold}")
    print(f"Checked joints = {len(pos_indices)} (*_pos only)")
    print(f"Exclude neck joints = {exclude_neck}\n")

    total_frames = 0
    error_frames = 0

    for fpath in tqdm(parquet_files, desc="Checking parquet files"):
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to read {fpath}: {e}")
            error_frames += 1
            continue

        required_cols = ["observation.state", "actions", "frame_index"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            tqdm.write(f"[ERROR] {fpath.name} missing columns: {missing}")
            error_frames += 1
            continue

        for _, row in df.iterrows():
            total_frames += 1

            obs = np.asarray(row["observation.state"], dtype=np.float32)
            act = np.asarray(row["actions"], dtype=np.float32)

            if obs.shape != act.shape:
                tqdm.write(
                    f"[ERROR] {fpath.name} frame={row['frame_index']} "
                    f"shape mismatch: obs={obs.shape}, act={act.shape}"
                )
                error_frames += 1
                continue

            obs_pos = obs[pos_indices]
            act_pos = act[pos_indices]

            diff = np.abs(act_pos - obs_pos)
            scaled_diff = diff * fps

            bad_mask = scaled_diff >= position_threshold
            if not np.any(bad_mask):
                continue

            error_frames += 1

            bad_entries = []
            for local_i in np.where(bad_mask)[0]:
                bad_entries.append(
                    f"{pos_names[local_i]}(idx={pos_indices[local_i]}): "
                    f"diff={diff[local_i]:.6f}, diff*fps={scaled_diff[local_i]:.3f}"
                )

            tqdm.write(
                f"[VIOLATION] {fpath.name} | frame={row['frame_index']} | "
                f"{len(bad_entries)} joints failed | "
                + " ; ".join(bad_entries)
            )

    print("\n========== Summary ==========")
    print(f"Total parquet files checked : {len(parquet_files)}")
    print(f"Total frames checked        : {total_frames}")
    print(f"Frames with violations     : {error_frames}")
    print(f"FPS                        : {fps}")
    print(f"Position threshold         : {position_threshold}")
    print(f"Exclude neck joints        : {exclude_neck}")

    return error_frames == 0


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check action vs observation consistency for *_pos joints "
                    "using scaled threshold (diff * fps)"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing data/ and meta/",
    )
    parser.add_argument(
        "--position-threshold",
        type=float,
        default=20.0,
        help="Threshold for |action - observation| * fps (default: 20)",
    )
    parser.add_argument(
        "--exclude-neck",
        action="store_true",
        help="Exclude joints whose name contains 'neck'",
    )

    args = parser.parse_args()

    ok = check_action_observation_consistency_func(
        args.dataset_root,
        position_threshold=args.position_threshold,
        exclude_neck=args.exclude_neck,
    )

    if not ok:
        exit(1)


if __name__ == "__main__":
    main()
