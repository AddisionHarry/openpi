#!/usr/bin/env python3
"""
Add hand_align_state to LeRobot dataset.

- Detect first rising edge per episode
- Build alignment mask ending at the rising edge
- Append as last dimension of observation.state
- Backup dataset before modification
- Restore backup on failure
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import ast


# -------------------- signal utils --------------------
def lowpass_iir_zero_phase(x, alpha):
    y_fwd = np.zeros_like(x)
    y_fwd[0] = x[0]
    for i in range(1, len(x)):
        y_fwd[i] = alpha * x[i] + (1 - alpha) * y_fwd[i - 1]

    y_bwd = np.zeros_like(x)
    y_bwd[-1] = x[-1]
    for i in range(len(x) - 2, -1, -1):
        y_bwd[i] = alpha * x[i] + (1 - alpha) * y_bwd[i + 1]

    return 0.5 * (y_fwd + y_bwd)


def schmitt_trigger(x, th_high, th_low):
    state = 0
    out = np.zeros_like(x, dtype=np.int8)
    for i, v in enumerate(x):
        if state == 0 and v > th_high:
            state = 1
        elif state == 1 and v < th_low:
            state = 0
        out[i] = state
    return out


def detect_first_rising(values, alpha, th_high, th_low):
    """Return index of first rising edge across all dims, or None."""
    rising_candidates = []

    for d in range(values.shape[1]):
        x = values[:, d]
        xf = lowpass_iir_zero_phase(x, alpha)
        diff = np.diff(xf, prepend=xf[0])
        df = lowpass_iir_zero_phase(diff, alpha)
        trig = schmitt_trigger(df, th_high, th_low)
        rising = np.where(np.diff(trig) == 1)[0] + 1
        if len(rising) > 0:
            rising_candidates.append(rising[0])

    if not rising_candidates:
        return None
    return int(min(rising_candidates))


# -------------------- dataset ops --------------------
def load_info(root: Path):
    with open(root / "meta" / "info.json", "r") as f:
        return json.load(f)


def backup_dataset(root: Path) -> Path:
    backup = root.with_name(root.name + "_backup")
    if backup.exists():
        raise RuntimeError(f"Backup already exists: {backup}")
    print(f"[Backup] Creating backup at {backup}")
    shutil.copytree(root, backup)
    return backup


def restore_backup(root: Path, backup: Path):
    print("[Restore] Restoring dataset from backup")
    shutil.rmtree(root)
    shutil.move(backup, root)


# -------------------- main logic --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dims", type=str, required=True)
    parser.add_argument("--align-seconds", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--th-high", type=float, default=0.025)
    parser.add_argument("--th-low", type=float, default=0.01)
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(",")]
    root = args.dataset_root

    backup = backup_dataset(root)

    try:
        info = load_info(root)
        fps = info["fps"]
        align_frames = int(round(args.align_seconds * fps))
        print(f"[Info] fps={fps}, align_frames={align_frames}")

        any_rising = False

        for chunk_dir in sorted((root / "data").glob("chunk-*")):
            for parquet_path in tqdm(chunk_dir.glob("episode_*.parquet"),
                                     desc=f"{chunk_dir.name}",
                                     unit="episode"):
                table = pq.read_table(parquet_path)
                df = table.to_pandas()

                # ---- read observation.state ----
                states = []
                for s in df["observation.state"]:
                    if isinstance(s, str):
                        s = ast.literal_eval(s)
                    states.append(s)
                states = np.asarray(states)

                values = states[:, dims]
                rising_idx = detect_first_rising(
                    values,
                    args.alpha,
                    args.th_high,
                    args.th_low
                )

                if rising_idx is None:
                    continue

                any_rising = True

                # ---- build align mask ----
                T = len(df)
                align = np.zeros(T, dtype=np.float32)
                end = rising_idx
                start = max(0, end - align_frames)
                align[start:end] = 1.0

                # ---- append to observation.state ----
                new_states = np.concatenate(
                    [states, align[:, None]],
                    axis=1
                )

                df["observation.state"] = list(new_states)

                pq.write_table(
                    pa.Table.from_pandas(df),
                    parquet_path
                )

        if not any_rising:
            raise RuntimeError("No rising edges detected in entire dataset.")

        # ---- update meta/features ----
        feat = info["features"]["observation.state"]
        feat["shape"][0] += 1
        feat["names"].append("hand_align_state")

        with open(root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        print("[Done] hand_align_state successfully added")
        shutil.rmtree(backup)

    except Exception as e:
        print(f"[Error] {e}")
        restore_backup(root, backup)
        sys.exit(1)


if __name__ == "__main__":
    main()
