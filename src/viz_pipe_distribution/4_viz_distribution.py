#!/usr/bin/env python3
"""
4_viz_distribution.py

Plot XY scatter of "xyz_zfixed" points from a JSONL file,
and overlay a rectangular box defined by corner1 and corner2.

Usage:
    python 4_viz_distribution.py \
        --jsonl /path/to/file.jsonl \
        --corner1 0.9 0.0 \
        --corner2 1.1 0.2 \
        --out /path/to/output.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def main(args):
    jsonl_path = args.jsonl
    corner1 = args.corner1
    corner2 = args.corner2
    out_path = args.out

    xs = []
    ys = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            x, y = data["xyz_zfixed"][:2]
            xs.append(x)
            ys.append(y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ys, xs, s=10, c="blue", alpha=0.6, label="xyz_zfixed points")

    y1, x1 = corner1[1], corner1[0]
    y2, x2 = corner2[1], corner2[0]
    rect_x = [y1, y1, y2, y2, y1]
    rect_y = [x1, x2, x2, x1, x1]
    ax.plot(rect_x, rect_y, c="red", lw=2, label="corner box")

    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.set_title("XY Scatter of xyz_zfixed with Corner Box")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(facecolor='white', framealpha=0.6, loc='upper right')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved scatter plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--corner1", type=float, nargs=2, required=True)
    parser.add_argument("--corner2", type=float, nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args)
