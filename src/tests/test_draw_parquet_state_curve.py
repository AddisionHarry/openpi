#!/usr/bin/env python3
"""
test_draw_parquet_state_curve.py

Draw value-vs-time curves for selected indices of a vector field
(e.g. "observation.state") stored in a parquet file.

Example:
    python test_draw_parquet_state_curve.py \
        --parquet-path <path/to/data.parquet> \
        --field "observation.state" \
        --draw-indices "[1,2,3]" \
        --out-path <path/to/state_curve.png>
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Draw curves from parquet vector field")
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Input parquet file path",
    )
    parser.add_argument(
        "--field",
        type=str,
        required=True,
        help='Field name, e.g. "observation.state"',
    )
    parser.add_argument(
        "--draw-indices",
        type=str,
        required=True,
        help='Indices to draw, e.g. "[1,2,3]"',
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output image path (png/pdf)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    parquet_path = Path(args.parquet_path)
    out_path = Path(args.out_path)

    # ---- parse indices safely ----
    try:
        draw_indices = ast.literal_eval(args.draw_indices)
        assert isinstance(draw_indices, (list, tuple))
        draw_indices = [int(i) for i in draw_indices]
    except Exception as e:
        raise ValueError(
            f"Invalid --draw-indices format: {args.draw_indices}"
        ) from e

    # ---- load parquet ----
    df = pd.read_parquet(parquet_path)

    if args.field not in df.columns:
        raise KeyError(
            f'Field "{args.field}" not found in parquet columns: {list(df.columns)}'
        )

    series = df[args.field]

    # ---- basic validation ----
    first_val = series.iloc[0]
    if not isinstance(first_val, (list, np.ndarray)):
        raise TypeError(
            f'Field "{args.field}" is not a list/array per row'
        )

    dim = len(first_val)
    for idx in draw_indices:
        if idx < 0 or idx >= dim:
            raise IndexError(
                f"Index {idx} out of range for field dimension {dim}"
            )

    # ---- stack into (T, D) ----
    values = np.stack(series.to_numpy(), axis=0)  # (T, D)
    T = values.shape[0]
    t = np.arange(T)

    # ---- plot ----
    plt.figure(figsize=(10, 4 + 1.5 * len(draw_indices)))

    for idx in draw_indices:
        plt.plot(
            t,
            values[:, idx],
            linewidth=1.8,
            label=f"{args.field}[{idx}]",
        )

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title(f"Time Series of {args.field} (selected indices)")
    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"[Done] Saved figure to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
