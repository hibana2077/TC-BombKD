#!/usr/bin/env python3
"""Plot alpha_{videomae,timesformer,vivit} and eval_top{1,5} from CSV.

Usage examples:
  python plot_alpha_eval.py --csv epoch.csv --font-size 12 --legend-fontsize 10 --linewidth 2 --out out.png
  python plot_alpha_eval.py --csv epoch.csv --show

No title is added to the plot (per request).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot alpha and eval_top curves from CSV (no title).")
    p.add_argument("--csv", type=Path, default=Path("epoch.csv"), help="Path to CSV file")
    p.add_argument("--font-size", type=float, default=11.0, help="Base font size for labels and ticks")
    p.add_argument("--legend-fontsize", type=float, default=10.0, help="Legend font size")
    p.add_argument("--linewidth", type=float, default=1.5, help="Line width for plot lines")
    p.add_argument("--out", type=Path, default=None, help="If set, save the figure to this path instead of showing")
    p.add_argument("--show", action="store_true", help="Show the figure interactively (if not saving)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)

    # Expected columns from provided CSV
    alphas = ["alpha_videomae", "alpha_timesformer", "alpha_vivit"]
    evals = ["eval_top1", "eval_top5"]

    for col in alphas + evals:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in {args.csv}", file=sys.stderr)

    epochs = df["epoch"] if "epoch" in df.columns else pd.RangeIndex(start=1, stop=len(df) + 1)

    plt.style.use("seaborn-whitegrid")
    fig, ax_left = plt.subplots(figsize=(8, 4.5))

    # Plot alphas on left y-axis
    left_lines = []
    for name in alphas:
        if name in df.columns:
            ln, = ax_left.plot(epochs, df[name], label=name, linewidth=args.linewidth)
            left_lines.append(ln)

    ax_left.set_xlabel("epoch", fontsize=args.font_size)
    ax_left.set_ylabel("alpha", fontsize=args.font_size)
    ax_left.tick_params(axis="both", labelsize=args.font_size)

    # Plot evals on right y-axis
    ax_right = ax_left.twinx()
    right_lines = []
    for name in evals:
        if name in df.columns:
            ln, = ax_right.plot(epochs, df[name], label=name, linewidth=args.linewidth, linestyle="--")
            right_lines.append(ln)

    ax_right.set_ylabel("eval (%)", fontsize=args.font_size)
    ax_right.tick_params(axis="both", labelsize=args.font_size)

    # Combine legends from both axes
    all_lines = left_lines + right_lines
    labels = [l.get_label() for l in all_lines]
    legend = ax_left.legend(all_lines, labels, fontsize=args.legend_fontsize, loc="upper left", frameon=True)

    fig.tight_layout()

    # If no output path provided, default to a PNG next to the CSV
    out_path = args.out
    if out_path is None:
        out_path = args.csv.parent / f"{args.csv.stem}_alpha_eval.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

    # Only show interactively if explicitly requested
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
# python3 plot_alpha_eval.py --csv epoch.csv --out breakfast.png