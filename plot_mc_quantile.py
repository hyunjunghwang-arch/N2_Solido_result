#!/usr/bin/env python3
"""
Plot "Mc values of s_BL1_sampled" (x-axis) vs "Mc normal quantile" (y-axis, sigma)
for the BL1 CSV files that differ by their 'drt' value.

Only BL1 files are used (BL0 files are ignored). The drt=3u file is excluded
because its Mc normal quantile column is entirely "nan" ("verifies to
-infinity sigma"), so it contains no sigma data to plot.

All curves are drawn on a single figure.

Usage
-----
    python plot_mc_quantile.py

Requirements
------------
    pip install matplotlib numpy

Notes
-----
* Each CSV begins with 2 metadata lines, e.g.
      "10.1G generated - Verifies to +infinity sigma"
      "0 failures for output s_BL1_sampled"
  followed by a column-header line, then the numeric data.
* The position of the "Mc values of s_BL1_sampled" / "Mc normal quantile"
  columns is NOT the same in every file, so this script locates them by
  reading the header line instead of assuming fixed column indices.
* Cells are often blank (e.g. " ") and some files contain "nan" values.
  Those rows are dropped before plotting.
"""

import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
# Only BL1 files are considered. BL0 (or any other prefix) files are ignored.
FILE_GLOB = "BL1_drt*u_ff*.csv"

# drt values to exclude from the plot. The 3u file has all-nan Mc quantiles
# (Solido reports it as "-infinity sigma"), so there is nothing to plot.
EXCLUDE_DRT = {"3u"}

# Substrings used to identify the two columns of interest in the header line.
X_COL_KEY = "mc values of s_bl1_sampled"
Y_COL_KEY = "mc normal quantile"

# Number of metadata lines before the column-header line.
N_META_LINES = 2


def discover_files():
    """Find BL1 CSVs in the current directory.

    Returns a dict mapping {path: legend_label}. The drt value is parsed from
    the filename (e.g. 'BL1_drt5u_ff125.csv' -> '5u'); excluded drt values are
    skipped.
    """
    found = {}
    for path in sorted(glob.glob(FILE_GLOB)):
        m = re.search(r"_drt([0-9.]+u)_", os.path.basename(path))
        drt = m.group(1) if m else None
        if drt in EXCLUDE_DRT:
            print(f"[skip] {path}: drt={drt} is excluded")
            continue
        label = f"drt = {drt}" if drt else os.path.basename(path)
        found[path] = label
    return found


def _find_mc_columns(header_fields):
    """Return (x_index, y_index) for the *first* Mc value / Mc quantile pair.

    The header contains, in order, columns such as:
        Max-first values ..., Max-first normal quantile, ...,
        Mc values of s_BL1_sampled, Mc normal quantile, Linear-fit MC ...
    We want the raw "Mc values" and "Mc normal quantile" columns (not the
    linear-fit ones), so we match the exact-ish substrings and skip any field
    that also contains "linear-fit".
    """
    x_idx = y_idx = None
    for i, field in enumerate(header_fields):
        f = field.strip().lower()
        if "linear-fit" in f:
            continue
        if x_idx is None and X_COL_KEY in f:
            x_idx = i
        elif y_idx is None and Y_COL_KEY in f:
            y_idx = i
    return x_idx, y_idx


def load_mc_data(path):
    """Read one CSV file and return (x_values, y_values) as float arrays.

    Rows with blank or non-numeric (e.g. 'nan') entries are discarded.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))

    if len(rows) <= N_META_LINES:
        raise ValueError(f"{path}: file too short to contain data")

    header = rows[N_META_LINES]
    x_idx, y_idx = _find_mc_columns(header)
    if x_idx is None or y_idx is None:
        raise ValueError(
            f"{path}: could not locate Mc value / Mc quantile columns in header"
        )

    xs, ys = [], []
    for row in rows[N_META_LINES + 1:]:
        if len(row) <= max(x_idx, y_idx):
            continue
        x_raw = row[x_idx].strip()
        y_raw = row[y_idx].strip()
        if not x_raw or not y_raw:
            continue
        try:
            x_val = float(x_raw)
            y_val = float(y_raw)
        except ValueError:
            continue
        if np.isnan(x_val) or np.isnan(y_val):
            continue
        xs.append(x_val)
        ys.append(y_val)

    # Sort by x so the connecting line is monotonic/clean.
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def main():
    files = discover_files()
    if not files:
        print(f"No files matching {FILE_GLOB!r} found in the current directory.")
        return

    plt.figure(figsize=(8, 6))

    plotted_any = False
    for filename, label in files.items():
        try:
            x, y = load_mc_data(filename)
        except ValueError as exc:
            print(f"[skip] {exc}")
            continue

        if x.size == 0:
            print(f"[skip] {filename}: no valid (non-nan) Mc data to plot")
            continue

        plt.plot(x, y, marker="", linewidth=1.5, label=label)
        plotted_any = True
        print(f"[ok]   {filename}: plotted {x.size} points  ({label})")

    if not plotted_any:
        print("No data was plotted. Make sure the CSV files are present.")
        return

    plt.xlabel("Mc values of s_BL1_sampled")
    plt.ylabel("Mc normal quantile (sigma)")
    plt.title("MC normal quantile vs MC values of s_BL1_sampled")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_png = "mc_quantile_plot.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved figure to {out_png}")

    # Also show the window if running interactively.
    plt.show()


if __name__ == "__main__":
    main()
