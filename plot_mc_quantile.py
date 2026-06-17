#!/usr/bin/env python3
"""
Plot "Mc values of s_BL1_sampled" (x-axis) vs "Mc normal quantile" (y-axis, sigma)
for the BL1 CSV files that differ by their 'drt' value.

Only BL1 files are used (BL0 files are ignored). The drt=3u file is excluded
because its Mc normal quantile column is entirely "nan" ("verifies to
-infinity sigma"), so it contains no sigma data to plot.

All curves are drawn on a single, presentation-ready figure.

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
import datetime as _dt
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ===========================================================================
# User-tunable settings
# ===========================================================================
# Only BL1 files are considered. BL0 (or any other prefix) files are ignored.
FILE_GLOB = "BL1_drt*u_ff*.csv"

# drt values to exclude from the plot. The 3u file has all-nan Mc quantiles
# (Solido reports it as "-infinity sigma"), so there is nothing to plot.
EXCLUDE_DRT = {"3u"}

# Figure text.
PLOT_TITLE = "BL1 Sense-Margin High-Sigma Verification"
PLOT_SUBTITLE = "MC Normal Quantile vs. Sampled Value (s_BL1_sampled)"
X_LABEL = "MC values of s_BL1_sampled"
Y_LABEL = "MC normal quantile  (sigma, \u03c3)"

# Optional footer / branding. Set to "" to disable.
FOOTER_TEXT = ""           # e.g. "RAAAM Technologies \u2014 Confidential"

# Sigma reference lines to draw as faint horizontal guides. Set to [] to skip.
SIGMA_REFERENCE_LINES = [3, 4, 5, 6]

# y-axis upper limit in sigma. Set to None to autoscale to the data
# (data reaches ~9-10 sigma). A finite value (e.g. 6) crops to a typical
# presentation threshold.
Y_MAX_SIGMA = None

# Output files.
OUT_PNG = "mc_quantile_plot.png"
OUT_PDF = "mc_quantile_plot.pdf"
DPI = 300

# Substrings used to identify the two columns of interest in the header line.
X_COL_KEY = "mc values of s_bl1_sampled"
Y_COL_KEY = "mc normal quantile"

# Number of metadata lines before the column-header line.
N_META_LINES = 2

# Per-curve styling, applied in order of discovered files. Colors are chosen
# to remain distinguishable in grayscale; line styles/markers reinforce this.
STYLE_CYCLE = [
    {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},   # blue
    {"color": "#d62728", "linestyle": "--", "marker": "s"},   # red
    {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},   # green
    {"color": "#9467bd", "linestyle": ":",  "marker": "D"},   # purple
    {"color": "#ff7f0e", "linestyle": "-",  "marker": "v"},   # orange
]


def discover_files():
    """Find BL1 CSVs in the current directory.

    Returns a dict mapping {path: drt_string}. The drt value is parsed from
    the filename (e.g. 'BL1_drt5u_ff125.csv' -> '5u'); excluded drt values are
    skipped. Sorted numerically by drt so the legend reads in a sensible order.
    """
    found = []
    for path in glob.glob(FILE_GLOB):
        m = re.search(r"_drt([0-9.]+)u_", os.path.basename(path))
        drt_num = float(m.group(1)) if m else float("inf")
        drt_str = f"{m.group(1)}u" if m else None
        if drt_str in EXCLUDE_DRT:
            print(f"[skip] {path}: drt={drt_str} is excluded")
            continue
        found.append((drt_num, path, drt_str))
    found.sort(key=lambda t: t[0])
    return {path: drt_str for _, path, drt_str in found}


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


def _apply_rc_params():
    """Set global matplotlib styling for a clean, professional look."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.0,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
    })


def main():
    files = discover_files()
    if not files:
        print(f"No files matching {FILE_GLOB!r} found in the current directory.")
        return

    _apply_rc_params()
    fig, ax = plt.subplots(figsize=(9, 6.5))

    plotted_any = False
    y_data_max = 0.0
    for idx, (filename, drt_str) in enumerate(files.items()):
        try:
            x, y = load_mc_data(filename)
        except ValueError as exc:
            print(f"[skip] {exc}")
            continue

        if x.size == 0:
            print(f"[skip] {filename}: no valid (non-nan) Mc data to plot")
            continue

        style = STYLE_CYCLE[idx % len(STYLE_CYCLE)]
        # Show a limited number of markers so dense curves stay readable.
        marker_every = max(1, x.size // 18)

        label = f"drt = {drt_str}   (n = {x.size:,})"
        ax.plot(
            x, y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker=style["marker"],
            markevery=marker_every,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markeredgecolor=style["color"],
            label=label,
            zorder=3,
        )

        # Annotate the maximum sigma reached at the end of the curve.
        y_top = float(y[-1])
        x_top = float(x[-1])
        y_data_max = max(y_data_max, float(np.nanmax(y)))
        ax.annotate(
            f"{y_top:.2f}\u03c3",
            xy=(x_top, y_top),
            xytext=(6, 0),
            textcoords="offset points",
            va="center", ha="left",
            fontsize=9, fontweight="bold",
            color=style["color"],
            zorder=4,
        )

        plotted_any = True
        print(f"[ok]   {filename}: plotted {x.size} points  (drt = {drt_str})")

    if not plotted_any:
        print("No data was plotted. Make sure the CSV files are present.")
        return

    # ----- Axis limits -----
    if Y_MAX_SIGMA is not None:
        ax.set_ylim(top=Y_MAX_SIGMA)
    else:
        ax.set_ylim(top=np.ceil(y_data_max) + 0.5)

    # ----- Sigma reference lines -----
    y_upper = ax.get_ylim()[1]
    x_right = ax.get_xlim()[1]
    for s in SIGMA_REFERENCE_LINES:
        if s <= y_upper:
            ax.axhline(s, color="#888888", linewidth=0.8,
                       linestyle=(0, (4, 4)), alpha=0.6, zorder=1)
            ax.text(x_right, s, f" {s}\u03c3", va="center", ha="left",
                    fontsize=8, color="#666666", clip_on=False)

    # ----- Grid -----
    ax.grid(True, which="major", linestyle="-", linewidth=0.6,
            color="#dddddd", zorder=0)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
            color="#eeeeee", zorder=0)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # ----- Spines: keep left/bottom, drop top/right -----
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----- Labels & titles -----
    ax.set_xlabel(X_LABEL, labelpad=8)
    ax.set_ylabel(Y_LABEL, labelpad=8)
    ax.set_title(PLOT_SUBTITLE, fontsize=11, color="#555555", pad=6)
    fig.suptitle(PLOT_TITLE, fontsize=15, fontweight="bold", y=0.97)

    # ----- Legend -----
    leg = ax.legend(
        title="Discharge time (drt)",
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
    )
    leg.get_frame().set_linewidth(0.8)

    # ----- Footer / caption -----
    today = _dt.date.today().isoformat()
    caption = f"Generated {today} \u00b7 BL1 only \u00b7 drt=3u excluded (\u2212\u221e \u03c3)"
    fig.text(0.01, 0.005, caption, fontsize=7.5, color="#999999", ha="left")
    if FOOTER_TEXT:
        fig.text(0.99, 0.005, FOOTER_TEXT, fontsize=8, color="#999999",
                 ha="right", fontweight="bold")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")  # vector copy for slides/reports
    print(f"\nSaved figures:\n  {OUT_PNG}  (PNG, {DPI} dpi)\n  {OUT_PDF}  (vector PDF)")

    plt.show()


if __name__ == "__main__":
    main()
