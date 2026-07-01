#!/usr/bin/env python3
"""
Unified MC quantile plotter for BL0 / BL1 / both.

Examples
--------
python plot_mc_quantile_unified.py --mode bl1
python plot_mc_quantile_unified.py --mode bl0
python plot_mc_quantile_unified.py --mode both
"""

import argparse
import csv
import datetime as _dt
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# =========================
# Defaults / settings
# =========================
DEFAULT_EXCLUDE_DRT = {"3u"}
SIGMA_MARK = 6.0
VALUE_DECIMALS = 3

PLOT_TITLE = "Sense-Margin High-Sigma Verification"
PLOT_SUBTITLE = "MC Normal Quantile vs. Sampled Value"
X_LABEL = "MC values of sampled value"
Y_LABEL = "MC normal quantile  (sigma, σ)"

SIGMA_REFERENCE_LINES = [3, 4, 5, 6]
Y_MIN_SIGMA = 0
Y_MAX_SIGMA = 7

DPI = 300
N_META_LINES = 2

STYLE_CYCLE = [
    {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},
    {"color": "#d62728", "linestyle": "--", "marker": "s"},
    {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    {"color": "#9467bd", "linestyle": ":",  "marker": "D"},
    {"color": "#ff7f0e", "linestyle": "-",  "marker": "v"},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["bl0", "bl1", "both"],
        default="bl1",
        help="Which dataset(s) to plot.",
    )
    p.add_argument(
        "--exclude-drt",
        default="3u",
        help="Comma-separated drt values to exclude (e.g. '3u,4u'). Empty string disables exclusion.",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=SIGMA_MARK,
        help="Sigma level used for annotation read-off.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window.",
    )
    return p.parse_args()


def mode_to_globs(mode):
    if mode == "bl0":
        return ["BL0_drt*u_ff*.csv"]
    if mode == "bl1":
        return ["BL1_drt*u_ff*.csv"]
    return ["BL0_drt*u_ff*.csv", "BL1_drt*u_ff*.csv"]


def _extract_info(path):
    base = os.path.basename(path)
    m_bl = re.search(r"^(BL[01])_", base, re.IGNORECASE)
    m_drt = re.search(r"_drt([0-9.]+)u_", base, re.IGNORECASE)
    bl = m_bl.group(1).upper() if m_bl else "BL?"
    drt_num = float(m_drt.group(1)) if m_drt else float("inf")
    drt_str = f"{m_drt.group(1)}u" if m_drt else None
    return bl, drt_num, drt_str


def discover_files(file_globs, exclude_drt):
    found = []
    for pat in file_globs:
        for path in glob.glob(pat):
            bl, drt_num, drt_str = _extract_info(path)
            if drt_str in exclude_drt:
                print(f"[skip] {path}: drt={drt_str} excluded")
                continue
            found.append((bl, drt_num, path, drt_str))
    found.sort(key=lambda t: (t[0], t[1], t[2]))  # BL0 then BL1, by drt
    return found


def _find_mc_columns(header_fields):
    """
    Find first non-linear-fit pair:
      - x: 'Mc values of ...sampled'
      - y: 'Mc normal quantile'
    """
    x_idx = y_idx = None
    for i, field in enumerate(header_fields):
        f = field.strip().lower()
        if "linear-fit" in f:
            continue
        if x_idx is None and ("mc values of" in f) and ("sampled" in f):
            x_idx = i
        elif y_idx is None and ("mc normal quantile" in f):
            y_idx = i
    return x_idx, y_idx


def load_mc_data(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))

    if len(rows) <= N_META_LINES:
        raise ValueError(f"{path}: too short")

    header = rows[N_META_LINES]
    x_idx, y_idx = _find_mc_columns(header)
    if x_idx is None or y_idx is None:
        raise ValueError(f"{path}: Mc columns not found")

    xs, ys = [], []
    for row in rows[N_META_LINES + 1:]:
        if len(row) <= max(x_idx, y_idx):
            continue
        x_raw = row[x_idx].strip()
        y_raw = row[y_idx].strip()
        if not x_raw or not y_raw:
            continue
        try:
            xv = float(x_raw)
            yv = float(y_raw)
        except ValueError:
            continue
        if np.isnan(xv) or np.isnan(yv):
            continue
        xs.append(xv)
        ys.append(yv)

    if not xs:
        return np.array([]), np.array([])

    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def x_at_sigma(x, y, target):
    if y.size == 0 or target < y[0] or target > y[-1]:
        return None
    i = int(np.searchsorted(y, target))
    if i == 0:
        return float(x[0])
    x0, x1 = float(x[i - 1]), float(x[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return x0
    return x0 + (x1 - x0) * (target - y0) / (y1 - y0)


def _apply_rc_params():
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
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
    })


def output_names(mode):
    return (
        f"mc_quantile_plot_{mode}.png",
        f"mc_quantile_plot_{mode}.pdf",
    )


def main():
    args = parse_args()

    exclude_drt = set()
    if args.exclude_drt.strip():
        exclude_drt = {x.strip() for x in args.exclude_drt.split(",") if x.strip()}

    file_globs = mode_to_globs(args.mode)
    files = discover_files(file_globs, exclude_drt)

    if not files:
        print(f"No files found for mode={args.mode}. Expected patterns: {file_globs}")
        return

    _apply_rc_params()
    fig, ax = plt.subplots(figsize=(10, 7))

    plotted_any = False
    y_data_max = 0.0
    sigma_marks = []  # (x_mark, color, label)

    for idx, (bl, _, filename, drt_str) in enumerate(files):
        try:
            x, y = load_mc_data(filename)
        except ValueError as exc:
            print(f"[skip] {exc}")
            continue

        if x.size == 0:
            print(f"[skip] {filename}: no valid data")
            continue

        style = STYLE_CYCLE[idx % len(STYLE_CYCLE)]
        # Keep BL0/BL1 visually distinct in combined mode:
        if args.mode == "both":
            linestyle = "-" if bl == "BL0" else "--"
        else:
            linestyle = style["linestyle"]

        marker_every = max(1, x.size // 18)
        label = f"{bl}, drt={drt_str}, n={x.size:,}"

        ax.plot(
            x, y,
            color=style["color"],
            linestyle=linestyle,
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

        y_data_max = max(y_data_max, float(np.nanmax(y)))

        x_mark = x_at_sigma(x, y, args.sigma)
        if x_mark is not None:
            sigma_marks.append((x_mark, style["color"], f"{bl} {drt_str}"))
            print(f"[ok] {filename}: {bl}={x_mark:.{VALUE_DECIMALS}f} @ {args.sigma:g}σ")
        else:
            print(f"[warn] {filename}: does not reach {args.sigma:g}σ")

        plotted_any = True
        print(f"[ok] {filename}: plotted {x.size} points ({bl}, drt={drt_str})")

    if not plotted_any:
        print("No data plotted.")
        return

    # Axis limits
    bottom = Y_MIN_SIGMA
    top = Y_MAX_SIGMA if Y_MAX_SIGMA is not None else (np.ceil(y_data_max) + 0.5)
    ax.set_ylim(bottom, top)

    # Sigma guide lines
    y_lower, y_upper = ax.get_ylim()
    x_right = ax.get_xlim()[1]
    for s in SIGMA_REFERENCE_LINES:
        if y_lower <= s <= y_upper:
            ax.axhline(s, color="#888888", linewidth=0.8,
                       linestyle=(0, (4, 4)), alpha=0.6, zorder=1)
            ax.text(x_right, s, f" {s}σ", va="center", ha="left",
                    fontsize=8, color="#666666", clip_on=False)

    # Sigma annotations
    for k, (x_mark, color, lbl) in enumerate(sigma_marks):
        ax.plot([x_mark], [args.sigma], marker="o", markersize=7,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        y_off = 10 + 16 * (k % 8)
        ax.annotate(
            f"{lbl}: {x_mark:.{VALUE_DECIMALS}f} @ {args.sigma:g}σ",
            xy=(x_mark, args.sigma),
            xytext=(8, y_off),
            textcoords="offset points",
            fontsize=8.5, color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.9, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=0.9),
            zorder=6,
        )

    # Styling
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="#dddddd", zorder=0)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#eeeeee", zorder=0)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(X_LABEL, labelpad=8)
    ax.set_ylabel(Y_LABEL, labelpad=8)
    ax.set_title(PLOT_SUBTITLE, fontsize=11, color="#555555", pad=6)
    fig.suptitle(f"{PLOT_TITLE} ({args.mode.upper()})", fontsize=15, fontweight="bold", y=0.97)

    leg = ax.legend(
        title="Curve list",
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
        ncol=1,
    )
    leg.get_frame().set_linewidth(0.8)

    today = _dt.date.today().isoformat()
    caption = f"Generated {today} · mode={args.mode} · excluded drt={sorted(exclude_drt) if exclude_drt else 'none'}"
    fig.text(0.01, 0.005, caption, fontsize=7.5, color="#999999", ha="left")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    out_png, out_pdf = output_names(args.mode)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()