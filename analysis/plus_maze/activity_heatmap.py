"""
activity_heatmap.py  (Plus/Cross Maze version)
------------------------------------------------
KDE spatial density heatmap for plus maze tracking.

Draws all 4 arms as zone overlays on the heatmap:
  Bottom arm → blue   Left arm → green
  Right arm  → orange  Top arm → purple

Usage:
    python activity_heatmap.py \\
        --csv ../../data/DLCfiltered/control/TmazeMA1_1/TmazeMA1_1.csv \\
        --bottom-arm 526 606 403 717 \\
        --left-arm   259 533 346 402 \\
        --right-arm  605 892 347 402 \\
        --top-arm    526 606   0 345
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

DEFAULT_CSV     = "../../data/DLCfiltered/control/PlusMazeMA1_1/PlusMazeMA1_1.csv"
DEFAULT_OUT_DIR = "../../data/DLCfiltered"
LIKELIHOOD_THRESH = 0.6

ZONE_COLORS = {
    "bottom_arm": "#4FC3F7",
    "left_arm":   "#A5D6A7",
    "right_arm":  "#FFCC80",
    "top_arm":    "#CE93D8",
}


def load_body_center(csv_path, likelihood_thresh, jump_thresh=60.0, smooth=5):
    df  = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    x   = df["body_center"]["x"].values.astype(float)
    y   = df["body_center"]["y"].values.astype(float)
    lkh = df["body_center"]["likelihood"].values.astype(float)
    x[lkh < likelihood_thresh] = np.nan
    y[lkh < likelihood_thresh] = np.nan
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    x[np.sqrt(dx**2 + dy**2) > jump_thresh] = np.nan
    y[np.sqrt(dx**2 + dy**2) > jump_thresh] = np.nan
    nan_mask = np.isnan(x) | np.isnan(y)
    xs = pd.Series(x).rolling(smooth, center=True, min_periods=1).median().values.copy()
    ys = pd.Series(y).rolling(smooth, center=True, min_periods=1).median().values.copy()
    xs[nan_mask] = np.nan
    ys[nan_mask] = np.nan
    return xs, ys


def auto_extent(zones, pad=30):
    all_b = list(zones.values())
    return (min(b[0] for b in all_b) - pad, max(b[1] for b in all_b) + pad,
            min(b[2] for b in all_b) - pad, max(b[3] for b in all_b) + pad)


def draw_zones(ax, zones):
    for name, (x0, x1, y0, y1) in zones.items():
        color = ZONE_COLORS[name]
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.0, edgecolor=color, facecolor="none",
            linestyle="--", zorder=10,
        ))
        ax.text(x0 + 4, y0 + 16, name.replace("_", " "),
                color=color, fontsize=8, fontweight="bold", zorder=11)


def plot_kde(x, y, zones, name, out_path, cmap="inferno", sigma=15.0):
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    if len(xv) < 10:
        print("Not enough data."); return

    x_min, x_max, y_min, y_max = auto_extent(zones)
    W, H = int(x_max - x_min), int(y_max - y_min)

    hist, _, _ = np.histogram2d(xv, yv, bins=[W, H],
                                range=[[x_min, x_max], [y_min, y_max]])
    smooth_h  = gaussian_filter(hist.T, sigma=sigma)
    log_h     = np.log1p(smooth_h)
    vmax      = log_h.max()
    norm_h    = log_h / vmax if vmax > 0 else log_h

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0A0A0A")
    ax.set_facecolor("#0A0A0A")
    im = ax.imshow(norm_h, origin="upper",
                   extent=[x_min, x_max, y_max, y_min],
                   cmap=cmap, aspect="auto", interpolation="bilinear",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Relative activity (log scale)").ax.tick_params(
        colors="#AAAAAA", labelsize=9)
    draw_zones(ax, zones)
    ax.set_xlabel("X (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_ylabel("Y (pixels)", color="#CCCCCC", fontsize=12)
    ax.tick_params(colors="#666666", labelsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"KDE heatmap -> {out_path}")


def plot_histogram(x, y, zones, name, out_path, bins=40):
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    if len(xv) == 0:
        return
    x_min, x_max, y_min, y_max = auto_extent(zones)
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0A0A0A")
    ax.set_facecolor("#111111")
    h = ax.hist2d(xv, yv, bins=bins, cmap="YlOrRd", cmin=1,
                  range=[[x_min, x_max], [y_min, y_max]])
    plt.colorbar(h[3], ax=ax, label="Frame count").ax.tick_params(
        colors="#AAAAAA", labelsize=9)
    draw_zones(ax, zones)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_ylabel("Y (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_title(f"Plus Maze Histogram — {name}\n({len(xv)} frames)",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="#666666", labelsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Histogram -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plus maze KDE heatmap")
    parser.add_argument("--csv",        default=DEFAULT_CSV)
    parser.add_argument("--out-dir",    default=DEFAULT_OUT_DIR, dest="out_dir")
    parser.add_argument("--bottom-arm", nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="bottom_arm")
    parser.add_argument("--left-arm",   nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="left_arm")
    parser.add_argument("--right-arm",  nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="right_arm")
    parser.add_argument("--top-arm",    nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="top_arm")
    parser.add_argument("--likelihood", type=float, default=LIKELIHOOD_THRESH)
    parser.add_argument("--jump-thresh",type=float, default=60.0, dest="jump_thresh")
    parser.add_argument("--smooth",     type=int,   default=5)
    parser.add_argument("--cmap",       default="inferno")
    parser.add_argument("--sigma",      type=float, default=15.0)
    parser.add_argument("--bins",       type=int,   default=40)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    zones = {
        "bottom_arm": tuple(args.bottom_arm),
        "left_arm":   tuple(args.left_arm),
        "right_arm":  tuple(args.right_arm),
        "top_arm":    tuple(args.top_arm),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]

    print(f"Loading: {args.csv}")
    x, y = load_body_center(args.csv, args.likelihood, args.jump_thresh, args.smooth)
    print(f"  {(~(np.isnan(x)|np.isnan(y))).sum()} valid frames")

    plot_kde(x, y, zones, stem,
             os.path.join(args.out_dir, f"{stem}_heatmap_kde.png"),
             cmap=args.cmap, sigma=args.sigma)
    plot_histogram(x, y, zones, stem,
                   os.path.join(args.out_dir, f"{stem}_heatmap_histogram.png"),
                   bins=args.bins)


if __name__ == "__main__":
    main()
