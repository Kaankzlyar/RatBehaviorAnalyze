"""
bodypart_heatmaps.py
--------------------
Per-bodypart activity density heatmaps in grid layout.

Creates a grid of heatmaps, one for each tracked body part,
showing spatial density of movement across the arena.

Usage:
    python bodypart_heatmaps.py --arena 396 776 153 530 --inner-zone 422 747 177 502
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# --- DEFAULTS -----------------------------------------------------------------

DEFAULT_CSV       = "../../data/DLCfiltered/OpenFieldMA1_2.csv"
DEFAULT_OUT_DIR   = "../../data/DLCfiltered"
LIKELIHOOD_THRESH = 0.6

BODYPART_COLORS = {
    "nose":          "#FF4136",
    "head":          "#FF851B",
    "neck":          "#FFDC00",
    "left_ear":      "#2ECC40",
    "right_ear":     "#01FF70",
    "body_center":   "#0074D9",
    "left_forepaw":  "#7FDBFF",
    "right_forepaw": "#B10DC9",
    "left_hindpaw":  "#F012BE",
    "right_hindpaw": "#FF69B4",
    "tail_base":     "#AAAAAA",
}


def apply_arena_filter(x: np.ndarray, y: np.ndarray, arena: tuple) -> tuple:
    """Set points outside arena bounds to NaN."""
    x, y = x.copy(), y.copy()
    if arena is None:
        return x, y
    x_min, x_max, y_min, y_max = arena
    outside = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)
    x[outside] = np.nan
    y[outside] = np.nan
    return x, y


def apply_jump_threshold(x: np.ndarray, y: np.ndarray, threshold_px: float = 60.0) -> tuple:
    """Set points to NaN where consecutive frame distance exceeds threshold."""
    x, y = x.copy(), y.copy()
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    dist = np.sqrt(dx**2 + dy**2)
    jumps = dist > threshold_px
    x[jumps] = np.nan
    y[jumps] = np.nan
    return x, y


def apply_rolling_median(x: np.ndarray, y: np.ndarray, window: int = 5) -> tuple:
    """Apply rolling median smoothing, preserving NaN gaps."""
    nan_mask = np.isnan(x) | np.isnan(y)
    x_s = pd.Series(x).rolling(window, center=True, min_periods=1).median().values.copy()
    y_s = pd.Series(y).rolling(window, center=True, min_periods=1).median().values.copy()
    x_s[nan_mask] = np.nan
    y_s[nan_mask] = np.nan
    return x_s, y_s


def load_dlc_csv(csv_path: str, likelihood_thresh: float,
                 arena: tuple = None, jump_thresh: float = 60.0,
                 smooth_window: int = 5) -> dict[str, dict]:
    """Load and filter all body parts."""
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(0).unique().tolist()
    tracking = {}
    for bp in bodyparts:
        x   = df[bp]["x"].values.astype(float)
        y   = df[bp]["y"].values.astype(float)
        lkh = df[bp]["likelihood"].values.astype(float)

        x[lkh < likelihood_thresh] = np.nan
        y[lkh < likelihood_thresh] = np.nan

        if arena is not None:
            x, y = apply_arena_filter(x, y, arena)

        x, y = apply_jump_threshold(x, y, jump_thresh)
        x, y = apply_rolling_median(x, y, smooth_window)

        pct = np.sum(~np.isnan(x)) / len(x) * 100
        print(f"  {bp:>15s}: {pct:.1f}% frames kept")
        tracking[bp] = {"x": x, "y": y}
    return tracking


def draw_arena_zones(ax, arena, inner_zone, arena_color="#FFFFFF", zone_color="#FF8800"):
    """Draw arena and inner zone boundaries."""
    x_min, x_max, y_min, y_max = arena
    ix_min, ix_max, iy_min, iy_max = inner_zone

    # arena boundary
    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1.0, edgecolor=arena_color,
        facecolor="none", zorder=10, linestyle="--",
    ))
    # inner zone boundary
    ax.add_patch(mpatches.Rectangle(
        (ix_min, iy_min), ix_max - ix_min, iy_max - iy_min,
        linewidth=0.8, edgecolor=zone_color,
        facecolor="none", zorder=10, linestyle=":",
    ))


def style_ax(ax):
    """Style axis for dark theme."""
    ax.set_facecolor("#111111")
    ax.tick_params(colors="#666666", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def plot_bodypart_heatmaps_grid(tracking: dict, arena: tuple, inner_zone: tuple,
                                video_name: str, out_path: str, bins: int = 30):
    """Create grid of per-bodypart 2D histogram heatmaps."""
    bodyparts = list(tracking.keys())
    n = len(bodyparts)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 4),
        facecolor="#0A0A0A",
    )
    axes = axes.flatten()

    for idx, bp in enumerate(bodyparts):
        ax = axes[idx]
        style_ax(ax)

        x = tracking[bp]["x"]
        y = tracking[bp]["y"]
        valid = ~(np.isnan(x) | np.isnan(y))

        if valid.sum() < 5:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   color="#AAAAAA", transform=ax.transAxes)
            ax.set_title(bp, color=BODYPART_COLORS.get(bp, "#FFFFFF"), fontsize=9, pad=4)
            continue

        xv, yv = x[valid], y[valid]

        # 2D histogram
        h = ax.hist2d(xv, yv, bins=bins, cmap="YlOrRd", cmin=1)

        # Arena zones
        draw_arena_zones(ax, arena, inner_zone)

        ax.invert_yaxis()
        ax.set_title(
            f"{bp}\n({len(xv)} frames)",
            color=BODYPART_COLORS.get(bp, "#FFFFFF"),
            fontsize=8, pad=3,
        )
        ax.set_xlabel("X (px)", color="#666666", fontsize=6)
        ax.set_ylabel("Y (px)", color="#666666", fontsize=6)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Per-Bodypart Activity Heatmaps â€” {video_name}\n"
        f"(2D histogram density, dashed white = arena wall, dotted orange = inner boundary)",
        color="#DDDDDD", fontsize=11, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Bodypart heatmap grid saved -> {out_path}")


# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-bodypart activity heatmap grid")
    parser.add_argument("--csv",         default=DEFAULT_CSV,       help="Path to DLC filtered CSV")
    parser.add_argument("--likelihood",  default=LIKELIHOOD_THRESH, type=float)
    parser.add_argument("--arena",       nargs=4, type=float,       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Manual arena wall bounds in pixels")
    parser.add_argument("--inner-zone",  nargs=4, type=float,       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        dest="inner_zone",
                        help="Manual inner zone bounds in pixels")
    parser.add_argument("--jump-thresh", default=60.0, type=float, dest="jump_thresh")
    parser.add_argument("--smooth",      default=5,    type=int)
    parser.add_argument("--out-dir",     default=DEFAULT_OUT_DIR,   dest="out_dir")
    parser.add_argument("--bins",        default=30,   type=int,    help="Histogram bin count (default 30)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]

    if not args.arena or not args.inner_zone:
        raise ValueError("Both --arena and --inner-zone are required.")

    arena = tuple(args.arena)
    inner_zone = tuple(args.inner_zone)

    print(f"Loading: {args.csv}  (likelihood >= {args.likelihood})")
    print(f"  jump_thresh={args.jump_thresh}px  smooth_window={args.smooth}")
    tracking = load_dlc_csv(args.csv, args.likelihood,
                            arena=arena,
                            jump_thresh=args.jump_thresh,
                            smooth_window=args.smooth)

    print(f"\nArena bounds: X {arena[0]:.0f}-{arena[1]:.0f}  Y {arena[2]:.0f}-{arena[3]:.0f}")
    print(f"Inner zone:   X {inner_zone[0]:.0f}-{inner_zone[1]:.0f}  Y {inner_zone[2]:.0f}-{inner_zone[3]:.0f}")

    print("\nRendering per-bodypart heatmap grid...")
    plot_bodypart_heatmaps_grid(
        tracking, arena, inner_zone, stem,
        os.path.join(args.out_dir, f"{stem}_bodypart_heatmaps.png"),
        bins=args.bins
    )


if __name__ == "__main__":
    main()

