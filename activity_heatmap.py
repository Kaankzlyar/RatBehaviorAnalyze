"""
activity_heatmap.py
-------------------
Spatial density heatmap showing where the animal spent most time.

Uses body_center tracking to create a 2D histogram of activity
across the arena coordinate plane.

Usage:
    python activity_heatmap.py --arena 396 776 153 530 --inner-zone 422 747 177 502
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULT_CSV       = "data/DLCfiltered/OpenFieldMA1_2.csv"
DEFAULT_OUT_DIR   = "data/DLCfiltered"
LIKELIHOOD_THRESH = 0.6


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


def load_body_center(csv_path: str, likelihood_thresh: float,
                     arena: tuple = None, jump_thresh: float = 60.0,
                     smooth_window: int = 5) -> tuple:
    """Load and filter body_center coordinates."""
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)

    x   = df["body_center"]["x"].values.astype(float)
    y   = df["body_center"]["y"].values.astype(float)
    lkh = df["body_center"]["likelihood"].values.astype(float)

    # Step 1: likelihood filter
    x[lkh < likelihood_thresh] = np.nan
    y[lkh < likelihood_thresh] = np.nan

    # Step 2: arena bounds filter
    if arena is not None:
        x, y = apply_arena_filter(x, y, arena)

    # Step 3: jump threshold
    x, y = apply_jump_threshold(x, y, jump_thresh)

    # Step 4: rolling median smoothing
    x, y = apply_rolling_median(x, y, smooth_window)

    return x, y


def draw_arena_zones(ax, arena, inner_zone, arena_color="#FFFFFF", zone_color="#FF8800"):
    """Draw arena and inner zone boundaries."""
    x_min, x_max, y_min, y_max = arena
    ix_min, ix_max, iy_min, iy_max = inner_zone

    # arena boundary
    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2.0, edgecolor=arena_color,
        facecolor="none", zorder=10, linestyle="--",
    ))
    # inner zone boundary
    ax.add_patch(mpatches.Rectangle(
        (ix_min, iy_min), ix_max - ix_min, iy_max - iy_min,
        linewidth=1.5, edgecolor=zone_color,
        facecolor="none", zorder=10, linestyle=":",
    ))


def plot_2d_histogram(x, y, arena, inner_zone, video_name: str, out_path: str,
                      bins: int = 40, cmap: str = "YlOrRd"):
    """Create 2D histogram heatmap of body_center positions."""
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]

    if len(xv) == 0:
        print("No valid data for heatmap.")
        return

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0A0A0A")
    ax.set_facecolor("#111111")

    # 2D histogram
    h = ax.hist2d(xv, yv, bins=bins, cmap=cmap, cmin=1)
    cbar = plt.colorbar(h[3], ax=ax, label="Frame count")
    cbar.ax.tick_params(colors="#AAAAAA", labelsize=9)

    # Arena zones
    draw_arena_zones(ax, arena, inner_zone)

    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_ylabel("Y (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_title(
        f"Activity Heatmap — {video_name}\n"
        f"(body_center density, {len(xv)} valid frames)",
        color="white", fontsize=13, pad=10,
    )
    ax.tick_params(colors="#666666", labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Heatmap saved → {out_path}")


def plot_kde_heatmap(x, y, arena, inner_zone, video_name: str, out_path: str,
                     cmap: str = "hot"):
    """Create smooth KDE (kernel density estimate) heatmap."""
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]

    if len(xv) < 10:
        print("Not enough valid data for KDE heatmap.")
        return

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0A0A0A")
    ax.set_facecolor("#111111")

    # Create KDE
    xy = np.vstack([xv, yv])
    kde = gaussian_kde(xy)

    # Grid for KDE evaluation
    x_min, x_max, y_min, y_max = arena
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    z = kde(positions).reshape(xx.shape)

    # Plot KDE
    h = ax.contourf(xx, yy, z, levels=20, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(h, ax=ax, label="Density")
    cbar.ax.tick_params(colors="#AAAAAA", labelsize=9)

    # Arena zones
    draw_arena_zones(ax, arena, inner_zone)

    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_ylabel("Y (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_title(
        f"Activity Density (KDE) — {video_name}\n"
        f"(body_center kernel density estimate, {len(xv)} frames)",
        color="white", fontsize=13, pad=10,
    )
    ax.tick_params(colors="#666666", labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"KDE heatmap saved → {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Activity density heatmap from body_center")
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
    parser.add_argument("--bins",        default=40,   type=int,    help="2D histogram bin count (default 40)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]

    # Setup arena/inner_zone (required for both heatmaps)
    if not args.arena or not args.inner_zone:
        raise ValueError("Both --arena and --inner-zone are required for heatmap visualization.")

    arena = tuple(args.arena)
    inner_zone = tuple(args.inner_zone)

    print(f"Loading: {args.csv}  (likelihood >= {args.likelihood})")
    print(f"  jump_thresh={args.jump_thresh}px  smooth_window={args.smooth}")
    x, y = load_body_center(args.csv, args.likelihood,
                            arena=arena,
                            jump_thresh=args.jump_thresh,
                            smooth_window=args.smooth)

    valid = ~(np.isnan(x) | np.isnan(y))
    print(f"  {valid.sum()} valid frames")

    print(f"\nArena bounds: X {arena[0]:.0f}–{arena[1]:.0f}  Y {arena[2]:.0f}–{arena[3]:.0f}")
    print(f"Inner zone:   X {inner_zone[0]:.0f}–{inner_zone[1]:.0f}  Y {inner_zone[2]:.0f}–{inner_zone[3]:.0f}")

    print("\nRendering heatmaps...")
    plot_2d_histogram(
        x, y, arena, inner_zone, stem,
        os.path.join(args.out_dir, f"{stem}_heatmap_histogram.png"),
        bins=args.bins
    )
    plot_kde_heatmap(
        x, y, arena, inner_zone, stem,
        os.path.join(args.out_dir, f"{stem}_heatmap_kde.png"),
    )


if __name__ == "__main__":
    main()
