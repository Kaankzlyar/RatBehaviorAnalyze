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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULT_CSV       = "../data/DLCfiltered/OpenFieldMA1_2.csv"
DEFAULT_OUT_DIR   = "../data/DLCfiltered"
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
        linewidth=2.5, edgecolor=arena_color,
        facecolor="none", zorder=10, linestyle="--",
    ))
    # inner zone boundary
    ax.add_patch(mpatches.Rectangle(
        (ix_min, iy_min), ix_max - ix_min, iy_max - iy_min,
        linewidth=2.0, edgecolor=zone_color,
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


def _make_football_cmap() -> mcolors.LinearSegmentedColormap:
    """Custom RGBA colormap: fully transparent → pale yellow → orange → red → dark red.

    Alpha=0 at density=0 ensures background shows through cleanly.
    No pale fog across the entire arena — only visited regions get color.
    """
    # (R, G, B, A) at positions 0.0 → 1.0
    rgba = [
        (1.00, 1.00, 1.00, 0.00),  # 0%   — fully transparent (background shows through)
        (1.00, 1.00, 0.55, 0.20),  # 15%  — barely-there pale yellow
        (1.00, 0.90, 0.10, 0.65),  # 40%  — yellow, mostly visible
        (1.00, 0.55, 0.00, 0.88),  # 65%  — orange
        (0.90, 0.10, 0.00, 0.96),  # 85%  — red
        (0.50, 0.00, 0.00, 1.00),  # 100% — dark red, fully opaque
    ]
    return mcolors.LinearSegmentedColormap.from_list("football", rgba)


def plot_kde_heatmap(x, y, arena, inner_zone, video_name: str, out_path: str,
                     sigma: float = 10.0, percentile: float = 75.0, **_):
    """Football-style activity heatmap.

    Pipeline:
      1. Per-pixel histogram (arena resolution, no grid artifacts)
      2. Moderate gaussian blur (sigma px)
      3. Percentile clipping — bottom p% of non-zero density → 0 (invisible)
      4. Custom RGBA colormap: alpha=0 at 0 → pale yellow → orange → dark red
      5. Transparent imshow — background shows through unvisited cells

    Args:
        sigma:      Gaussian blur radius in pixels (default 10; lower=sharper)
        percentile: Suppress the bottom N% of non-zero density values (default 75)
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]

    if len(xv) < 10:
        print("Not enough valid data for heatmap.")
        return

    x_min, x_max, y_min, y_max = arena
    W = int(x_max - x_min)
    H = int(y_max - y_min)

    # ── 1. Per-pixel visit count ──────────────────────────────────────────────
    hist, _, _ = np.histogram2d(
        xv, yv, bins=[W, H],
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # ── 2. Moderate gaussian blur ─────────────────────────────────────────────
    density = gaussian_filter(hist.T, sigma=sigma)

    # ── 3. Percentile clipping — suppress weak / noise density ───────────────
    nonzero = density[density > 0]
    if len(nonzero):
        threshold = np.percentile(nonzero, percentile)
        density = np.clip(density - threshold, 0, None)

    # ── 4. Normalize 0 → 1 ───────────────────────────────────────────────────
    vmax = density.max()
    density_norm = density / vmax if vmax > 0 else density

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    BG = "#F5F5F2"   # warm off-white, like a football pitch scan
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    ax.set_facecolor(BG)

    # Arena background fill
    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), W, H,
        facecolor=BG, edgecolor="none", zorder=0,
    ))

    cmap = _make_football_cmap()
    ax.imshow(
        density_norm,
        origin="upper",
        extent=[x_min, x_max, y_max, y_min],
        cmap=cmap,
        aspect="auto",
        interpolation="bilinear",
        vmin=0, vmax=1,
        zorder=1,
    )

    # Subtle boundary overlays
    draw_arena_zones(ax, arena, inner_zone,
                     arena_color="#888888", zone_color="#BBBBBB")

    # Colorbar using the football cmap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Activity intensity", color="#333333", fontsize=10)
    cbar.ax.tick_params(colors="#555555", labelsize=8)

    ax.set_xlabel("X (pixels)", color="#333333", fontsize=12)
    ax.set_ylabel("Y (pixels)", color="#333333", fontsize=12)
    ax.set_title(
        f"Activity Heatmap — {video_name}\n"
        f"body_center  |  {len(xv)} frames  |  σ={sigma}  |  clip={percentile:.0f}th pct",
        color="#111111", fontsize=13, pad=10,
    )
    ax.tick_params(colors="#555555", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#DDDDDD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close()
    print(f"Heatmap saved → {out_path}")


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
    parser.add_argument("--sigma",       default=10.0, type=float,  help="Gaussian blur radius in pixels (default: 10)")
    parser.add_argument("--percentile",  default=75.0, type=float,  help="Suppress bottom N%% of non-zero density (default: 75)")
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
        sigma=args.sigma,
        percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
