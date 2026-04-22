"""
orbit_plot.py
-------------
Creates per-bodypart orbit visualizations from a DLC filtered CSV.

Outputs two files:
  1. <name>_orbit_grid.png  — all body parts as individual subplots in a grid
  2. <name>_thigmotaxis.png — body_center only, with arena + thigmotaxis zone overlay

Arena edges and inner zone can be set manually (recommended) using
show_frame_coords.py, or auto-detected from the data.

Usage:
    python orbit_plot.py
    python orbit_plot.py --csv data/DLCfiltered/OpenFieldMA1_2.csv
    python orbit_plot.py --csv data/DLCfiltered/OpenFieldMA1_2.csv \\
        --arena 397 775 158 532 \\
        --inner-zone 460 710 220 470
        
        PYTHONIOENCODING=utf-8 python orbit_plot.py --arena 396 776 153 530 --inner-zone 422 747 177 502
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULT_CSV       = "data/DLCfiltered/OpenFieldMA1_2.csv"
DEFAULT_OUT_DIR   = "data/DLCfiltered"
LIKELIHOOD_THRESH = 0.6
THIGMO_MARGIN     = 0.20   # outer 20% of arena width/height = thigmotaxis zone

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

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

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
    # Re-apply NaN mask from original (smoothing shouldn't fill real gaps)
    x_s[nan_mask] = np.nan
    y_s[nan_mask] = np.nan
    return x_s, y_s


def load_dlc_csv(csv_path: str, likelihood_thresh: float,
                 arena: tuple = None, jump_thresh: float = 60.0,
                 smooth_window: int = 5) -> dict[str, dict]:
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(0).unique().tolist()
    tracking = {}
    for bp in bodyparts:
        x   = df[bp]["x"].values.astype(float)
        y   = df[bp]["y"].values.astype(float)
        lkh = df[bp]["likelihood"].values.astype(float)

        # Step 1: likelihood filter
        x[lkh < likelihood_thresh] = np.nan
        y[lkh < likelihood_thresh] = np.nan

        # Step 2: arena bounds filter (set outside to NaN)
        if arena is not None:
            x, y = apply_arena_filter(x, y, arena)

        # Step 3: jump threshold (temporal consistency)
        x, y = apply_jump_threshold(x, y, jump_thresh)

        # Step 4: rolling median smoothing
        x, y = apply_rolling_median(x, y, smooth_window)

        pct = np.sum(~np.isnan(x)) / len(x) * 100
        print(f"  {bp:>15s}: {pct:.1f}% frames kept")
        tracking[bp] = {"x": x, "y": y}
    return tracking


# ─── ARENA & THIGMOTAXIS ──────────────────────────────────────────────────────

def detect_arena(tracking: dict, percentile: float = 0.5) -> tuple:
    """
    Estimate arena bounds from all valid positions combined.
    Returns (x_min, x_max, y_min, y_max).
    """
    all_x, all_y = [], []
    for data in tracking.values():
        valid = ~(np.isnan(data["x"]) | np.isnan(data["y"]))
        all_x.append(data["x"][valid])
        all_y.append(data["y"][valid])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    return (
        np.percentile(all_x, percentile),
        np.percentile(all_x, 100 - percentile),
        np.percentile(all_y, percentile),
        np.percentile(all_y, 100 - percentile),
    )


def compute_inner_zone(arena: tuple, margin: float) -> tuple:
    """Compute inner zone bounds from arena + margin fraction."""
    x_min, x_max, y_min, y_max = arena
    w = x_max - x_min
    h = y_max - y_min
    return (x_min + margin * w, x_max - margin * w,
            y_min + margin * h, y_max - margin * h)


def thigmotaxis_rate(x: np.ndarray, y: np.ndarray, inner_zone: tuple) -> float:
    """
    Fraction of valid frames where position is OUTSIDE the inner zone
    (i.e. in the border/thigmotaxis zone).
    inner_zone = (ix_min, ix_max, iy_min, iy_max)
    """
    ix_min, ix_max, iy_min, iy_max = inner_zone

    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() == 0:
        return float("nan")

    xv, yv = x[valid], y[valid]
    in_border = (xv < ix_min) | (xv > ix_max) | (yv < iy_min) | (yv > iy_max)
    return float(in_border.sum() / len(xv))


# ─── COLORMAP ─────────────────────────────────────────────────────────────────

def make_temporal_cmap(base_hex: str) -> mcolors.LinearSegmentedColormap:
    base = mcolors.to_rgb(base_hex)
    dark  = tuple(max(0.0, c * 0.4) for c in base)
    light = tuple(min(1.0, 0.2 + c * 0.5) for c in base)
    return mcolors.LinearSegmentedColormap.from_list("t", [light, base, dark])


def draw_trajectory(ax, x, y, color_hex, linewidth=0.9, alpha=0.75):
    """Draw a single trajectory with temporal color fade on the given axes."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 5:
        return
    xi = x[valid]
    yi = y[valid]
    t  = np.where(valid)[0]
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1)
    cmap = make_temporal_cmap(color_hex)
    for i in range(len(xi) - 1):
        ax.plot(
            [xi[i], xi[i + 1]], [yi[i], yi[i + 1]],
            color=cmap(t_norm[i]),
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )
    # start (circle) / end (diamond)
    ax.scatter(xi[0],  yi[0],  color=cmap(0.15), s=50, zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.scatter(xi[-1], yi[-1], color=cmap(0.85), s=35, zorder=5,
               edgecolors="white", linewidths=0.5, marker="D")


def style_ax(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="#AAAAAA", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def draw_arena_zones(ax, arena, inner_zone, arena_color="#FFFFFF", zone_color="#FF8800"):
    """Draw the arena rectangle and inner thigmotaxis boundary."""
    x_min, x_max, y_min, y_max = arena
    ix_min, ix_max, iy_min, iy_max = inner_zone

    # outer arena boundary
    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1.5, edgecolor=arena_color,
        facecolor="none", zorder=10, linestyle="--",
    ))
    # inner (centre) zone boundary
    ax.add_patch(mpatches.Rectangle(
        (ix_min, iy_min), ix_max - ix_min, iy_max - iy_min,
        linewidth=1.0, edgecolor=zone_color,
        facecolor="none", zorder=10, linestyle=":",
    ))


# ─── PLOT 1: GRID OF ALL BODY PARTS ───────────────────────────────────────────

def plot_grid(tracking: dict, arena: tuple, inner_zone: tuple,
              video_name: str, out_path: str) -> None:
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
        color = BODYPART_COLORS.get(bp, "#FFFFFF")

        draw_trajectory(ax, x, y, color)
        draw_arena_zones(ax, arena, inner_zone)

        ax.invert_yaxis()
        ax.set_title(
            bp,
            color=BODYPART_COLORS.get(bp, "#FFFFFF"),
            fontsize=9, pad=4,
        )
        ax.set_xlabel("X (px)", color="#666666", fontsize=7)
        ax.set_ylabel("Y (px)", color="#666666", fontsize=7)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Per-Bodypart Orbit — {video_name}\n"
        f"(dashed white = arena wall  |  dotted orange = manually selected inner boundary)\n"
        f"circle = start  |  diamond = end  |  pale→dark = early→late",
        color="#DDDDDD", fontsize=11, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Grid saved  → {out_path}")


# ─── PLOT 2: BODY_CENTER THIGMOTAXIS DETAIL ───────────────────────────────────

def plot_thigmotaxis_detail(tracking: dict, arena: tuple, inner_zone: tuple,
                            video_name: str, out_path: str) -> None:
    bp = "body_center" if "body_center" in tracking else list(tracking.keys())[0]
    x = tracking[bp]["x"]
    y = tracking[bp]["y"]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0A0A0A")
    style_ax(ax)

    x_min, x_max, y_min, y_max = arena
    ix_min, ix_max, iy_min, iy_max = inner_zone

    # Shade the thigmotaxis border zone in translucent orange
    outer = mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor="#FF8800", alpha=0.12, zorder=1,
    )
    inner = mpatches.Rectangle(
        (ix_min, iy_min), ix_max - ix_min, iy_max - iy_min,
        facecolor="#0A0A0A", alpha=1.0, zorder=2,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)

    draw_trajectory(ax, x, y, BODYPART_COLORS.get(bp, "#0074D9"), linewidth=1.1)
    draw_arena_zones(ax, arena, inner_zone)

    rate = thigmotaxis_rate(x, y, inner_zone)
    rate_str = f"{rate * 100:.1f}%" if not np.isnan(rate) else "N/A"

    # Colour-code each dot as border (orange) vs centre (blue)
    valid = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[valid], y[valid]
    in_border = (xv < ix_min) | (xv > ix_max) | (yv < iy_min) | (yv > iy_max)
    ax.scatter(xv[in_border],  yv[in_border],  color="#FF8800",
               s=4, alpha=0.5, zorder=3, label="border zone")
    ax.scatter(xv[~in_border], yv[~in_border], color="#4488FF",
               s=4, alpha=0.5, zorder=3, label="centre zone")

    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)", color="#CCCCCC", fontsize=11)
    ax.set_ylabel("Y (pixels)", color="#CCCCCC", fontsize=11)
    ax.set_title(
        f"Thigmotaxis Detail — {video_name}  [{bp}]\n"
        f"Thigmotaxis rate: {rate_str}  (orange = border zone, blue = centre zone)",
        color="white", fontsize=12, pad=10,
    )
    ax.legend(loc="lower right", framealpha=0.3, facecolor="#222222",
              edgecolor="#555555", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Thigmo saved → {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DLC orbit + thigmotaxis plot")
    parser.add_argument("--csv",         default=DEFAULT_CSV,       help="Path to DLC filtered CSV")
    parser.add_argument("--likelihood",  default=LIKELIHOOD_THRESH, type=float)
    parser.add_argument("--margin",      default=THIGMO_MARGIN,     type=float,
                        help="Fallback border fraction if --inner-zone not given (default 0.20)")
    parser.add_argument("--arena",       nargs=4, type=float,       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Manual arena wall bounds in pixels (from show_frame_coords.py)")
    parser.add_argument("--inner-zone",  nargs=4, type=float,       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        dest="inner_zone",
                        help="Manual inner zone bounds in pixels (from show_frame_coords.py)")
    parser.add_argument("--out-dir",      default=DEFAULT_OUT_DIR,   dest="out_dir")
    parser.add_argument("--jump-thresh",  default=60.0, type=float,  dest="jump_thresh",
                        help="Max px between consecutive frames before marking as outlier (default 60)")
    parser.add_argument("--smooth",       default=5,    type=int,
                        help="Rolling median window size for smoothing (default 5)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]

    # Determine arena early so load_dlc_csv can apply arena filter
    pre_arena = tuple(args.arena) if args.arena else None

    print(f"Loading: {args.csv}  (likelihood >= {args.likelihood})")
    print(f"  jump_thresh={args.jump_thresh}px  smooth_window={args.smooth}")
    tracking = load_dlc_csv(args.csv, args.likelihood,
                            arena=pre_arena,
                            jump_thresh=args.jump_thresh,
                            smooth_window=args.smooth)

    # ── Arena bounds ─────────────────────────────────────────────────────────
    if args.arena:
        arena = tuple(args.arena)
        print(f"\nArena bounds (manual):        X {arena[0]:.0f}–{arena[1]:.0f}  Y {arena[2]:.0f}–{arena[3]:.0f}")
    else:
        arena = detect_arena(tracking)
        print(f"\nArena bounds (auto-detected): X {arena[0]:.0f}–{arena[1]:.0f}  Y {arena[2]:.0f}–{arena[3]:.0f}")

    # ── Inner zone ───────────────────────────────────────────────────────────
    if args.inner_zone:
        inner_zone = tuple(args.inner_zone)
        print(f"Inner zone   (manual):        X {inner_zone[0]:.0f}–{inner_zone[1]:.0f}  Y {inner_zone[2]:.0f}–{inner_zone[3]:.0f}")
    else:
        inner_zone = compute_inner_zone(arena, args.margin)
        print(f"Inner zone   (auto {args.margin*100:.0f}% margin): X {inner_zone[0]:.0f}–{inner_zone[1]:.0f}  Y {inner_zone[2]:.0f}–{inner_zone[3]:.0f}")

    # ── Thigmotaxis summary (reference: body_center only) ───────────────────
    REF_BP = "body_center"
    if REF_BP not in tracking:
        REF_BP = list(tracking.keys())[0]
    ref_rate = thigmotaxis_rate(tracking[REF_BP]["x"], tracking[REF_BP]["y"], inner_zone)
    bar = "█" * int(ref_rate * 20) if not np.isnan(ref_rate) else ""
    print(f"\nThigmotaxis rate [{REF_BP}]: {ref_rate*100:.1f}%  {bar}")

    print("\nRendering plots...")
    plot_grid(
        tracking, arena, inner_zone, stem,
        os.path.join(args.out_dir, f"{stem}_orbit_grid.png"),
    )
    plot_thigmotaxis_detail(
        tracking, arena, inner_zone, stem,
        os.path.join(args.out_dir, f"{stem}_thigmotaxis.png"),
    )


if __name__ == "__main__":
    main()
