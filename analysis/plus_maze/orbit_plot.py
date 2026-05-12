"""
orbit_plot.py  (Plus/Cross Maze version)
-----------------------------------------
Trajectory visualization for plus maze DLC tracking.

Outputs:
  <name>_tmaze_orbit.png    — body_center trajectory, colored by zone
  <name>_tmaze_bodyparts.png — 4-panel grid: nose/head/body_center/tail_base

Zone colors:
  Bottom → blue (#4FC3F7)   Left  → green  (#66BB6A)
  Right  → orange (#FFA726) Top   → purple (#BA68C8)
  Junction → grey  (#888888)

Usage:
    python orbit_plot.py \\
        --csv ../../data/DLCfiltered/control/TmazeMA1_1/TmazeMA1_1.csv \\
        --bottom-arm 526 606 403 717 \\
        --left-arm   259 533 346 402 \\
        --right-arm  605 892 347 402 \\
        --top-arm    526 606   0 345
"""

import argparse
import json
import os
import pathlib

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV       = "../../data/DLCfiltered/control/PlusMazeMA1_1/PlusMazeMA1_1.csv"
DEFAULT_OUT_DIR   = "../../data/DLCfiltered"
LIKELIHOOD_THRESH = 0.6

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DLC_DIR = ROOT / "data" / "DLCfiltered"

ZONE_COLORS = {
    "bottom_arm": "#4FC3F7",
    "left_arm":   "#66BB6A",
    "right_arm":  "#FFA726",
    "top_arm":    "#BA68C8",
    "junction":   "#888888",
    "unknown":    "#222222",
}

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

ARMS = ["bottom_arm", "left_arm", "right_arm", "top_arm"]


def load_dlc_csv(csv_path, likelihood_thresh, jump_thresh=60.0, smooth=5):
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    tracking = {}
    for bp in df.columns.get_level_values(0).unique():
        x   = df[bp]["x"].values.astype(float)
        y   = df[bp]["y"].values.astype(float)
        lkh = df[bp]["likelihood"].values.astype(float)
        x[lkh < likelihood_thresh] = np.nan
        y[lkh < likelihood_thresh] = np.nan
        dx = np.diff(x, prepend=np.nan)
        dy = np.diff(y, prepend=np.nan)
        jumps = np.sqrt(dx**2 + dy**2) > jump_thresh
        x[jumps] = np.nan
        y[jumps] = np.nan
        nan_mask = np.isnan(x) | np.isnan(y)
        xs = pd.Series(x).rolling(smooth, center=True, min_periods=1).median().values.copy()
        ys = pd.Series(y).rolling(smooth, center=True, min_periods=1).median().values.copy()
        xs[nan_mask] = np.nan
        ys[nan_mask] = np.nan
        tracking[bp] = {"x": xs, "y": ys}
    return tracking


def assign_zones(x, y, zones):
    labels = np.full(len(x), "junction", dtype=object)
    valid  = ~(np.isnan(x) | np.isnan(y))
    labels[~valid] = "unknown"
    for name, (x0, x1, y0, y1) in zones.items():
        mask = valid & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        labels[mask] = name
    return labels


def auto_extent(zones, pad=30):
    all_b = list(zones.values())
    return (min(b[0] for b in all_b) - pad, max(b[1] for b in all_b) + pad,
            min(b[2] for b in all_b) - pad, max(b[3] for b in all_b) + pad)


def draw_zones(ax, zones, alpha=0.10):
    for name, (x0, x1, y0, y1) in zones.items():
        c = ZONE_COLORS[name]
        ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                     facecolor=c, alpha=alpha, zorder=1))
        ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                     linewidth=1.5, edgecolor=c, facecolor="none",
                     linestyle="--", zorder=10))
        label = name.replace("_", " ")
        if name == "right_arm":
            # Sag ust kose
            ax.text(x1 - 4, y0 + 16, label,
                    color=c, fontsize=8, fontweight="bold", zorder=11,
                    ha="right", va="top")
        elif name == "bottom_arm":
            # Kutunun en alti (y1 = görsel alt kenar, invert_yaxis ile)
            ax.text(x0 + 4, y1 - 4, label,
                    color=c, fontsize=8, fontweight="bold", zorder=11,
                    ha="left", va="bottom")
        else:
            ax.text(x0 + 4, y0 + 16, label,
                    color=c, fontsize=8, fontweight="bold", zorder=11)


def make_cmap(base_hex):
    base  = mcolors.to_rgb(base_hex)
    dark  = tuple(max(0.0, c*0.4) for c in base)
    light = tuple(min(1.0, 0.2+c*0.5) for c in base)
    return mcolors.LinearSegmentedColormap.from_list("t", [light, base, dark])


def draw_trajectory(ax, x, y, color_hex, lw=0.9, alpha=0.80):
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 5:
        return
    xi, yi = x[valid], y[valid]
    t = np.where(valid)[0]
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1)
    cmap   = make_cmap(color_hex)
    for i in range(len(xi)-1):
        ax.plot([xi[i], xi[i+1]], [yi[i], yi[i+1]],
                color=cmap(t_norm[i]), linewidth=lw,
                alpha=alpha, solid_capstyle="round")
    ax.scatter(xi[0],  yi[0],  color=cmap(0.15), s=55, zorder=6,
               edgecolors="white", linewidths=0.6)
    ax.scatter(xi[-1], yi[-1], color=cmap(0.85), s=40, zorder=6,
               edgecolors="white", linewidths=0.6, marker="D")


def draw_zone_colored(ax, x, y, labels):
    for i in range(len(x)-1):
        if np.isnan(x[i]) or np.isnan(y[i]) or np.isnan(x[i+1]) or np.isnan(y[i+1]):
            continue
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                color=ZONE_COLORS.get(labels[i], "#555555"),
                linewidth=1.0, alpha=0.75, solid_capstyle="round")


def style_ax(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="#AAAAAA", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def plot_zone_trajectory(tracking, zones, labels, name, out_path):
    bp = "body_center" if "body_center" in tracking else list(tracking.keys())[0]
    x, y = tracking[bp]["x"], tracking[bp]["y"]
    xmin, xmax, ymin, ymax = auto_extent(zones)
    valid = ~(np.isnan(x) | np.isnan(y))
    n     = valid.sum()

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0A0A0A")
    style_ax(ax)
    draw_zones(ax, zones)
    draw_zone_colored(ax, x, y, labels)

    if valid.sum() > 0:
        xi, yi = x[valid], y[valid]
        ax.scatter(xi[0],  yi[0],  color="white", s=60, zorder=8,
                   edgecolors="#00FF00", linewidths=1.5, label="start")
        ax.scatter(xi[-1], yi[-1], color="white", s=45, zorder=8,
                   edgecolors="#FF0000", linewidths=1.5, marker="D", label="end")

    zone_labels = ARMS + ["junction"]
    patches = [
        mpatches.Patch(
            color=ZONE_COLORS[z],
            label=f"{z.replace('_',' ')} {(labels==z).sum()/max(n,1)*100:.0f}%"
        )
        for z in zone_labels
    ]
    ax.legend(handles=patches, loc="lower right",
              framealpha=0.3, facecolor="#222222",
              edgecolor="#555555", labelcolor="white", fontsize=8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_ylabel("Y (pixels)", color="#CCCCCC", fontsize=12)
    ax.set_title(
        f"Plus Maze Trajectory — {name}  [{bp}]\n"
        f"zone-colored  |  circle=start  diamond=end",
        color="white", fontsize=12, pad=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Zone trajectory -> {out_path}")


def plot_bodypart_grid(tracking, zones, name, out_path):
    KEY_BPS = ["nose", "head", "body_center", "tail_base"]
    bps = [bp for bp in KEY_BPS if bp in tracking]
    ncols = 2
    nrows = int(np.ceil(len(bps) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*5, nrows*4.5),
                             facecolor="#0A0A0A")
    axes = np.array(axes).flatten()
    xmin, xmax, ymin, ymax = auto_extent(zones)

    for idx, bp in enumerate(bps):
        ax = axes[idx]
        style_ax(ax)
        draw_zones(ax, zones, alpha=0.08)
        draw_trajectory(ax, tracking[bp]["x"], tracking[bp]["y"],
                        BODYPART_COLORS.get(bp, "#FFFFFF"))
        ax.set_xlim(xmin-10, xmax+10)
        ax.set_ylim(ymin-10, ymax+10)
        ax.invert_yaxis()
        ax.set_title(bp, color=BODYPART_COLORS.get(bp, "#FFFFFF"), fontsize=9, pad=4)
        ax.set_xlabel("X (px)", color="#666666", fontsize=7)
        ax.set_ylabel("Y (px)", color="#666666", fontsize=7)

    for idx in range(len(bps), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Plus Maze Bodypart Trajectories — {name}\n"
        f"B=blue | L=green | R=orange | T=purple | junction=grey\n"
        f"circle=start  diamond=end  pale->dark=early->late",
        color="#DDDDDD", fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Bodypart grid -> {out_path}")


def run_single(csv_path, zones, out_dir, likelihood, jump_thresh, smooth):
    """Tek bir subject icin her iki grafigi uretir."""
    stem_name = os.path.splitext(os.path.basename(csv_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"  {stem_name} ...", end=" ", flush=True)
    tracking = load_dlc_csv(csv_path, likelihood, jump_thresh, smooth)
    bc       = tracking.get("body_center", list(tracking.values())[0])
    labels   = assign_zones(bc["x"], bc["y"], zones)

    valid = ~(np.isnan(bc["x"]) | np.isnan(bc["y"]))
    n     = valid.sum()
    zone_pct = {z: (labels == z).sum() / max(n, 1) * 100
                for z in list(zones.keys()) + ["junction"]}
    print("  ".join(f"{z.split('_')[0][0].upper()}={v:.0f}%"
                    for z, v in zone_pct.items()))

    plot_zone_trajectory(tracking, zones, labels, stem_name,
                         os.path.join(out_dir, f"{stem_name}_plus_maze_orbit.png"))
    plot_bodypart_grid(tracking, zones, stem_name,
                       os.path.join(out_dir, f"{stem_name}_plus_maze_bodyparts.png"))


def run_batch(likelihood, jump_thresh, smooth, fallback_zones=None):
    """
    data/DLCfiltered/ altindaki tum PlusMaze klasorlerini tarar.
    Oncelik: per-subject arm_coords.json → yoksa fallback_zones (CLI'dan).
    """
    csvs = sorted([
        p for p in DLC_DIR.rglob("*.csv")
        if "PlusMaze" in p.name
        and "metrics" not in p.name
    ])

    central_json = ROOT / "data" / "arm_coords.json"

    found = skipped = 0
    for csv_path in csvs:
        json_path = csv_path.parent / f"{csv_path.stem}_arm_coords.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as fh:
                zones = {k: tuple(v) for k, v in json.load(fh).items()}
        elif central_json.exists():
            with open(central_json, encoding="utf-8") as fh:
                zones = {k: tuple(v) for k, v in json.load(fh).items()}
        elif fallback_zones is not None:
            zones = fallback_zones
        else:
            print(f"  [skip] {csv_path.stem} — koordinat bulunamadi")
            skipped += 1
            continue
        run_single(str(csv_path), zones, str(csv_path.parent),
                   likelihood, jump_thresh, smooth)
        found += 1

    print(f"\n[done] {found} subject islendi"
          + (f", {skipped} atlandi" if skipped else ""))


def main():
    parser = argparse.ArgumentParser(
        description="Plus maze trajectory visualization.\n"
                    "Argumansiz calistirilirsa tum PlusMaze subject'lerini batch isler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv",        default=None,
                        help="Tek subject DLC CSV (verilmezse batch mod)")
    parser.add_argument("--out-dir",    default=None, dest="out_dir",
                        help="Cikti klasoru (tek mod; batch modda CSV klasoru kullanilir)")
    parser.add_argument("--bottom-arm", nargs=4, type=float,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="bottom_arm")
    parser.add_argument("--left-arm",   nargs=4, type=float,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="left_arm")
    parser.add_argument("--right-arm",  nargs=4, type=float,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="right_arm")
    parser.add_argument("--top-arm",    nargs=4, type=float,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="top_arm")
    parser.add_argument("--likelihood", type=float, default=LIKELIHOOD_THRESH)
    parser.add_argument("--jump-thresh",type=float, default=60.0, dest="jump_thresh")
    parser.add_argument("--smooth",     type=int,   default=5)
    args = parser.parse_args()

    kw = dict(likelihood=args.likelihood,
              jump_thresh=args.jump_thresh,
              smooth=args.smooth)

    if args.csv is None:
        # Batch mod: arm_coords.json varsa onu kullan, yoksa CLI koordinatlari
        fallback = None
        if all([args.bottom_arm, args.left_arm, args.right_arm, args.top_arm]):
            fallback = {
                "bottom_arm": tuple(args.bottom_arm),
                "left_arm":   tuple(args.left_arm),
                "right_arm":  tuple(args.right_arm),
                "top_arm":    tuple(args.top_arm),
            }
        print(f"Batch mod — {DLC_DIR} altindaki tum PlusMaze subject'leri taranıyor...\n")
        run_batch(fallback_zones=fallback, **kw)
    else:
        # Tek subject modu
        if not os.path.isfile(args.csv):
            raise FileNotFoundError(f"CSV not found: {args.csv}")
        if not all([args.bottom_arm, args.left_arm, args.right_arm, args.top_arm]):
            raise ValueError("Tek mod icin --bottom-arm / --left-arm / --right-arm / --top-arm gerekli")
        zones = {
            "bottom_arm": tuple(args.bottom_arm),
            "left_arm":   tuple(args.left_arm),
            "right_arm":  tuple(args.right_arm),
            "top_arm":    tuple(args.top_arm),
        }
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
        run_single(args.csv, zones, out_dir, **kw)


if __name__ == "__main__":
    main()
