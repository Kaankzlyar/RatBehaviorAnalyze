"""
kutu_validation.py
------------------
Cross-validate DLC trunk-centroid speed against the professor's classical
HSV-blob tracker (Kutu_v1.m), per subject.

Inputs
------
  - Professor `_res.mat` files in `data/DLCfiltered/Kare/` (xc, yc in pixels @ 30 fps)
  - DLC `_speed.csv` files in `data/DLCfiltered/<cohort>/<subject>/`
    produced by speed_analysis.py

What it computes
----------------
For each subject present in BOTH pipelines (intersection of MA1/3/5/7 ×
{_1,_2,_3} with the Kare/ MAk-N files), we apply the SAME Savitzky-Golay
analytical derivative (sg_velocity from speed_analysis.py) to the two
centroid traces. Any disagreement therefore reflects the centroid source
(DLC trunk-median vs HSV-blob centroid), not the smoothing.

Per-subject metrics
-------------------
  Pearson r, mean abs diff (px/s), RMSE, mean speed each pipeline, ratio.

Outputs
-------
  data/DLCfiltered/<cohort>/<subject>/<subject>_kutu_validation.png
      two-panel: overlay of speed traces + Bland-Altman
  data/kutu_validation_summary.csv
      one row per subject with agreement metrics
  data/kutu_validation_overlay.png
      across-subject scatter (Kutu mean vs DLC mean)

Usage
-----
  python analysis/kutu_validation.py
  python analysis/kutu_validation.py --kare-dir data/DLCfiltered/Kare \\
                                     --dlc-dir  data/DLCfiltered \\
                                     --out-dir  data
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_analysis import (  # noqa: E402
    DEFAULT_FPS, DEFAULT_SG_WINDOW, DEFAULT_SG_ORDER,
    COHORT_MAP, GROUP_COLORS,
    sg_velocity, interpolate_short_gaps,
)

COHORT_DIR = {
    "MA1": "control",
    "MA3": "ASP",
    "MA5": "Greyfurt",
    "MA7": "ASP ve Greyfurt",
}


def load_kutu_centroid(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    m = loadmat(mat_path)
    xc = np.asarray(m["xc"]).ravel().astype(float)
    yc = np.asarray(m["yc"]).ravel().astype(float)
    xc[xc == 0] = np.nan
    yc[yc == 0] = np.nan
    return xc, yc


def kutu_speed(xc: np.ndarray, yc: np.ndarray, fps: float,
               sg_window: int, sg_order: int) -> np.ndarray:
    xc = interpolate_short_gaps(xc, max_gap=10)
    yc = interpolate_short_gaps(yc, max_gap=10)
    vx = sg_velocity(xc, fps, sg_window, sg_order)
    vy = sg_velocity(yc, fps, sg_window, sg_order)
    return np.sqrt(vx ** 2 + vy ** 2)


def kutu_to_subject(mat_basename: str) -> tuple[str, str, str] | None:
    """`MA1-1_res` -> ('MA1', 'control', 'OpenFieldMA1_1'). None if not in DLC set."""
    m = re.match(r"MA(\d+)-(\d+)_res", mat_basename)
    if not m:
        return None
    cohort = f"MA{m.group(1)}"
    run    = m.group(2)
    if cohort not in COHORT_DIR:
        return None
    return cohort, COHORT_DIR[cohort], f"OpenField{cohort}_{run}"


def find_pairs(kare_dir: str, dlc_dir: str) -> list[tuple[str, str, str, str, str]]:
    """Return [(cohort, group, subject_id, mat_path, dlc_speed_csv), ...]."""
    pairs = []
    for f in sorted(os.listdir(kare_dir)):
        if not f.endswith("_res.mat"):
            continue
        info = kutu_to_subject(os.path.splitext(f)[0])
        if info is None:
            continue
        cohort, cohort_dir, subject = info
        dlc_csv = os.path.join(dlc_dir, cohort_dir, subject, f"{subject}_speed.csv")
        if not os.path.isfile(dlc_csv):
            print(f"[skip] {subject}: no DLC speed CSV at {dlc_csv}")
            continue
        group = COHORT_MAP.get(cohort, (cohort, "Unknown"))[1]
        pairs.append((cohort, group, subject, os.path.join(kare_dir, f), dlc_csv))
    return pairs


def per_subject_validation(cohort: str, group: str, subject: str,
                           mat_path: str, dlc_csv: str,
                           fps: float, sg_window: int, sg_order: int,
                           out_dir: str) -> dict:
    xc, yc = load_kutu_centroid(mat_path)
    sp_kutu = kutu_speed(xc, yc, fps, sg_window, sg_order)

    dlc = pd.read_csv(dlc_csv)
    sp_dlc = dlc["speed_px_s"].values

    n = min(len(sp_kutu), len(sp_dlc))
    sp_kutu = sp_kutu[:n]
    sp_dlc  = sp_dlc[:n]
    t       = np.arange(n) / fps

    valid = ~(np.isnan(sp_kutu) | np.isnan(sp_dlc))
    a = sp_kutu[valid]
    b = sp_dlc[valid]
    if len(a) < 30:
        print(f"[skip] {subject}: <30 paired valid samples")
        return {}

    diff = a - b
    metrics = {
        "subject_id":          subject.replace("OpenField", "").replace("_", "_"),
        "cohort":              cohort,
        "group":               group,
        "n_paired":            int(len(a)),
        "mean_kutu_px_s":      round(float(np.mean(a)), 3),
        "mean_dlc_px_s":       round(float(np.mean(b)), 3),
        "mean_diff_px_s":      round(float(np.mean(diff)), 3),
        "mean_abs_diff_px_s":  round(float(np.mean(np.abs(diff))), 3),
        "rmse_px_s":           round(float(np.sqrt(np.mean(diff ** 2))), 3),
        "pearson_r":           round(float(np.corrcoef(a, b)[0, 1]), 3),
        "ratio_kutu_over_dlc": round(float(np.mean(a) / np.mean(b)) if np.mean(b) > 0 else float("nan"), 3),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(t, sp_kutu, lw=0.6, color="#C44E52", label="Kutu (HSV-blob)")
    axes[0].plot(t, sp_dlc,  lw=0.6, color="#4C72B0", label="DLC (trunk-median)", alpha=0.8)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (px/s)")
    axes[0].set_title(f"{subject} — speed overlay (r={metrics['pearson_r']:.2f})")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    mean_ab = (a + b) / 2
    sd_diff = float(np.std(diff))
    bias    = float(np.mean(diff))
    axes[1].scatter(mean_ab, diff, s=4, alpha=0.3, color="#666666")
    axes[1].axhline(bias, color="#C44E52", lw=1, label=f"bias = {bias:.1f}")
    axes[1].axhline(bias + 1.96 * sd_diff, color="#C44E52", lw=0.7, ls="--",
                    label=f"±1.96 SD = {1.96 * sd_diff:.1f}")
    axes[1].axhline(bias - 1.96 * sd_diff, color="#C44E52", lw=0.7, ls="--")
    axes[1].set_xlabel("Mean speed (px/s)")
    axes[1].set_ylabel("Kutu − DLC (px/s)")
    axes[1].set_title("Bland-Altman")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{subject} ({group}) — Kutu_v1 vs DLC speed", fontsize=12)
    fig.tight_layout()

    subject_dir = os.path.join(out_dir, COHORT_DIR[cohort], subject)
    os.makedirs(subject_dir, exist_ok=True)
    fig_path = os.path.join(subject_dir, f"{subject}_kutu_validation.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return metrics


def cross_subject_overlay(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 6))
    for grp, sub in df.groupby("group"):
        ax.scatter(sub["mean_dlc_px_s"], sub["mean_kutu_px_s"],
                   s=70, alpha=0.85, label=grp,
                   color=GROUP_COLORS.get(grp, "#888888"),
                   edgecolor="black", linewidth=0.5)
    lim = max(df["mean_kutu_px_s"].max(), df["mean_dlc_px_s"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("DLC mean speed (px/s)")
    ax.set_ylabel("Kutu mean speed (px/s)")
    ax.set_title("Cross-pipeline mean speed agreement")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate DLC speed against professor's Kutu_v1 _res.mat files.")
    p.add_argument("--kare-dir", default="data/DLCfiltered/Kare",
                   help="Folder with MAk-N_res.mat files.")
    p.add_argument("--dlc-dir",  default="data/DLCfiltered",
                   help="Root of cohort folders with DLC speed CSVs.")
    p.add_argument("--out-dir",  default="data",
                   help="Where to write summary CSV + overlay PNG.")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--sg-window", type=int, default=DEFAULT_SG_WINDOW, dest="sg_window")
    p.add_argument("--sg-order",  type=int, default=DEFAULT_SG_ORDER,  dest="sg_order")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.sg_window % 2 == 0:
        args.sg_window += 1

    pairs = find_pairs(args.kare_dir, args.dlc_dir)
    if not pairs:
        print("[!] No matched (Kutu .mat, DLC speed CSV) pairs found.")
        sys.exit(1)
    print(f"Found {len(pairs)} matched subject pairs\n")

    rows = []
    for cohort, group, subject, mat_path, dlc_csv in pairs:
        print(f"  -> {subject:24s} ({group})")
        m = per_subject_validation(
            cohort, group, subject, mat_path, dlc_csv,
            fps=args.fps, sg_window=args.sg_window, sg_order=args.sg_order,
            out_dir=args.dlc_dir,
        )
        if m:
            rows.append(m)

    if not rows:
        print("[!] No subjects produced metrics.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    sum_csv = os.path.join(args.out_dir, "kutu_validation_summary.csv")
    df.to_csv(sum_csv, index=False)
    print(f"\nSummary CSV  -> {sum_csv}")

    overlay_png = os.path.join(args.out_dir, "kutu_validation_overlay.png")
    cross_subject_overlay(rows, overlay_png)
    print(f"Overlay PNG  -> {overlay_png}")

    print(f"\nMean Pearson r across subjects: {df['pearson_r'].mean():.3f}")
    print(f"Mean ratio Kutu / DLC          : {df['ratio_kutu_over_dlc'].mean():.3f}")
    print(f"Mean RMSE (px/s)               : {df['rmse_px_s'].mean():.2f}")


if __name__ == "__main__":
    main()
