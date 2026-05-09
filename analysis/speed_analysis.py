"""
speed_analysis.py
-----------------
Per-subject and cohort-level rat speed analysis from DLC tracking.

Why this module exists
----------------------
DLC keypoint speed estimates are noisy: localisation jitter on a single
keypoint, when differentiated, produces a high-variance velocity signal that
inflates within- and between-subject SD on every locomotor metric. This
module is built around four SD-reducing steps:

  1. Multi-keypoint trunk centroid:
     median of {body_center, head, tail_base} per frame. Independent
     localisation noise on the three trunk keypoints averages out — the
     trunk-median position has noticeably lower frame-to-frame jitter than
     any single keypoint.
  2. Per-keypoint cleaning (likelihood + jump filters) before the median,
     so a single bad keypoint doesn't poison the trunk centroid.
  3. Linear interpolation across short NaN gaps (default ≤ 10 frames),
     so the smoother isn't broken into fragmented runs by transient
     dropouts.
  4. Savitzky-Golay analytical first derivative (window 31 ≈ 1.0 s @ 30 fps,
     polyorder 3): fits a local polynomial and reads off the derivative at
     the centre. Equivalent to differentiating a smoothed signal but
     better behaved at the edges and lower variance than diff(rolling_mean).

Each subject's `sd_speed_raw_px_s` (single-keypoint diff-based speed SD) is
reported alongside `sd_speed_filt_px_s` (this pipeline's speed SD), and
`sd_reduction_pct` quantifies the gain.

Outputs (per subject, written next to the input CSV)
----------------------------------------------------
  <subject>_speed.csv          frame, t_s, x_px, y_px, vx_px_s, vy_px_s, speed_px_s
  <subject>_speed.png          three-panel time series (Vx, Vy, |V|)
  <subject>_speed_summary.csv  one-row summary (mean / median / sd / max / p95)

Batch outputs (under --out-dir, default `data/`)
------------------------------------------------
  speed_summary_all.csv     one row per subject with cohort labels
  speed_cohort_summary.csv  cohort means ± SD
  speed_comparison.png      per-cohort boxplot + bar (mean ± SD)

Usage
-----
  # single subject
  python analysis/speed_analysis.py \\
      --csv data/DLCfiltered/control/OpenFieldMA1_1/OpenFieldMA1_1.csv

  # all subjects
  python analysis/speed_analysis.py --batch-dir data/DLCfiltered

  # also report cm/s (60-cm OFT with default arena ⇒ ~6.3 px/cm)
  python analysis/speed_analysis.py --batch-dir data/DLCfiltered --px-per-cm 6.3
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_FPS         = 30
DEFAULT_LIKELIHOOD  = 0.6
DEFAULT_JUMP_THRESH = 60.0   # px frame-to-frame
DEFAULT_KEYPOINTS   = ("body_center", "head", "tail_base")
DEFAULT_SG_WINDOW   = 31     # frames; ≈ 1.0 s @ 30 fps
DEFAULT_SG_ORDER    = 3
DEFAULT_MAX_GAP     = 10     # frames; longer NaN gaps stay NaN

COHORT_MAP = {
    "MA1": ("MA1", "Control"),
    "MA2": ("MA2", "Control"),
    "MA3": ("MA3", "Aspartame"),
    "MA4": ("MA4", "Aspartame"),
    "MA5": ("MA5", "Grapefruit"),
    "MA6": ("MA6", "Grapefruit"),
    "MA7": ("MA7", "Aspartame+Grapefruit"),
    "MA8": ("MA8", "Aspartame+Grapefruit"),
}

GROUP_COLORS = {
    "Control":              "#4C72B0",
    "Aspartame":            "#DD8452",
    "Grapefruit":           "#55A868",
    "Aspartame+Grapefruit": "#C44E52",
}


# ── DLC loading + per-keypoint cleaning ────────────────────────────────────────

def load_dlc(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = ["_".join(c[-2:]).strip() for c in df.columns.values]
    return df


def clean_keypoint(x: np.ndarray, y: np.ndarray, lk: np.ndarray,
                   likelihood_thresh: float, jump_thresh: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    bad = lk < likelihood_thresh
    x[bad] = np.nan
    y[bad] = np.nan
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    jump = np.sqrt(dx ** 2 + dy ** 2) > jump_thresh
    x[jump] = np.nan
    y[jump] = np.nan
    return x, y


def trunk_centroid(df: pd.DataFrame, keypoints: tuple,
                   likelihood_thresh: float, jump_thresh: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise nanmedian of cleaned keypoints — robust to single-keypoint dropouts."""
    xs, ys = [], []
    for kp in keypoints:
        if f"{kp}_x" not in df.columns:
            print(f"[!] keypoint '{kp}' not in CSV — skipping")
            continue
        x = df[f"{kp}_x"].values
        y = df[f"{kp}_y"].values
        lk = df[f"{kp}_likelihood"].values
        x, y = clean_keypoint(x, y, lk, likelihood_thresh, jump_thresh)
        xs.append(x); ys.append(y)
    if not xs:
        raise ValueError(f"None of the requested keypoints {keypoints} found in CSV")
    X = np.vstack(xs)
    Y = np.vstack(ys)
    cx = np.nanmedian(X, axis=0)
    cy = np.nanmedian(Y, axis=0)
    n_valid = np.sum(~np.isnan(X), axis=0)
    cx[n_valid == 0] = np.nan
    cy[n_valid == 0] = np.nan
    return cx, cy


def interpolate_short_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    return pd.Series(arr).interpolate(
        method="linear", limit=max_gap, limit_area="inside"
    ).values


# ── velocity via Savitzky-Golay analytical derivative ──────────────────────────

def sg_velocity(x: np.ndarray, fps: float, window: int, order: int) -> np.ndarray:
    """Smoothed first derivative (px/s). NaN-safe: SG is applied per
    contiguous valid run; runs shorter than `window` stay NaN."""
    valid = ~np.isnan(x)
    out = np.full_like(x, np.nan, dtype=float)
    i = 0
    n = len(x)
    while i < n:
        if not valid[i]:
            i += 1; continue
        j = i
        while j < n and valid[j]:
            j += 1
        if j - i >= window:
            out[i:j] = savgol_filter(x[i:j], window, order,
                                     deriv=1, delta=1.0 / fps)
        i = j
    return out


def compute_speed(df: pd.DataFrame, fps: float, keypoints: tuple,
                  likelihood_thresh: float, jump_thresh: float,
                  sg_window: int, sg_order: int, max_gap: int):
    cx, cy = trunk_centroid(df, keypoints, likelihood_thresh, jump_thresh)
    cx = interpolate_short_gaps(cx, max_gap)
    cy = interpolate_short_gaps(cy, max_gap)
    vx = sg_velocity(cx, fps, sg_window, sg_order)
    vy = sg_velocity(cy, fps, sg_window, sg_order)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    return cx, cy, vx, vy, speed


# ── summary helpers ────────────────────────────────────────────────────────────

def summarize_speed(speed: np.ndarray) -> dict:
    s = speed[~np.isnan(speed)]
    if not len(s):
        return dict(n_valid=0,
                    mean_speed_px_s=float("nan"),
                    median_speed_px_s=float("nan"),
                    sd_speed_px_s=float("nan"),
                    max_speed_px_s=float("nan"),
                    p95_speed_px_s=float("nan"))
    return dict(
        n_valid           = int(len(s)),
        mean_speed_px_s   = round(float(np.mean(s)),   3),
        median_speed_px_s = round(float(np.median(s)), 3),
        sd_speed_px_s     = round(float(np.std(s)),    3),
        max_speed_px_s    = round(float(np.max(s)),    3),
        p95_speed_px_s    = round(float(np.percentile(s, 95)), 3),
    )


def diagnostic_sd_reduction(df: pd.DataFrame, fps: float, keypoints: tuple,
                             likelihood_thresh: float, jump_thresh: float,
                             sg_window: int, sg_order: int, max_gap: int
                             ) -> tuple[float, float]:
    """Compare speed SD: naive single-keypoint diff vs full pipeline."""
    x = df["body_center_x"].values.astype(float).copy()
    y = df["body_center_y"].values.astype(float).copy()
    lk = df["body_center_likelihood"].values.astype(float)
    x[lk < likelihood_thresh] = np.nan
    y[lk < likelihood_thresh] = np.nan
    raw = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) * fps
    sd_raw = float(np.nanstd(raw))

    _, _, _, _, sp = compute_speed(df, fps, keypoints,
                                    likelihood_thresh, jump_thresh,
                                    sg_window, sg_order, max_gap)
    sd_filt = float(np.nanstd(sp))
    return sd_raw, sd_filt


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_speed_timeseries(t, vx, vy, speed, subject_id, group, out_path,
                          px_per_cm=None) -> None:
    factor = 1.0 / px_per_cm if px_per_cm else 1.0
    unit = "cm/s" if px_per_cm else "px/s"
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, vx * factor, lw=0.7, color="#4C72B0")
    axes[0].axhline(0, color="grey", lw=0.5, ls="--")
    axes[0].set_ylabel(f"Vx ({unit})")
    axes[1].plot(t, vy * factor, lw=0.7, color="#DD8452")
    axes[1].axhline(0, color="grey", lw=0.5, ls="--")
    axes[1].set_ylabel(f"Vy ({unit})")
    axes[2].plot(t, speed * factor, lw=0.7, color="#55A868")
    axes[2].set_ylabel(f"|V| ({unit})")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"{subject_id} ({group}) — trunk-centroid speed", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cohort_comparison(df: pd.DataFrame, out_path: str,
                            px_per_cm: float | None = None) -> None:
    factor = 1.0 / px_per_cm if px_per_cm else 1.0
    unit = "cm/s" if px_per_cm else "px/s"
    groups = sorted(df["group"].unique())
    colors = [GROUP_COLORS.get(g, "#888888") for g in groups]
    data_mean = [df[df["group"] == g]["mean_speed_px_s"].values * factor for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bp = axes[0].boxplot(data_mean, labels=groups, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_title(f"Per-subject mean speed")
    axes[0].set_ylabel(f"Mean speed ({unit})")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    means = [float(np.mean(d)) if len(d) else float("nan") for d in data_mean]
    sds   = [float(np.std(d))  if len(d) else 0.0 for d in data_mean]
    axes[1].bar(groups, means, yerr=sds, capsize=6, color=colors, alpha=0.8)
    axes[1].set_title(f"Cohort mean ± SD")
    axes[1].set_ylabel(f"Speed ({unit})")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── per-subject + batch pipeline ───────────────────────────────────────────────

def find_subject_csvs(batch_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(batch_dir):
        for f in files:
            if (f.startswith("OpenField") and f.endswith(".csv")
                    and "_behavior" not in f
                    and "_oft_metrics" not in f
                    and "_speed" not in f):
                out.append(os.path.join(root, f))
    return sorted(out)


def process_subject(csv_path: str, fps: float, keypoints: tuple,
                    likelihood_thresh: float, jump_thresh: float,
                    sg_window: int, sg_order: int, max_gap: int,
                    px_per_cm: float | None = None) -> dict:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    subject_dir = os.path.dirname(csv_path)
    df = load_dlc(csv_path)

    cx, cy, vx, vy, speed = compute_speed(
        df, fps, keypoints, likelihood_thresh, jump_thresh,
        sg_window, sg_order, max_gap,
    )
    n_frames = len(cx)
    t = np.arange(n_frames) / fps

    ts_csv = os.path.join(subject_dir, f"{base}_speed.csv")
    pd.DataFrame({
        "frame":      np.arange(n_frames),
        "t_s":        t.round(3),
        "x_px":       np.round(cx, 2),
        "y_px":       np.round(cy, 2),
        "vx_px_s":    np.round(vx, 3),
        "vy_px_s":    np.round(vy, 3),
        "speed_px_s": np.round(speed, 3),
    }).to_csv(ts_csv, index=False)

    m = re.search(r"(MA\d+)_(\d+)", base)
    cohort = m.group(1) if m else "unknown"
    run    = m.group(2) if m else "?"
    subject_id = f"{cohort}_{run}"
    group_info = COHORT_MAP.get(cohort, (cohort, "Unknown"))

    fig_path = os.path.join(subject_dir, f"{base}_speed.png")
    plot_speed_timeseries(t, vx, vy, speed, subject_id, group_info[1],
                          fig_path, px_per_cm=px_per_cm)

    sd_raw, sd_filt = diagnostic_sd_reduction(
        df, fps, keypoints, likelihood_thresh, jump_thresh,
        sg_window, sg_order, max_gap,
    )

    summary = summarize_speed(speed)
    summary.update({
        "subject_id":         subject_id,
        "cohort":             group_info[0],
        "group":              group_info[1],
        "n_frames":           n_frames,
        "session_s":          round(n_frames / fps, 1),
        "sd_speed_raw_px_s":  round(sd_raw, 3),
        "sd_speed_filt_px_s": round(sd_filt, 3),
        "sd_reduction_pct":   round((sd_raw - sd_filt) / sd_raw * 100, 1)
                                if sd_raw > 0 else float("nan"),
    })
    if px_per_cm:
        for k in list(summary):
            if k.endswith("_px_s") and isinstance(summary[k], float):
                summary[k.replace("_px_s", "_cm_s")] = round(summary[k] / px_per_cm, 3)

    sum_csv = os.path.join(subject_dir, f"{base}_speed_summary.csv")
    pd.DataFrame([summary]).to_csv(sum_csv, index=False)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DLC-based per-subject + cohort speed analysis (low-SD pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", help="Single-subject DLC CSV.")
    p.add_argument("--batch-dir", dest="batch_dir",
                   help="Folder to scan recursively for OpenField*.csv.")
    p.add_argument("--out-dir", dest="out_dir", default="data",
                   help="Output dir for batch summaries (default: data).")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help=f"Video frame rate (default {DEFAULT_FPS}).")
    p.add_argument("--keypoints", default=",".join(DEFAULT_KEYPOINTS),
                   help="Comma-separated keypoints to median per frame "
                        "(default: body_center,head,tail_base).")
    p.add_argument("--likelihood", type=float, default=DEFAULT_LIKELIHOOD,
                   help=f"Per-keypoint likelihood threshold (default {DEFAULT_LIKELIHOOD}).")
    p.add_argument("--jump-thresh", type=float, default=DEFAULT_JUMP_THRESH,
                   dest="jump_thresh",
                   help=f"Max per-keypoint frame-to-frame jump (px) "
                        f"(default {DEFAULT_JUMP_THRESH}).")
    p.add_argument("--sg-window", type=int, default=DEFAULT_SG_WINDOW,
                   dest="sg_window",
                   help=f"Savitzky-Golay window in frames; must be odd "
                        f"(default {DEFAULT_SG_WINDOW} ≈ 1 s @ 30 fps).")
    p.add_argument("--sg-order", type=int, default=DEFAULT_SG_ORDER,
                   dest="sg_order",
                   help=f"SG polynomial order (default {DEFAULT_SG_ORDER}).")
    p.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP,
                   dest="max_gap",
                   help="Max NaN gap (frames) to bridge by linear interpolation "
                        f"(default {DEFAULT_MAX_GAP}).")
    p.add_argument("--px-per-cm", type=float, default=None, dest="px_per_cm",
                   help="If set, also report cm/s alongside px/s.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.sg_window % 2 == 0:
        args.sg_window += 1
        print(f"[i] sg_window must be odd — bumped to {args.sg_window}")
    keypoints = tuple(k.strip() for k in args.keypoints.split(",") if k.strip())

    common = dict(
        fps=args.fps, keypoints=keypoints,
        likelihood_thresh=args.likelihood, jump_thresh=args.jump_thresh,
        sg_window=args.sg_window, sg_order=args.sg_order,
        max_gap=args.max_gap, px_per_cm=args.px_per_cm,
    )

    if args.batch_dir:
        csvs = find_subject_csvs(args.batch_dir)
        if not csvs:
            print(f"[!] No OpenField*.csv under {args.batch_dir}")
            sys.exit(1)
        print(f"Batch: {len(csvs)} subjects\n")
        rows = []
        for csv_path in csvs:
            print(f"  -> {os.path.basename(csv_path)}")
            row = process_subject(csv_path, **common)
            rows.append(row)
        all_df = pd.DataFrame(rows)

        os.makedirs(args.out_dir, exist_ok=True)
        out_csv = os.path.join(args.out_dir, "speed_summary_all.csv")
        all_df.to_csv(out_csv, index=False)
        print(f"\nSubject summary  -> {out_csv}")

        num_cols = all_df.select_dtypes(include="number").columns.tolist()
        cohort_df = all_df.groupby("group")[num_cols].agg(["mean", "std"]).round(3)
        cohort_csv = os.path.join(args.out_dir, "speed_cohort_summary.csv")
        cohort_df.to_csv(cohort_csv)
        print(f"Cohort summary   -> {cohort_csv}")

        fig_path = os.path.join(args.out_dir, "speed_comparison.png")
        plot_cohort_comparison(all_df, fig_path, px_per_cm=args.px_per_cm)
        print(f"Comparison fig   -> {fig_path}")

        red_mean = all_df["sd_reduction_pct"].mean()
        print(f"\nMean per-subject speed-SD reduction "
              f"(naive diff → trunk-median + SG-deriv): {red_mean:.1f}%")

    elif args.csv:
        if not os.path.isfile(args.csv):
            print(f"[!] CSV not found: {args.csv}"); sys.exit(1)
        row = process_subject(args.csv, **common)
        print(f"\n  {row['subject_id']} ({row['group']})")
        print(f"  mean / median speed : {row['mean_speed_px_s']:.2f} / "
              f"{row['median_speed_px_s']:.2f} px/s")
        print(f"  speed SD            : {row['sd_speed_px_s']:.2f} px/s")
        print(f"  max / p95 speed     : {row['max_speed_px_s']:.2f} / "
              f"{row['p95_speed_px_s']:.2f} px/s")
        print(f"  speed SD reduction  : {row['sd_speed_raw_px_s']:.2f} → "
              f"{row['sd_speed_filt_px_s']:.2f} px/s "
              f"({row['sd_reduction_pct']:.1f}%)")

    else:
        build_parser().print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
