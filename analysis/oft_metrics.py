"""
oft_metrics.py
--------------
Compute quantitative Open Field Test (OFT) metrics from DLC tracking data.

Metrics computed
----------------
  Locomotion  : total_distance_px, mean_speed_px_s, max_speed_px_s
  Thigmotaxis : pct_time_center, pct_time_periphery, center_zone_entries
  Freezing    : freeze_bout_count, total_freeze_s, pct_time_freeze
  Entropy     : spatial_entropy_bits (stereotypy / looping measure)
  Rearing     : rear_bout_count, rear_total_s, pct_time_rearing
  Grooming    : groom_bout_count, groom_total_s, pct_time_grooming

Biological context
------------------
  Group       Treatment                   Expected profile
  -------     --------                    ----------------
  MA1         Control (vehicle)           Balanced exploration, moderate rearing/grooming
  MA3         Aspartame                   Manik: high distance, low thigmotaxis
                                          Depresif: low distance, high thigmotaxis/freeze
  MA5         Aspartame + Grapefruit      Intermediate (interaction arm)
  MA7         Grapefruit only             Anxiolytic-like: higher center time,
                                          less freezing vs MA3

Usage — single subject
----------------------
  python analysis/oft_metrics.py \\
      --csv data/DLCfiltered/OpenFieldMA1_1/OpenFieldMA1_1.csv \\
      --arena 396 776 153 530

Usage — batch (all subjects under a parent folder)
--------------------------------------------------
  python analysis/oft_metrics.py \\
      --batch-dir data/DLCfiltered \\
      --arena 396 776 153 530 \\
      --out data/oft_metrics_all.csv

The per-subject output is saved next to the input CSV as
  <subject>_oft_metrics.csv
The batch output (--out) contains one row per subject plus cohort labels.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_FPS          = 30
DEFAULT_LIKELIHOOD   = 0.6
DEFAULT_JUMP_THRESH  = 60.0    # px — discard unrealistic frame-to-frame jumps
DEFAULT_SMOOTH       = 5       # rolling-median window for speed computation
DEFAULT_MARGIN       = 0.20    # inner-zone margin (20 % of arena dimensions)

# freezing: consecutive frames with speed below threshold = freezing episode
# 10 px/s catches genuinely stationary rats; grooming bouts overlap with this
# range, so interpret freeze metric alongside grooming.
FREEZE_SPEED_THRESH  = 10.0   # px/s  — below this → candidate freeze frame
FREEZE_MIN_FRAMES    = 15     # frames — minimum length to count as a bout (0.5 s @ 30 fps)

# spatial-entropy grid resolution
ENTROPY_GRID_N       = 10     # 10 × 10 cells

COHORT_MAP = {
    "MA1": ("MA1", "Control"),
    "MA3": ("MA3", "Aspartame"),
    "MA5": ("MA5", "Grapefruit"),           # folder: data/DLCfiltered/Greyfurt/
    "MA7": ("MA7", "Aspartame+Grapefruit"), # folder: data/DLCfiltered/ASP ve Greyfurt/
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_body_center(csv_path: str, likelihood_thresh: float,
                     jump_thresh: float, smooth: int) -> tuple[np.ndarray, np.ndarray]:
    """Load DLC CSV, extract cleaned body_center x/y."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = ["_".join(c[-2:]).strip() for c in df.columns.values]

    x = df["body_center_x"].values.astype(float)
    y = df["body_center_y"].values.astype(float)
    lk = df["body_center_likelihood"].values.astype(float)

    # likelihood filter
    x[lk < likelihood_thresh] = np.nan
    y[lk < likelihood_thresh] = np.nan

    # jump filter
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    x[dist > jump_thresh] = np.nan
    y[dist > jump_thresh] = np.nan

    # rolling-median smoothing (preserves NaN gaps)
    nan_mask = np.isnan(x) | np.isnan(y)
    xs = pd.Series(x).rolling(smooth, center=True, min_periods=1).median().values.copy()
    ys = pd.Series(y).rolling(smooth, center=True, min_periods=1).median().values.copy()
    xs[nan_mask] = np.nan
    ys[nan_mask] = np.nan

    return xs, ys


def load_bouts(bouts_csv: str) -> pd.DataFrame:
    if bouts_csv and os.path.isfile(bouts_csv):
        return pd.read_csv(bouts_csv)
    return pd.DataFrame(columns=["behaviour", "bout", "start_frame",
                                  "end_frame", "start_s", "end_s", "duration_s"])


# ── metric helpers ─────────────────────────────────────────────────────────────

def locomotion_metrics(x: np.ndarray, y: np.ndarray,
                       fps: float) -> dict:
    """Total distance, mean speed, max speed from body_center path."""
    dx = np.diff(x)
    dy = np.diff(y)
    step = np.sqrt(dx ** 2 + dy ** 2)
    valid_step = step[~np.isnan(step)]

    total_dist = float(np.nansum(step))
    speed_px_s = valid_step * fps
    mean_speed = float(np.mean(speed_px_s)) if len(speed_px_s) else float("nan")
    max_speed  = float(np.max(speed_px_s))  if len(speed_px_s) else float("nan")
    return {
        "total_distance_px": round(total_dist, 1),
        "mean_speed_px_s":   round(mean_speed, 2),
        "max_speed_px_s":    round(max_speed,  2),
    }


def thigmotaxis_metrics(x: np.ndarray, y: np.ndarray,
                        inner_zone: tuple) -> dict:
    """% time in center, % in periphery, center zone entry count."""
    ix_min, ix_max, iy_min, iy_max = inner_zone
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() == 0:
        return {"pct_time_center": float("nan"),
                "pct_time_periphery": float("nan"),
                "center_zone_entries": 0}

    xv, yv = x[valid], y[valid]
    in_center = (
        (xv >= ix_min) & (xv <= ix_max)
        & (yv >= iy_min) & (yv <= iy_max)
    )
    pct_center = float(in_center.sum() / len(xv) * 100)

    # entry count: rising edges (False→True) in in_center boolean array
    entries = int(np.sum(np.diff(in_center.astype(int)) == 1))

    return {
        "pct_time_center":    round(pct_center, 2),
        "pct_time_periphery": round(100.0 - pct_center, 2),
        "center_zone_entries": entries,
    }


def freezing_metrics(x: np.ndarray, y: np.ndarray,
                     fps: float,
                     speed_thresh: float = FREEZE_SPEED_THRESH,
                     min_frames: int = FREEZE_MIN_FRAMES) -> dict:
    """Freezing bouts, total freeze time, % session freezing."""
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    speed = np.sqrt(dx ** 2 + dy ** 2) * fps  # px/s per frame

    # candidate freeze frames: speed below threshold or NaN (untracked)
    freeze_candidate = (speed < speed_thresh) & ~np.isnan(speed)

    # merge into bouts (same logic as behavior_detection)
    bouts = []
    in_bout = False
    start = 0
    for i, f in enumerate(freeze_candidate):
        if f and not in_bout:
            start = i
            in_bout = True
        elif not f and in_bout:
            duration = i - start
            if duration >= min_frames:
                bouts.append((start, i - 1))
            in_bout = False
    if in_bout:
        duration = len(freeze_candidate) - start
        if duration >= min_frames:
            bouts.append((start, len(freeze_candidate) - 1))

    total_freeze_frames = sum(e - s + 1 for s, e in bouts)
    total_frames = len(x)
    total_freeze_s = total_freeze_frames / fps
    pct_freeze = total_freeze_frames / total_frames * 100 if total_frames else float("nan")

    return {
        "freeze_bout_count": len(bouts),
        "total_freeze_s":    round(total_freeze_s, 2),
        "pct_time_freeze":   round(pct_freeze, 2),
    }


def spatial_entropy(x: np.ndarray, y: np.ndarray,
                    arena: tuple, n: int = ENTROPY_GRID_N) -> float:
    """
    Shannon entropy of spatial occupancy on an n×n arena grid.
    High entropy = exploratory (visits many cells evenly).
    Low entropy  = stereotypic / thigmotactic (concentrates in few cells).
    Returned in bits; normalised by log2(n*n) so 0–1 range.
    """
    x_min, x_max, y_min, y_max = arena
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 10:
        return float("nan")

    xv = np.clip(x[valid], x_min, x_max)
    yv = np.clip(y[valid], y_min, y_max)

    col = np.floor((xv - x_min) / (x_max - x_min + 1e-9) * n).astype(int).clip(0, n - 1)
    row = np.floor((yv - y_min) / (y_max - y_min + 1e-9) * n).astype(int).clip(0, n - 1)
    cell_idx = row * n + col

    counts = np.bincount(cell_idx, minlength=n * n).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log2(probs))
    H_max = np.log2(n * n)
    return round(float(H / H_max), 4)


def behavior_bout_metrics(bouts_df: pd.DataFrame,
                          session_duration_s: float) -> dict:
    """Per-behavior count + duration from the bouts CSV."""
    out = {}
    for beh, prefix in [("rearing", "rear"), ("grooming", "groom")]:
        sub = bouts_df[bouts_df["behaviour"] == beh]
        count    = len(sub)
        total_s  = float(sub["duration_s"].sum()) if not sub.empty else 0.0
        pct_time = total_s / session_duration_s * 100 if session_duration_s else float("nan")
        out[f"{prefix}_bout_count"] = count
        out[f"{prefix}_total_s"]    = round(total_s, 2)
        out[f"{prefix}_pct_time"]   = round(pct_time, 2)
    return out


# ── inner zone helper ──────────────────────────────────────────────────────────

def auto_inner_zone(arena: tuple, margin: float = DEFAULT_MARGIN) -> tuple:
    x_min, x_max, y_min, y_max = arena
    w = x_max - x_min
    h = y_max - y_min
    return (x_min + margin * w, x_max - margin * w,
            y_min + margin * h, y_max - margin * h)


# ── subject-level pipeline ─────────────────────────────────────────────────────

def compute_subject_metrics(csv_path: str, arena: tuple, inner_zone: tuple,
                             fps: float = DEFAULT_FPS,
                             likelihood: float = DEFAULT_LIKELIHOOD,
                             jump_thresh: float = DEFAULT_JUMP_THRESH,
                             smooth: int = DEFAULT_SMOOTH) -> dict:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    subject_dir = os.path.dirname(csv_path)

    # Try to load bouts CSV from the same folder
    bouts_csv = os.path.join(subject_dir, f"{base}_behavior_bouts.csv")

    x, y = load_body_center(csv_path, likelihood, jump_thresh, smooth)
    bouts_df = load_bouts(bouts_csv)

    n_frames = len(x)
    session_s = n_frames / fps

    # ── subject identity ────────────────────────────────────────────────────
    m = re.search(r"(MA\d+)_(\d+)", base)
    cohort = m.group(1) if m else "unknown"
    run    = m.group(2) if m else "?"
    subject_id = f"{cohort}_{run}"
    group_info = COHORT_MAP.get(cohort, (cohort, "Unknown"))

    row = {
        "subject_id":       subject_id,
        "cohort":           group_info[0],
        "group":            group_info[1],
        "session_duration_s": round(session_s, 1),
        "n_frames":         n_frames,
        "bouts_csv_found":  os.path.isfile(bouts_csv),
    }

    row.update(locomotion_metrics(x, y, fps))
    row.update(thigmotaxis_metrics(x, y, inner_zone))
    row.update(freezing_metrics(x, y, fps))
    row["spatial_entropy_norm"] = spatial_entropy(x, y, arena)
    row.update(behavior_bout_metrics(bouts_df, session_s))

    return row


# ── printing helpers ───────────────────────────────────────────────────────────

def print_subject_report(row: dict) -> None:
    w = 52
    print(f"\n{'='*w}")
    print(f"  OFT Metrics - {row['subject_id']}  ({row['group']})")
    print(f"{'='*w}")
    print(f"  Session: {row['session_duration_s']} s  ({row['n_frames']} frames)")
    print(f"\n  LOCOMOTION (Line Crossing equivalent)")
    print(f"    Total distance   : {row['total_distance_px']:>10.1f} px")
    print(f"    Mean speed       : {row['mean_speed_px_s']:>10.2f} px/s")
    print(f"    Max speed        : {row['max_speed_px_s']:>10.2f} px/s")
    print(f"\n  THIGMOTAXIS / CENTER ZONE")
    print(f"    Center time      : {row['pct_time_center']:>9.2f} %")
    print(f"    Periphery time   : {row['pct_time_periphery']:>9.2f} %")
    print(f"    Center entries   : {row['center_zone_entries']:>5d}")
    print(f"\n  FREEZING (Anxiety)")
    print(f"    Freeze bouts     : {row['freeze_bout_count']:>5d}")
    print(f"    Total freeze     : {row['total_freeze_s']:>9.2f} s")
    print(f"    % time freeze    : {row['pct_time_freeze']:>9.2f} %")
    print(f"\n  SPATIAL ENTROPY (Stereotypy)")
    print(f"    Entropy (0-1)    : {row['spatial_entropy_norm']:>9.4f}")
    print(f"      (0 = stereotypic / wall-only, 1 = uniform exploration)")
    print(f"\n  REARING")
    print(f"    Bouts            : {row['rear_bout_count']:>5d}")
    print(f"    Total time       : {row['rear_total_s']:>9.2f} s"
          f"  ({row['rear_pct_time']:.1f}%)")
    print(f"\n  GROOMING")
    print(f"    Bouts            : {row['groom_bout_count']:>5d}")
    print(f"    Total time       : {row['groom_total_s']:>9.2f} s"
          f"  ({row['groom_pct_time']:.1f}%)")
    print(f"{'='*w}")
    if not row["bouts_csv_found"]:
        print("  [!] Behavior bouts CSV not found — rearing/grooming set to 0.")
        print("      Run behavior_detection.py first for complete metrics.")
    print()


# ── batch mode ─────────────────────────────────────────────────────────────────

def find_subject_csvs(batch_dir: str) -> list[str]:
    """
    Walk batch_dir looking for DLC CSVs named OpenField*.csv
    (one level of subfolders expected).
    """
    csvs = []
    for root, dirs, files in os.walk(batch_dir):
        for f in files:
            if f.startswith("OpenField") and f.endswith(".csv") \
                    and "_behavior" not in f and "_oft_metrics" not in f:
                csvs.append(os.path.join(root, f))
    return sorted(csvs)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute OFT metrics from DLC tracking data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single subject
  python analysis/oft_metrics.py \\
      --csv data/DLCfiltered/OpenFieldMA1_1/OpenFieldMA1_1.csv \\
      --arena 396 776 153 530

  # All subjects — saves combined CSV
  python analysis/oft_metrics.py \\
      --batch-dir data/DLCfiltered \\
      --arena 396 776 153 530 \\
      --out data/oft_metrics_all.csv
        """
    )
    p.add_argument("--csv", help="Path to DLC filtered CSV (single-subject mode)")
    p.add_argument("--batch-dir", dest="batch_dir",
                   help="Parent folder containing subject subfolders (batch mode)")
    p.add_argument("--arena", nargs=4, type=float,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   default=[396, 776, 153, 530],
                   help="Arena bounds in pixels (default: 396 776 153 530)")
    p.add_argument("--inner-zone", nargs=4, type=float, dest="inner_zone",
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="Inner/center zone bounds (auto-calculated from --arena if omitted)")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                   help=f"Center-zone margin fraction (default {DEFAULT_MARGIN})")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help=f"Video frame rate (default {DEFAULT_FPS})")
    p.add_argument("--likelihood", type=float, default=DEFAULT_LIKELIHOOD,
                   help=f"Likelihood filter threshold (default {DEFAULT_LIKELIHOOD})")
    p.add_argument("--jump-thresh", type=float, default=DEFAULT_JUMP_THRESH,
                   dest="jump_thresh",
                   help=f"Max frame-to-frame displacement in px (default {DEFAULT_JUMP_THRESH})")
    p.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH,
                   help=f"Rolling-median window for smoothing (default {DEFAULT_SMOOTH})")
    p.add_argument("--out", help="Output CSV path for batch mode "
                               "(default: data/oft_metrics_all.csv)")
    p.add_argument("--freeze-speed", type=float, default=FREEZE_SPEED_THRESH,
                   dest="freeze_speed",
                   help=f"Speed threshold for freeze detection (default {FREEZE_SPEED_THRESH} px/s)")
    p.add_argument("--freeze-min-frames", type=int, default=FREEZE_MIN_FRAMES,
                   dest="freeze_min_frames",
                   help=f"Min frames to count as freeze bout (default {FREEZE_MIN_FRAMES})")
    return p


def main() -> None:
    args = build_parser().parse_args()

    arena = tuple(args.arena)
    inner_zone = tuple(args.inner_zone) if args.inner_zone else auto_inner_zone(arena, args.margin)
    print(f"Arena      : X {arena[0]:.0f}–{arena[1]:.0f}  Y {arena[2]:.0f}–{arena[3]:.0f}")
    print(f"Inner zone : X {inner_zone[0]:.0f}–{inner_zone[1]:.0f}  Y {inner_zone[2]:.0f}–{inner_zone[3]:.0f}")

    # Monkey-patch the module-level constants for freeze params
    global FREEZE_SPEED_THRESH, FREEZE_MIN_FRAMES
    FREEZE_SPEED_THRESH = args.freeze_speed
    FREEZE_MIN_FRAMES   = args.freeze_min_frames

    common_kw = dict(
        arena=arena, inner_zone=inner_zone,
        fps=args.fps, likelihood=args.likelihood,
        jump_thresh=args.jump_thresh, smooth=args.smooth,
    )

    if args.batch_dir:
        csvs = find_subject_csvs(args.batch_dir)
        if not csvs:
            print(f"[!] No OpenField*.csv files found under {args.batch_dir}")
            sys.exit(1)
        print(f"\nBatch mode: {len(csvs)} subjects found.\n")

        rows = []
        for csv_path in csvs:
            print(f"  Processing {os.path.basename(csv_path)} ...")
            row = compute_subject_metrics(csv_path, **common_kw)
            rows.append(row)
            # per-subject CSV
            out_path = os.path.join(
                os.path.dirname(csv_path),
                os.path.splitext(os.path.basename(csv_path))[0] + "_oft_metrics.csv",
            )
            pd.DataFrame([row]).to_csv(out_path, index=False)
            print(f"    -> {out_path}")

        all_df = pd.DataFrame(rows)

        # Cohort-level summary
        numeric_cols = all_df.select_dtypes(include="number").columns.tolist()
        summary = all_df.groupby("group")[numeric_cols].agg(["mean", "std"]).round(3)

        out_csv = args.out or os.path.join(args.batch_dir, "..", "oft_metrics_all.csv")
        out_csv = os.path.abspath(out_csv)
        all_df.to_csv(out_csv, index=False)
        print(f"\nAll subjects -> {out_csv}")

        # cohort summary
        summary_csv = out_csv.replace(".csv", "_cohort_summary.csv")
        summary.to_csv(summary_csv)
        print(f"Cohort summary -> {summary_csv}")

        print("\n-- Cohort means --------------------------------------------------")
        key_cols = [c for c in numeric_cols if c not in
                    ("n_frames", "session_duration_s", "bouts_csv_found")]
        print(all_df.groupby("group")[key_cols].mean().round(2).to_string())
        print()

    elif args.csv:
        if not os.path.isfile(args.csv):
            print(f"[!] CSV not found: {args.csv}")
            sys.exit(1)
        row = compute_subject_metrics(args.csv, **common_kw)
        print_subject_report(row)

        # Save per-subject CSV
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join(os.path.dirname(args.csv), f"{base}_oft_metrics.csv")
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    else:
        print("[!] Provide --csv (single subject) or --batch-dir (all subjects).")
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
