"""
tmaze_metrics.py  (Plus/Cross Maze version)
--------------------------------------------
Compute quantitative plus maze metrics from DLC tracking data.

This maze has 4 arms meeting at a central junction:
          [TOP ARM]
              ||
[LEFT ARM] ==JUNCTION== [RIGHT ARM]
              ||
         [BOTTOM ARM]

Metrics computed
----------------
  Zone occupancy  : pct_time_bottom, pct_time_left, pct_time_right,
                    pct_time_top, pct_time_junction
  Arm entries     : bottom_entries, left_entries, right_entries, top_entries,
                    total_entries
  Alternation     : successive_alternation_pct
                    (each entry differs from the previous one)
                    tetrad_alternation_pct
                    (each group of 4 consecutive entries visits all 4 arms)
  Perseveration   : perseveration_count (same arm twice in a row)
  Arm preference  : most_visited_arm, arm_preference_index
  Locomotion      : total_distance_px, mean_speed_px_s

Zone definitions (all in pixels, format: xmin xmax ymin ymax):
  --bottom-arm, --left-arm, --right-arm, --top-arm

Usage — single subject
----------------------
  python tmaze_metrics.py \\
      --csv ../../data/DLCfiltered/control/TmazeMA1_1/TmazeMA1_1.csv \\
      --bottom-arm 526 606 403 717 \\
      --left-arm   259 533 346 402 \\
      --right-arm  605 892 347 402 \\
      --top-arm    526 606   0 345

Usage — batch
-------------
  python tmaze_metrics.py \\
      --batch-dir ../../data/DLCfiltered \\
      --bottom-arm 526 606 403 717 \\
      --left-arm   259 533 346 402 \\
      --right-arm  605 892 347 402 \\
      --top-arm    526 606   0 345 \\
      --out ../../data/tmaze_metrics_all.csv
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_FPS         = 30
DEFAULT_LIKELIHOOD  = 0.6
DEFAULT_JUMP_THRESH = 60.0
DEFAULT_SMOOTH      = 5

COHORT_MAP = {
    "MA1": "Control",
    "MA3": "Aspartame",
    "MA5": "Grapefruit",
    "MA7": "ASP+Greyfurt",
}

ARMS = ["bottom_arm", "left_arm", "right_arm", "top_arm"]


# ── zone helpers ───────────────────────────────────────────────────────────────

def in_zone(x, y, bounds):
    x0, x1, y0, y1 = bounds
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def assign_zones(x, y, zones: dict) -> np.ndarray:
    labels = np.full(len(x), "junction", dtype=object)
    valid  = ~(np.isnan(x) | np.isnan(y))
    labels[~valid] = "unknown"
    for name, bounds in zones.items():
        mask = valid & in_zone(x, y, bounds)
        labels[mask] = name
    return labels


# ── data loading ───────────────────────────────────────────────────────────────

def load_body_center(csv_path, likelihood_thresh, jump_thresh=60.0, smooth=5):
    df  = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    x   = df["body_center"]["x"].values.astype(float)
    y   = df["body_center"]["y"].values.astype(float)
    lkh = df["body_center"]["likelihood"].values.astype(float)

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
    return xs, ys


# ── arm entry detection ────────────────────────────────────────────────────────

def detect_arm_entries(labels: np.ndarray, fps: float) -> dict:
    """
    Detect arm entries (transitions into any of the 4 arms).
    An entry requires >= MIN_FRAMES consecutive frames in the arm.
    """
    MIN_ARM_FRAMES = max(3, int(fps * 0.1))

    entries  = []
    sequence = []   # ordered list of arm names

    i = 0
    n = len(labels)
    while i < n:
        lbl = labels[i]
        if lbl in ARMS:
            start = i
            arm   = lbl
            while i < n and labels[i] == arm:
                i += 1
            end      = i
            duration = end - start
            if duration >= MIN_ARM_FRAMES:
                entries.append({
                    "arm":         arm,
                    "entry_frame": start,
                    "exit_frame":  end,
                    "duration_s":  round(duration / fps, 3),
                })
                sequence.append(arm)
        else:
            i += 1

    total   = len(sequence)
    counts  = {arm: sequence.count(arm) for arm in ARMS}

    # Successive alternation: each entry ≠ previous entry
    succ_alt   = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
    succ_pers  = sum(1 for a, b in zip(sequence, sequence[1:]) if a == b)
    succ_rate  = succ_alt / (total - 1) * 100 if total > 1 else float("nan")
    pers_rate  = succ_pers / (total - 1) * 100 if total > 1 else float("nan")

    # Tetrad alternation: groups of 4 consecutive entries all different
    tetrad_possible = total - 3
    if tetrad_possible > 0:
        tetrad_alts = sum(
            1 for i in range(tetrad_possible)
            if len(set(sequence[i:i+4])) == 4
        )
        tetrad_rate = tetrad_alts / tetrad_possible * 100
    else:
        tetrad_alts = 0
        tetrad_rate = float("nan")

    most_visited = max(counts, key=counts.get) if total > 0 else "none"

    # Arm preference index: (max_entries - min_entries) / total  (0=uniform, 1=exclusive)
    if total > 0:
        api = (max(counts.values()) - min(counts.values())) / total
    else:
        api = float("nan")

    return {
        "total_entries":             total,
        "bottom_entries":            counts["bottom_arm"],
        "left_entries":              counts["left_arm"],
        "right_entries":             counts["right_arm"],
        "top_entries":               counts["top_arm"],
        "most_visited_arm":          most_visited,
        "arm_preference_index":      round(api, 4) if not np.isnan(api) else float("nan"),
        "successive_alternation_pct": round(succ_rate, 2) if not np.isnan(succ_rate) else float("nan"),
        "perseveration_count":       succ_pers,
        "perseveration_rate_pct":    round(pers_rate, 2) if not np.isnan(pers_rate) else float("nan"),
        "tetrad_alternation_pct":    round(tetrad_rate, 2) if not np.isnan(tetrad_rate) else float("nan"),
        "entry_sequence":            "->".join(
                                         a.replace("_arm", "")[0].upper()
                                         for a in sequence
                                     ),
        "_entries_list": entries,
    }


# ── subject info ───────────────────────────────────────────────────────────────

def parse_subject(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    m = re.search(r"(MA\d+)[_-](\d+)", name, re.IGNORECASE)
    if m:
        cohort_id = m.group(1).upper()
        session   = m.group(2)
        return {"subject_id": name, "cohort_id": cohort_id,
                "cohort": COHORT_MAP.get(cohort_id, cohort_id), "session": session}
    return {"subject_id": name, "cohort_id": "unknown",
            "cohort": "unknown", "session": "1"}


# ── core ───────────────────────────────────────────────────────────────────────

def compute_metrics(csv_path, zones, fps=DEFAULT_FPS,
                    likelihood_thresh=DEFAULT_LIKELIHOOD,
                    jump_thresh=DEFAULT_JUMP_THRESH,
                    smooth=DEFAULT_SMOOTH) -> dict:
    info = parse_subject(csv_path)
    print(f"\n  Subject: {info['subject_id']}  cohort: {info['cohort']}")

    x, y = load_body_center(csv_path, likelihood_thresh, jump_thresh, smooth)
    n_frames = len(x)
    valid    = ~(np.isnan(x) | np.isnan(y))
    n_valid  = int(valid.sum())
    dur_s    = n_frames / fps

    labels   = assign_zones(x, y, zones)

    valid_labels = labels[valid]
    n_v          = max(len(valid_labels), 1)

    zone_labels = ARMS + ["junction"]
    pct = {z: round((valid_labels == z).sum() / n_v * 100, 2) for z in zone_labels}
    time_s = {z: round(pct[z] / 100 * dur_s, 2) for z in zone_labels}

    arm_stats = detect_arm_entries(labels, fps)

    # Speed
    dx     = np.diff(x, prepend=np.nan)
    dy     = np.diff(y, prepend=np.nan)
    speed  = np.sqrt(dx**2 + dy**2) * fps
    speed[~valid] = np.nan
    mean_speed = float(np.nanmean(speed)) if n_valid > 1 else float("nan")

    dxv = np.diff(x[valid])
    dyv = np.diff(y[valid])
    total_dist = float(np.nansum(np.sqrt(dxv**2 + dyv**2)))

    row = {
        **info,
        "n_frames":           n_frames,
        "n_valid_frames":     n_valid,
        "session_duration_s": round(dur_s, 2),
        # occupancy
        "pct_time_bottom":    pct["bottom_arm"],
        "pct_time_left":      pct["left_arm"],
        "pct_time_right":     pct["right_arm"],
        "pct_time_top":       pct["top_arm"],
        "pct_time_junction":  pct["junction"],
        "time_bottom_s":      time_s["bottom_arm"],
        "time_left_s":        time_s["left_arm"],
        "time_right_s":       time_s["right_arm"],
        "time_top_s":         time_s["top_arm"],
        # entries
        "total_entries":      arm_stats["total_entries"],
        "bottom_entries":     arm_stats["bottom_entries"],
        "left_entries":       arm_stats["left_entries"],
        "right_entries":      arm_stats["right_entries"],
        "top_entries":        arm_stats["top_entries"],
        "most_visited_arm":   arm_stats["most_visited_arm"],
        "arm_preference_index": arm_stats["arm_preference_index"],
        # alternation
        "successive_alternation_pct": arm_stats["successive_alternation_pct"],
        "tetrad_alternation_pct":     arm_stats["tetrad_alternation_pct"],
        "perseveration_count":        arm_stats["perseveration_count"],
        "perseveration_rate_pct":     arm_stats["perseveration_rate_pct"],
        "entry_sequence":             arm_stats["entry_sequence"],
        # locomotion
        "mean_speed_px_s":    round(mean_speed, 2),
        "total_distance_px":  round(total_dist, 1),
    }

    # Print summary
    print(f"    Duration: {dur_s:.1f}s  |  valid: {n_valid}/{n_frames}")
    print(f"    Zone time — B:{pct['bottom_arm']:.0f}%  L:{pct['left_arm']:.0f}%  "
          f"R:{pct['right_arm']:.0f}%  T:{pct['top_arm']:.0f}%  "
          f"junc:{pct['junction']:.0f}%")
    print(f"    Entries — total:{arm_stats['total_entries']}  "
          f"B:{arm_stats['bottom_entries']}  L:{arm_stats['left_entries']}  "
          f"R:{arm_stats['right_entries']}  T:{arm_stats['top_entries']}")
    if not np.isnan(arm_stats["successive_alternation_pct"]):
        print(f"    Successive alternation: {arm_stats['successive_alternation_pct']:.1f}%  "
              f"Tetrad: {arm_stats['tetrad_alternation_pct']:.1f}%"
              if not np.isnan(arm_stats["tetrad_alternation_pct"])
              else f"    Successive alternation: {arm_stats['successive_alternation_pct']:.1f}%")
    if arm_stats["entry_sequence"]:
        seq = arm_stats["entry_sequence"]
        print(f"    Sequence: {seq[:60]}{'...' if len(seq) > 60 else ''}")

    return row


def save_per_subject(row, csv_path, zones=None):
    out_path = csv_path.replace(".csv", "_plus_maze_metrics.csv")
    save_row = {k: v for k, v in row.items() if not k.startswith("_")}
    pd.DataFrame([save_row]).to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    if zones is not None:
        json_path = csv_path.replace(".csv", "_arm_coords.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({k: list(v) for k, v in zones.items()}, fh, indent=2)
        print(f"  Saved -> {json_path}")

    return out_path


def find_tmaze_csvs(batch_dir):
    found = []
    for root, _, files in os.walk(batch_dir):
        for f in sorted(files):
            if (f.endswith(".csv")
                    and not any(f.endswith(s) for s in [
                        "_plus_maze_metrics.csv", "_speed.csv", "_speed_summary.csv",
                        "_behavior_bouts.csv", "_behavior_frames.csv",
                    ])
                    and "PlusMaze" in root):
                found.append(os.path.join(root, f))
    return found


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute plus maze metrics from DLC CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv",        help="Single subject DLC CSV")
    parser.add_argument("--batch-dir",  help="Parent directory to scan", dest="batch_dir")
    parser.add_argument("--out",        default="../../data/plus_maze_metrics_all.csv")
    parser.add_argument("--bottom-arm", nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="bottom_arm")
    parser.add_argument("--left-arm",   nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="left_arm")
    parser.add_argument("--right-arm",  nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="right_arm")
    parser.add_argument("--top-arm",    nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="top_arm")
    parser.add_argument("--fps",        type=float, default=DEFAULT_FPS)
    parser.add_argument("--likelihood", type=float, default=DEFAULT_LIKELIHOOD)
    parser.add_argument("--jump-thresh",type=float, default=DEFAULT_JUMP_THRESH, dest="jump_thresh")
    parser.add_argument("--smooth",     type=int,   default=DEFAULT_SMOOTH)
    return parser.parse_args()


def main():
    args  = parse_args()
    zones = {
        "bottom_arm": tuple(args.bottom_arm),
        "left_arm":   tuple(args.left_arm),
        "right_arm":  tuple(args.right_arm),
        "top_arm":    tuple(args.top_arm),
    }

    kw = dict(fps=args.fps, likelihood_thresh=args.likelihood,
              jump_thresh=args.jump_thresh, smooth=args.smooth)

    if args.csv:
        if not os.path.isfile(args.csv):
            print(f"ERROR: {args.csv} not found"); sys.exit(1)
        row = compute_metrics(args.csv, zones, **kw)
        save_per_subject(row, args.csv, zones=zones)

    elif args.batch_dir:
        csvs = find_tmaze_csvs(args.batch_dir)
        if not csvs:
            print(f"No Tmaze CSVs found under: {args.batch_dir}"); sys.exit(1)
        print(f"Found {len(csvs)} Tmaze CSVs")
        rows = []
        for csv_path in csvs:
            try:
                row = compute_metrics(csv_path, zones, **kw)
                save_per_subject(row, csv_path, zones=zones)
                rows.append({k: v for k, v in row.items() if not k.startswith("_")})
            except Exception as e:
                print(f"  ERROR {csv_path}: {e}")
        if rows:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"\nBatch output -> {args.out}  ({len(rows)} subjects)")
    else:
        print("ERROR: provide --csv or --batch-dir"); sys.exit(1)


if __name__ == "__main__":
    main()
