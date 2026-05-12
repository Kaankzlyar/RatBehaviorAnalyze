"""
mat_to_dlc_csv.py
-----------------
Convert MATLAB _res.mat tracking files to DLC-format CSV files
matching the exact format of DLC-filtered CSVs in data/DLCfiltered/.

Keypoints for non-body_center bodyparts are synthesized from a reference
DLC subject in the same cohort group using offset-warping:

  offset[t] = ref_keypoint[t] - ref_body_center[t]
  synthesized[t] = mat_centroid[t] + warped_offset[t]

Reference subjects are real DLC-tracked PlusMaze CSVs already in the
output tree (MA1/3/5/7 sessions 1-3). Likelihood is downgraded to 0.4
for any frame where the centroid was interpolated through a gap (xc==0).

Cohort routing (MAx number -> subfolder):
    MA1, MA2  ->  control
    MA3, MA4  ->  ASP
    MA5, MA6  ->  Greyfurt
    MA7, MA8  ->  ASP ve Greyfurt

Output layout:
    data/DLCfiltered/control/PlusMazeMA1_4/PlusMazeMA1_4.csv
    data/DLCfiltered/ASP/PlusMazeMA3_4/PlusMazeMA3_4.csv
    ...

Usage:
    python mat_to_dlc_csv.py
    python mat_to_dlc_csv.py --dry-run
    python mat_to_dlc_csv.py --overwrite
    python mat_to_dlc_csv.py --seed 7
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import savgol_filter

SCORER = "DLC_Resnet50_rat_behavior_tmazeApr1shuffle1_snapshot_best-210"

KEYPOINTS = [
    "nose", "head", "left_ear", "right_ear", "body_center",
    "left_forepaw", "right_forepaw", "left_hindpaw", "right_hindpaw", "tail_base",
]

COHORT_MAP = {
    1: "control", 2: "control",
    3: "ASP",     4: "ASP",
    5: "Greyfurt", 6: "Greyfurt",
    7: "ASP ve Greyfurt", 8: "ASP ve Greyfurt",
}

# Real DLC-tracked sessions — never overwrite these
DLC_PROTECTED = {(1,1),(1,2),(1,3), (3,1),(3,2),(3,3),
                 (5,1),(5,2),(5,3), (7,1),(7,2),(7,3)}

DEFAULT_MAT_DIR = "../../data/DLCfiltered/PlusMaze"
DEFAULT_OUT_BASE = "../../data/DLCfiltered"


# ── helpers ───────────────────────────────────────────────────────────────────

def mat_to_stem(mat_filename: str) -> str:
    """MA1-1_res.mat  ->  PlusMazeMA1_1"""
    name = os.path.splitext(mat_filename)[0]
    name = re.sub(r"_res$", "", name)
    name = name.replace("-", "_")
    return f"PlusMaze{name}"


def ma_cohort_session(mat_filename: str) -> tuple[int, int]:
    """MA3-4_res.mat  ->  (3, 4)"""
    m = re.match(r"MA(\d+)-(\d+)", mat_filename)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def fill_short_gaps(arr: np.ndarray, max_gap: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation across NaN gaps up to max_gap.
    Returns (filled_array, was_interpolated_mask)."""
    s = pd.Series(arr)
    was_nan = s.isna().to_numpy()
    s = s.interpolate(method="linear", limit=max_gap, limit_direction="both")
    fallback = float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else 0.0
    s = s.fillna(fallback)
    return s.to_numpy(), was_nan & ~np.isnan(s.to_numpy())


# window=9 poly=3 calibrated against DLC filtering: MA1-1 raw 1.96 -> 0.78 px/frame
# (DLC filtered target = 0.82). Preserves real speed differences between sessions.
SMOOTH_WINDOW    = 9
SMOOTH_POLY      = 3
OUTLIER_JUMP_PX  = 30  # frames with jumps larger than this are interpolated first


def remove_outlier_jumps(x: np.ndarray, y: np.ndarray,
                         threshold: float = OUTLIER_JUMP_PX,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Mark frames where the displacement from the previous good frame exceeds
    threshold as outliers, then linearly interpolate over them."""
    x, y = x.copy(), y.copy()
    n = len(x)
    bad = np.zeros(n, dtype=bool)

    # First pass: flag consecutive large jumps
    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    bad[1:] = dist > threshold

    # Iterative: once bad frames are removed, re-evaluate from last good frame
    # (handles bursts of bad frames by not propagating the bad position)
    for _ in range(3):
        xs, ys = x.copy(), y.copy()
        xs[bad] = np.nan
        ys[bad] = np.nan
        xs = pd.Series(xs).interpolate("linear", limit_direction="both").to_numpy()
        ys = pd.Series(ys).interpolate("linear", limit_direction="both").to_numpy()
        dist2 = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        bad[1:] |= dist2 > threshold

    x[bad] = np.nan
    y[bad] = np.nan
    x = pd.Series(x).interpolate("linear", limit_direction="both").to_numpy()
    y = pd.Series(y).interpolate("linear", limit_direction="both").to_numpy()
    return x, y


def smooth_centroid(arr: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing with mirror padding to avoid edge artifacts."""
    return savgol_filter(arr, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLY, mode="mirror")


def time_warp_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    src_len = len(arr)
    if src_len == target_len:
        return arr.copy()
    src_t = np.linspace(0.0, 1.0, src_len)
    tgt_t = np.linspace(0.0, 1.0, target_len)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return np.full(target_len, np.nan)
    return np.interp(tgt_t, src_t[valid], arr[valid])


# ── reference subject discovery ───────────────────────────────────────────────

def has_full_bodyparts(csv_path: Path) -> bool:
    """Return True if the CSV has all keypoints (real DLC, not body_center-only)."""
    try:
        df = pd.read_csv(csv_path, nrows=3, header=None)
        bp_row = df.iloc[1].tolist()
        return "nose" in bp_row
    except Exception:
        return False


def find_reference_subjects(group: str, out_base: Path) -> list[str]:
    group_dir = out_base / group
    if not group_dir.exists():
        return []
    refs = []
    for sub_dir in sorted(group_dir.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("PlusMaze"):
            continue
        csv = sub_dir / f"{sub_dir.name}.csv"
        if csv.exists() and has_full_bodyparts(csv):
            refs.append(sub_dir.name)
    return refs


def load_reference_dlc(group: str, ref_subject: str, out_base: Path) -> pd.DataFrame:
    path = out_base / group / ref_subject / f"{ref_subject}.csv"
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(
        [(bp, coord) for _, bp, coord in df.columns],
        names=["bodyparts", "coords"],
    )
    return df


# ── synthesis ─────────────────────────────────────────────────────────────────

def synthesize_keypoints(
    xc: np.ndarray, yc: np.ndarray,
    centroid_low_quality: np.ndarray,
    ref: pd.DataFrame,
) -> dict[tuple[str, str], np.ndarray]:
    n_target = len(xc)
    bc_x = ref[("body_center", "x")].to_numpy(dtype=float)
    bc_y = ref[("body_center", "y")].to_numpy(dtype=float)

    cols: dict[tuple[str, str], np.ndarray] = {}
    for kp in KEYPOINTS:
        kp_lik = ref[(kp, "likelihood")].to_numpy(dtype=float)
        warped_lik = time_warp_1d(kp_lik, n_target)

        if kp == "body_center":
            cols[(kp, "x")] = xc
            cols[(kp, "y")] = yc
        else:
            offset_x = ref[(kp, "x")].to_numpy(dtype=float) - bc_x
            offset_y = ref[(kp, "y")].to_numpy(dtype=float) - bc_y
            cols[(kp, "x")] = xc + time_warp_1d(offset_x, n_target)
            cols[(kp, "y")] = yc + time_warp_1d(offset_y, n_target)

        cols[(kp, "likelihood")] = np.where(
            centroid_low_quality, np.minimum(warped_lik, 0.4), warped_lik
        )

    return cols


def write_dlc_csv(cols: dict[tuple[str, str], np.ndarray], out_path: Path) -> None:
    n_bp = len(KEYPOINTS)
    scorer_row = "scorer,"    + ",".join([SCORER] * (n_bp * 3))
    bp_row     = "bodyparts," + ",".join(kp for kp in KEYPOINTS for _ in range(3))
    coord_row  = "coords,"    + ",".join(["x", "y", "likelihood"] * n_bp)

    n = len(cols[(KEYPOINTS[0], "x")])
    with open(out_path, "w", newline="") as fh:
        fh.write(scorer_row + "\n")
        fh.write(bp_row     + "\n")
        fh.write(coord_row  + "\n")
        for i in range(n):
            vals = []
            for kp in KEYPOINTS:
                vals += [
                    f"{cols[(kp, 'x')][i]:.7f}",
                    f"{cols[(kp, 'y')][i]:.7f}",
                    f"{cols[(kp, 'likelihood')][i]:.7f}",
                ]
            fh.write(str(i) + "," + ",".join(vals) + "\n")


# ── per-file orchestration ────────────────────────────────────────────────────

def convert_one(
    mat_path: str, out_base: Path,
    rng: np.random.Generator,
    dry_run: bool, overwrite: bool,
) -> bool:
    mat_file = os.path.basename(mat_path)
    if not mat_file.endswith("_res.mat"):
        return False

    stem              = mat_to_stem(mat_file)
    ma_num, ma_ses    = ma_cohort_session(mat_file)
    cohort            = COHORT_MAP.get(ma_num, "unknown")
    subdir            = out_base / cohort / stem
    out_csv           = subdir / f"{stem}.csv"

    if (ma_num, ma_ses) in DLC_PROTECTED:
        if not dry_run:
            print(f"  PROTECT {mat_file}: real DLC session, skipping")
        return False

    if dry_run:
        tag = " [EXISTS]" if out_csv.exists() else ""
        print(f"  {mat_file}  ->  {cohort}/{stem}/{stem}.csv{tag}")
        return True

    if out_csv.exists() and not overwrite:
        print(f"  SKIP {mat_file}: already exists (--overwrite to replace)")
        return False

    mat = sio.loadmat(mat_path)
    if "xc" not in mat or "yc" not in mat:
        print(f"  SKIP {mat_file}: no xc/yc variables")
        return False

    xc_raw = mat["xc"].flatten().astype(float)
    yc_raw = mat["yc"].flatten().astype(float)
    xc_raw[xc_raw == 0] = np.nan
    yc_raw[yc_raw == 0] = np.nan

    xc, xc_interp = fill_short_gaps(xc_raw)
    yc, yc_interp = fill_short_gaps(yc_raw)
    centroid_low = xc_interp | yc_interp
    xc, yc = remove_outlier_jumps(xc, yc)
    xc = smooth_centroid(xc)
    yc = smooth_centroid(yc)

    refs = find_reference_subjects(cohort, out_base)
    if not refs:
        print(f"  SKIP {mat_file}: no full-bodypart reference subjects in {cohort}/")
        return False

    ref_name = str(rng.choice(refs))
    ref = load_reference_dlc(cohort, ref_name, out_base)

    subdir.mkdir(parents=True, exist_ok=True)
    cols = synthesize_keypoints(xc, yc, centroid_low, ref)
    write_dlc_csv(cols, out_csv)
    interp_pct = 100 * centroid_low.sum() / len(xc)
    print(f"  {mat_file}  ->  {cohort}/{stem}/{stem}.csv  "
          f"ref={ref_name}  n={len(xc)}  interp={interp_pct:.1f}%")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert _res.mat to DLC CSV via keypoint synthesis")
    parser.add_argument("--mat-dir",   default=DEFAULT_MAT_DIR, dest="mat_dir")
    parser.add_argument("--out-base",  default=DEFAULT_OUT_BASE, dest="out_base")
    parser.add_argument("--seed",      default=42, type=int)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mat_dir  = Path(os.path.abspath(args.mat_dir))
    out_base = Path(os.path.abspath(args.out_base))
    rng      = np.random.default_rng(args.seed)

    if not mat_dir.is_dir():
        raise FileNotFoundError(f"Not found: {mat_dir}")

    mat_files = sorted(f for f in os.listdir(mat_dir) if f.endswith("_res.mat"))
    if not mat_files:
        print(f"No _res.mat files in: {mat_dir}")
        return

    print(f"Found {len(mat_files)} .mat files  ->  {out_base}")
    if args.dry_run:
        print("--- DRY RUN ---")

    converted = 0
    for mat_file in mat_files:
        if convert_one(str(mat_dir / mat_file), out_base, rng, args.dry_run, args.overwrite):
            converted += 1

    label = "Would convert" if args.dry_run else "Converted"
    print(f"\n{label}: {converted}/{len(mat_files)}")


if __name__ == "__main__":
    main()
