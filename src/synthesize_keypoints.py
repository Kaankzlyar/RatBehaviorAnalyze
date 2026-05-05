"""
Generate DLC-format keypoint CSVs for centroid-only MAT subjects.

Method
------
For each MAT subject we have only the HSV-blob centroid (xc, yc) over time.
To produce a 10-keypoint DLC-format CSV we:

  1. Parse cohort number from filename (e.g. MA3-2_res.mat -> cohort 3)
  2. Map cohort -> treatment group (Control, ASP, Greyfurt, ASP ve Greyfurt)
  3. Pick a cohort-matched DLC subject as "donor" (deterministic by seed)
  4. Compute donor keypoint offsets relative to donor's body_center
  5. Time-warp donor offsets (linear interp) to MAT length
  6. synth_keypoint[t] = mat_centroid[t] + warped_donor_offset[t]
  7. Likelihood column: copy donor's (after time-warp) — preserves realistic
     occlusion / low-confidence patterns
  8. body_center is replaced with the MAT centroid itself

Centroid gaps (xc==0 in MAT convention) are interpolated up to 10 frames;
likelihood for interpolated frames is set to 0.4.

Output preserves the exact 3-row header schema of real DLC CSVs so all
downstream scripts (behavior_detection, oft_metrics, etc.) accept it.

Usage
-----
    python -m src.synthesize_keypoints
    python -m src.synthesize_keypoints --seed 7

Output dir
----------
    data/synthetic_DLCfiltered/<group>/<subject>/<subject>.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
DLC_DIR = ROOT / "data" / "DLCfiltered"
KARE_BASE = DLC_DIR / "Kare"
OUT_DIR_DEFAULT = ROOT / "data" / "synthetic_DLCfiltered"

COHORT_NUM_TO_GROUP = {
    1: "control", 2: "control",
    3: "ASP", 4: "ASP",
    5: "Greyfurt", 6: "Greyfurt",
    7: "ASP ve Greyfurt", 8: "ASP ve Greyfurt",
}

KEYPOINTS = [
    "nose", "head", "left_ear", "right_ear", "body_center",
    "left_forepaw", "right_forepaw", "left_hindpaw", "right_hindpaw",
    "tail_base",
]
SCORER = "DLC_Resnet50_rat_behavior_openfieldApr1shuffle1_snapshot_best-180"


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_mat_name(stem: str) -> tuple[int, int] | None:
    m = re.match(r"MA(\d+)-(\d+)_res$", stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_mat_centroid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = loadmat(path)
    xc = np.asarray(m["xc"]).ravel().astype(float)
    yc = np.asarray(m["yc"]).ravel().astype(float)
    xc[xc == 0] = np.nan
    yc[yc == 0] = np.nan
    return xc, yc


def load_donor_dlc(group: str, donor_subject: str) -> pd.DataFrame:
    path = DLC_DIR / group / donor_subject / f"{donor_subject}.csv"
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(
        [(bp, coord) for _, bp, coord in df.columns],
        names=["bodyparts", "coords"],
    )
    return df


def find_donor_subjects(group: str) -> list[str]:
    group_dir = DLC_DIR / group
    if not group_dir.exists():
        return []
    return sorted(p.name for p in group_dir.iterdir()
                  if p.is_dir() and p.name.startswith("OpenField"))


def fill_short_gaps(arr: np.ndarray, max_gap: int = 10,
                    fallback: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation across NaN gaps up to max_gap. Returns (filled, was_filled_mask)."""
    s = pd.Series(arr)
    was_nan = s.isna().to_numpy()
    s = s.interpolate(method="linear", limit=max_gap, limit_direction="both")
    if fallback is None:
        fallback = float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else 0.0
    s = s.fillna(fallback)
    return s.to_numpy(), was_nan & ~np.isnan(s.to_numpy())


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


# ── synthesis core ───────────────────────────────────────────────────────────

def synthesize_one(mat_path: Path, group: str, donor_subject: str,
                   ) -> tuple[pd.DataFrame, dict]:
    xc_raw, yc_raw = load_mat_centroid(mat_path)

    # interpolate centroid gaps (max 10 frames) to keep CSV NaN-free
    xc, xc_filled = fill_short_gaps(xc_raw, max_gap=10)
    yc, yc_filled = fill_short_gaps(yc_raw, max_gap=10)
    centroid_low_quality = xc_filled | yc_filled

    n_target = len(xc)
    donor = load_donor_dlc(group, donor_subject)

    bc_x = donor[("body_center", "x")].to_numpy(dtype=float)
    bc_y = donor[("body_center", "y")].to_numpy(dtype=float)

    cols_data: dict[tuple[str, str], np.ndarray] = {}
    for kp in KEYPOINTS:
        kp_lik_full = donor[(kp, "likelihood")].to_numpy(dtype=float)
        if kp == "body_center":
            cols_data[(kp, "x")] = xc
            cols_data[(kp, "y")] = yc
            warped_lik = time_warp_1d(kp_lik_full, n_target)
        else:
            kp_x = donor[(kp, "x")].to_numpy(dtype=float)
            kp_y = donor[(kp, "y")].to_numpy(dtype=float)
            offset_x = kp_x - bc_x
            offset_y = kp_y - bc_y
            warped_off_x = time_warp_1d(offset_x, n_target)
            warped_off_y = time_warp_1d(offset_y, n_target)
            warped_lik = time_warp_1d(kp_lik_full, n_target)
            cols_data[(kp, "x")] = xc + warped_off_x
            cols_data[(kp, "y")] = yc + warped_off_y

        # downgrade likelihood where centroid was interpolated
        warped_lik = np.where(centroid_low_quality, np.minimum(warped_lik, 0.4), warped_lik)
        cols_data[(kp, "likelihood")] = warped_lik

    # build the 3-row header DataFrame matching real DLC CSV format
    multi_cols = pd.MultiIndex.from_tuples(
        [(SCORER, kp, c) for kp in KEYPOINTS for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    arr = np.column_stack(
        [cols_data[(kp, c)] for kp in KEYPOINTS for c in ("x", "y", "likelihood")]
    )
    df = pd.DataFrame(arr, columns=multi_cols)
    df.index.name = "scorer"  # matches real DLC index name (column header in row 0)

    meta = {
        "donor_subject": donor_subject,
        "donor_n_frames": len(donor),
        "synth_n_frames": n_target,
        "centroid_gap_frames": int(centroid_low_quality.sum()),
    }
    return df, meta


# ── orchestration ────────────────────────────────────────────────────────────

def discover_mat_files() -> list[Path]:
    """Return de-duplicated MAT files (KareFinal duplicates skipped if cohort
    folder has the same MAk-N file)."""
    seen: dict[str, Path] = {}
    for mat in KARE_BASE.glob("*/*.mat"):
        if mat.parent.name == "KareFinal":
            # only use KareFinal entries that aren't represented elsewhere
            key = mat.stem
            seen.setdefault(key, mat)
        else:
            seen[mat.stem] = mat
    return sorted(seen.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate even if output exists")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    mats = discover_mat_files()
    print(f"[load] {len(mats)} MAT subjects (deduplicated)")

    donors_by_group = {g: find_donor_subjects(g) for g in set(COHORT_NUM_TO_GROUP.values())}
    for g, subs in donors_by_group.items():
        print(f"  donors[{g}] = {subs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []

    for mat_path in mats:
        parsed = parse_mat_name(mat_path.stem)
        if parsed is None:
            print(f"  skip {mat_path.name}: cannot parse")
            continue
        cohort_num, run = parsed
        group = COHORT_NUM_TO_GROUP.get(cohort_num)
        if group is None:
            print(f"  skip MA{cohort_num}-{run}: unknown cohort")
            continue
        donors = donors_by_group.get(group, [])
        if not donors:
            print(f"  skip MA{cohort_num}-{run}: no donor subjects in {group}")
            continue
        donor = str(rng.choice(donors))

        synth_subject = f"OpenFieldMA{cohort_num}_{run}"
        out_path = args.out_dir / group / synth_subject / f"{synth_subject}.csv"

        if out_path.exists() and not args.overwrite:
            print(f"  exists  {out_path.relative_to(ROOT)}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df, meta = synthesize_one(mat_path, group, donor)
        df.to_csv(out_path)
        print(f"  MA{cohort_num}-{run:>2}  ->  {out_path.relative_to(ROOT)}  "
              f"donor={donor}  n={meta['synth_n_frames']}  "
              f"interp_frames={meta['centroid_gap_frames']}")

        log_rows.append({
            "mat_subject": f"MA{cohort_num}-{run}",
            "synth_subject": synth_subject,
            "group": group,
            "donor_subject": donor,
            "n_frames": meta["synth_n_frames"],
            "centroid_gap_frames": meta["centroid_gap_frames"],
        })

    log_df = pd.DataFrame(log_rows)
    log_path = args.out_dir / "synthesis_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[done] {len(log_df)} synthetic CSVs.  log -> {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
