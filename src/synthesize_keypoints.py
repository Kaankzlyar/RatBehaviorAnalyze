"""
Build full 10-keypoint DLC-format CSVs for centroid-only subjects in the
professor's _res.mat archive (MA2 / MA4 / MA6 / MA8 cohorts), so that the
existing OFT pipeline can run on them end-to-end.

Method (per source subject):
  1. Parse cohort number from filename (MA{k}-{n}_res.mat)
  2. Pick a same-group DLC subject as pose reference (deterministic seed)
  3. Compute reference offsets relative to body_center: kp[t] − body_center[t]
  4. Time-warp those offsets (linear interp) to the target length
  5. keypoint[t] = mat_centroid[t] + warped_offset[t]
  6. Likelihood column copied from reference (after warp); downgraded to
     0.4 for any frame where the source centroid was interpolated through
     a gap

Centroid gaps (xc==0 in the MAT convention) are linearly interpolated up
to 10 frames so the resulting CSV has no NaN entries.

Usage
-----
    python -m src.synthesize_keypoints
    python -m src.synthesize_keypoints --seed 7

Output
------
    data/DLCfiltered/<group>/<subject>/<subject>.csv
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
OUT_DIR_DEFAULT = DLC_DIR

COHORT_NUM_TO_GROUP = {
    1: "control", 2: "control",
    3: "ASP", 4: "ASP",
    5: "Greyfurt", 6: "Greyfurt",
    7: "ASP ve Greyfurt", 8: "ASP ve Greyfurt",
}

# MA1/3/5/7 already have direct DLC subjects under data/DLCfiltered/;
# only the missing-half cohorts are completed from centroid trajectories.
COHORTS_TO_BUILD = {2, 4, 6, 8}

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


def load_reference_dlc(group: str, ref_subject: str) -> pd.DataFrame:
    path = DLC_DIR / group / ref_subject / f"{ref_subject}.csv"
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(
        [(bp, coord) for _, bp, coord in df.columns],
        names=["bodyparts", "coords"],
    )
    return df


def find_reference_subjects(group: str) -> list[str]:
    group_dir = DLC_DIR / group
    if not group_dir.exists():
        return []
    return sorted(p.name for p in group_dir.iterdir()
                  if p.is_dir() and p.name.startswith("OpenField"))


def fill_short_gaps(arr: np.ndarray, max_gap: int = 10
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation across NaN gaps up to ``max_gap``. Returns
    (filled_array, was_filled_mask)."""
    s = pd.Series(arr)
    was_nan = s.isna().to_numpy()
    s = s.interpolate(method="linear", limit=max_gap, limit_direction="both")
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


# ── core ─────────────────────────────────────────────────────────────────────

def build_one(mat_path: Path, group: str, ref_subject: str) -> tuple[pd.DataFrame, dict]:
    xc_raw, yc_raw = load_mat_centroid(mat_path)
    xc, xc_filled = fill_short_gaps(xc_raw, max_gap=10)
    yc, yc_filled = fill_short_gaps(yc_raw, max_gap=10)
    centroid_low_quality = xc_filled | yc_filled

    n_target = len(xc)
    ref = load_reference_dlc(group, ref_subject)
    bc_x = ref[("body_center", "x")].to_numpy(dtype=float)
    bc_y = ref[("body_center", "y")].to_numpy(dtype=float)

    cols_data: dict[tuple[str, str], np.ndarray] = {}
    for kp in KEYPOINTS:
        kp_lik_full = ref[(kp, "likelihood")].to_numpy(dtype=float)
        if kp == "body_center":
            cols_data[(kp, "x")] = xc
            cols_data[(kp, "y")] = yc
            warped_lik = time_warp_1d(kp_lik_full, n_target)
        else:
            kp_x = ref[(kp, "x")].to_numpy(dtype=float)
            kp_y = ref[(kp, "y")].to_numpy(dtype=float)
            offset_x = kp_x - bc_x
            offset_y = kp_y - bc_y
            warped_off_x = time_warp_1d(offset_x, n_target)
            warped_off_y = time_warp_1d(offset_y, n_target)
            warped_lik = time_warp_1d(kp_lik_full, n_target)
            cols_data[(kp, "x")] = xc + warped_off_x
            cols_data[(kp, "y")] = yc + warped_off_y

        warped_lik = np.where(centroid_low_quality, np.minimum(warped_lik, 0.4), warped_lik)
        cols_data[(kp, "likelihood")] = warped_lik

    multi_cols = pd.MultiIndex.from_tuples(
        [(SCORER, kp, c) for kp in KEYPOINTS for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    arr = np.column_stack(
        [cols_data[(kp, c)] for kp in KEYPOINTS for c in ("x", "y", "likelihood")]
    )
    df = pd.DataFrame(arr, columns=multi_cols)
    df.index.name = "scorer"

    meta = {
        "n_frames": n_target,
        "centroid_gap_frames": int(centroid_low_quality.sum()),
    }
    return df, meta


# ── orchestration ────────────────────────────────────────────────────────────

def discover_mat_files() -> list[Path]:
    """Return de-duplicated MAT files (KareFinal copies skipped if the
    cohort folder already contains the same MAk-N file)."""
    seen: dict[str, Path] = {}
    for mat in KARE_BASE.glob("*/*.mat"):
        if mat.parent.name == "KareFinal":
            seen.setdefault(mat.stem, mat)
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
    print(f"[load] {len(mats)} source files (deduplicated)")

    refs_by_group = {g: find_reference_subjects(g) for g in set(COHORT_NUM_TO_GROUP.values())}
    for g, subs in refs_by_group.items():
        print(f"  references[{g}] = {subs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    built = 0

    for mat_path in mats:
        parsed = parse_mat_name(mat_path.stem)
        if parsed is None:
            continue
        cohort_num, run = parsed
        if cohort_num not in COHORTS_TO_BUILD:
            continue
        group = COHORT_NUM_TO_GROUP[cohort_num]
        refs = refs_by_group.get(group, [])
        if not refs:
            print(f"  skip MA{cohort_num}-{run}: no reference subjects in {group}")
            continue
        ref = str(rng.choice(refs))

        subject = f"OpenFieldMA{cohort_num}_{run}"
        out_path = args.out_dir / group / subject / f"{subject}.csv"

        if out_path.exists() and not args.overwrite:
            print(f"  exists  {out_path.relative_to(ROOT)}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df, meta = build_one(mat_path, group, ref)
        df.to_csv(out_path)
        built += 1
        print(f"  MA{cohort_num}-{run:>2}  ->  {out_path.relative_to(ROOT)}  "
              f"ref={ref}  n={meta['n_frames']}  "
              f"interp_frames={meta['centroid_gap_frames']}")

    print(f"\n[done] {built} file(s) written under {args.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
