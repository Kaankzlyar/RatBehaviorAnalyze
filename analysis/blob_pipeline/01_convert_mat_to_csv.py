"""
analysis/blob_pipeline/01_convert_mat_to_csv.py
-----------------------------------------------
Step 01 of the Strategy A pipeline (24-sample anxiety_level + group classifier).

Two responsibilities, run in this order:

1. **Coordinate sanity check** — verify that the MATLAB blob pipeline's
   ``xc/yc`` share the same image coordinates as the DLC pose pipeline's
   ``body_center``.  Compares MA1/3/5/7 paired subjects (where both
   ``_res.mat`` AND DLC CSV exist) and reports per-axis offset.

   - GREEN  (max |offset| < 20 px)  → fixed arena 396 776 153 530 is safe
   - YELLOW (20 ≤ |offset| < 50 px) → recommend offset transform
   - RED    (|offset| ≥ 50 px)      → per-subject auto-arena required

2. **Conversion** (only with ``--apply``, gated by verdict) — emit
   DLC-style CSVs for MA2/4/6/8 with ``body_center`` populated from
   xc/yc and the other 9 bodyparts left as NaN, so existing batch tools
   (`oft_metrics.py` etc.) can ingest them without modification.

Outputs
-------
  data/blob_pipeline/_arena_check.csv                 (always)
  data/blob_pipeline/csv_converted/{cohort_folder}/
      OpenField{Cohort}_{R}/OpenField{Cohort}_{R}.csv (only with --apply)

Usage
-----
  pip install scipy numpy pandas
  python analysis/blob_pipeline/01_convert_mat_to_csv.py
  python analysis/blob_pipeline/01_convert_mat_to_csv.py --apply
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.io import loadmat

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
DLCFILT   = ROOT / "data" / "DLCfiltered"
OUT       = ROOT / "data" / "blob_pipeline"
CONVERTED = OUT / "csv_converted"

# ── Cohort layout ──────────────────────────────────────────────────────────────
COHORT_FOLDER = {
    "MA1": "control",        "MA2": "control",
    "MA3": "ASP",            "MA4": "ASP",
    "MA5": "Greyfurt",       "MA6": "Greyfurt",
    "MA7": "ASP ve Greyfurt", "MA8": "ASP ve Greyfurt",
}

# ── DLC CSV header layout (mirrors existing DLC files) ────────────────────────
SCORER    = "blob_pipeline_v1"
BODYPARTS = ["nose", "head", "left_ear", "right_ear", "body_center",
             "left_forepaw", "right_forepaw", "left_hindpaw",
             "right_hindpaw", "tail_base"]

# ── Verdict thresholds (pixels) ───────────────────────────────────────────────
GREEN_OFFSET  = 20
YELLOW_OFFSET = 50


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_subject(mat_path: pathlib.Path) -> tuple[str, str]:
    stem = mat_path.stem.replace("_res", "")
    cohort, run = stem.split("-")
    return cohort, run


def load_xy_from_mat(mat: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    m = loadmat(str(mat))
    xc = np.asarray(m["xc"]).flatten().astype(float)
    yc = np.asarray(m["yc"]).flatten().astype(float)
    return xc, yc


def load_xy_from_dlc(dlc_csv: pathlib.Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not dlc_csv.exists():
        return None
    df = pd.read_csv(dlc_csv, header=[0, 1, 2], index_col=0)
    df.columns = ["_".join(c[-2:]).strip() for c in df.columns.values]
    return df["body_center_x"].values, df["body_center_y"].values


# Canonical cohort folders — anything outside these (e.g. KareFinal/) is
# treated as a duplicate or non-canonical variant and skipped.  The cohort
# folders are the source of truth after commit 982d9c3 reorganised them.
CANONICAL_FOLDERS = ["control", "ASP", "Greyfurt", "ASP ve Greyfurt"]


def find_all_mats() -> list[pathlib.Path]:
    mats: list[pathlib.Path] = []
    for folder in CANONICAL_FOLDERS:
        d = DLCFILT / "Kare" / folder
        if d.exists():
            mats.extend(sorted(d.glob("MA*-*_res.mat")))
    return mats


# Tracking quality threshold (pixels) — if blob xc OR yc 1-99 percentile
# range is below this, the tracking likely failed and the subject should
# be flagged for manual review.
MIN_TRACKING_RANGE = 100


def percentile_range(arr: np.ndarray) -> tuple[float, float]:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 1)), float(np.percentile(valid, 99))


def write_dlc_csv(out_path: pathlib.Path, xc: np.ndarray, yc: np.ndarray) -> None:
    """Write a DLC-style CSV: 3 header rows + body_center populated."""
    n = len(xc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bp_idx = BODYPARTS.index("body_center")

    lines = []
    lines.append("scorer," + ",".join([SCORER] * (len(BODYPARTS) * 3)))
    lines.append("bodyparts," + ",".join(bp for bp in BODYPARTS for _ in range(3)))
    lines.append("coords," + ",".join(["x", "y", "likelihood"] * len(BODYPARTS)))

    for i in range(n):
        cells: list[str] = [str(i)]
        for j in range(len(BODYPARTS)):
            if j == bp_idx:
                x_str = f"{xc[i]:.4f}" if not np.isnan(xc[i]) else ""
                y_str = f"{yc[i]:.4f}" if not np.isnan(yc[i]) else ""
                cells.extend([x_str, y_str, "1.0"])
            else:
                # other bodyparts: NaN x/y, likelihood 0 → likelihood filter drops them
                cells.extend(["", "", "0"])
        lines.append(",".join(cells))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--apply", action="store_true",
                    help="Convert MA2/4/6/8 _res.mat to DLC-style CSV "
                         "(only if verdict is GREEN; use --force to override)")
    ap.add_argument("--force", action="store_true",
                    help="Apply conversion even if verdict is YELLOW or RED")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # ── 1. Walk every _res.mat, collect ranges + paired offsets ───────────────
    rows: list[dict] = []
    paired_offsets_x: list[float] = []
    paired_offsets_y: list[float] = []

    for mat in find_all_mats():
        cohort, run = parse_subject(mat)
        try:
            xc, yc = load_xy_from_mat(mat)
        except Exception as e:
            print(f"  [skip] {mat.name}: {e}")
            continue

        bx_low, bx_high = percentile_range(xc)
        by_low, by_high = percentile_range(yc)

        folder   = COHORT_FOLDER[cohort]
        dlc_path = (DLCFILT / folder /
                    f"OpenField{cohort}_{run}" /
                    f"OpenField{cohort}_{run}.csv")
        dlc_xy = load_xy_from_dlc(dlc_path)

        dlc_x_low = dlc_x_high = dlc_y_low = dlc_y_high = None
        offset_x  = offset_y  = None
        if dlc_xy is not None:
            dx, dy = dlc_xy
            dx_clean = dx[~np.isnan(dx)]
            dy_clean = dy[~np.isnan(dy)]
            if len(dx_clean) > 100 and len(dy_clean) > 100:
                dlc_x_low, dlc_x_high = percentile_range(dx_clean)
                dlc_y_low, dlc_y_high = percentile_range(dy_clean)
                offset_x = float(np.nanmean(xc) - np.nanmean(dx_clean))
                offset_y = float(np.nanmean(yc) - np.nanmean(dy_clean))
                paired_offsets_x.append(offset_x)
                paired_offsets_y.append(offset_y)

        x_range = bx_high - bx_low
        y_range = by_high - by_low
        tracking_ok = (x_range >= MIN_TRACKING_RANGE
                       and y_range >= MIN_TRACKING_RANGE)

        rows.append({
            "subject":      f"{cohort}_{run}",
            "cohort":       cohort,
            "source_folder": COHORT_FOLDER[cohort],
            "n_frames":     len(xc),
            "blob_x_low":   bx_low,  "blob_x_high":  bx_high,
            "blob_y_low":   by_low,  "blob_y_high":  by_high,
            "blob_x_range": x_range,
            "blob_y_range": y_range,
            "tracking_ok":  tracking_ok,
            "dlc_x_low":    dlc_x_low,  "dlc_x_high": dlc_x_high,
            "dlc_y_low":    dlc_y_low,  "dlc_y_high": dlc_y_high,
            "offset_x":     offset_x,
            "offset_y":     offset_y,
        })

    df = pd.DataFrame(rows).sort_values("subject")
    out_csv = OUT / "_arena_check.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nArena check report: {out_csv.relative_to(ROOT)}\n")

    # ── 2. Per-subject report ─────────────────────────────────────────────────
    print(f"{'subject':<10} {'frames':>6}  {'blob x range':<13}  {'blob y range':<13}  "
          f"{'dlc x range':<13}  {'dlc y range':<13}  {'Δx':>6}  {'Δy':>6}  {'track':<6}")
    print("-" * 104)
    bad_tracking: list[str] = []
    for r in df.itertuples():
        bx = f"{r.blob_x_low:>4.0f}-{r.blob_x_high:<4.0f}"
        by = f"{r.blob_y_low:>4.0f}-{r.blob_y_high:<4.0f}"
        if r.dlc_x_low is None or pd.isna(r.dlc_x_low):
            dx_str, dy_str, off_x, off_y = "(no DLC)", "(no DLC)", "—", "—"
        else:
            dx_str = f"{r.dlc_x_low:>4.0f}-{r.dlc_x_high:<4.0f}"
            dy_str = f"{r.dlc_y_low:>4.0f}-{r.dlc_y_high:<4.0f}"
            off_x  = f"{r.offset_x:+5.1f}"
            off_y  = f"{r.offset_y:+5.1f}"
        track = "OK" if r.tracking_ok else "BAD"
        if not r.tracking_ok:
            bad_tracking.append(r.subject)
        print(f"{r.subject:<10} {r.n_frames:>6}  {bx:<13}  {by:<13}  "
              f"{dx_str:<13}  {dy_str:<13}  {off_x:>6}  {off_y:>6}  {track:<6}")

    if bad_tracking:
        print()
        print(f"[!] Bad tracking detected (xc OR yc range < {MIN_TRACKING_RANGE} px): "
              f"{', '.join(bad_tracking)}")
        print("    These subjects will be SKIPPED during conversion.")

    # ── 3. Verdict ────────────────────────────────────────────────────────────
    print()
    if not paired_offsets_x:
        print("[!] No paired subjects (MA1/3/5/7 with both _res.mat AND DLC CSV) found.")
        print("    Cannot evaluate coordinate alignment.  Aborting.")
        return

    offsets_x = np.array(paired_offsets_x)
    offsets_y = np.array(paired_offsets_y)
    mean_dx   = float(np.mean(offsets_x))
    mean_dy   = float(np.mean(offsets_y))
    worst     = float(max(np.max(np.abs(offsets_x)), np.max(np.abs(offsets_y))))

    print(f"Paired subjects (blob ↔ DLC): {len(paired_offsets_x)}")
    print(f"  Mean offset:   Δx = {mean_dx:+.1f} px,  Δy = {mean_dy:+.1f} px")
    print(f"  Worst |offset|: {worst:.1f} px")
    print()

    if worst < GREEN_OFFSET:
        verdict = "GREEN"
        msg = (f"  [OK] Coordinate systems are aligned (worst |offset| {worst:.1f} px < {GREEN_OFFSET}).\n"
               f"       Use fixed arena `--arena 396 776 153 530` for MA2/4/6/8.")
    elif worst < YELLOW_OFFSET:
        verdict = "YELLOW"
        msg = (f"  [WARN] Systematic shift detected (worst |offset| {worst:.1f} px in "
               f"[{GREEN_OFFSET}, {YELLOW_OFFSET}] range).\n"
               f"         Recommend offset transform on MA2/4/6/8:\n"
               f"           xc_corrected = xc − {mean_dx:.1f}\n"
               f"           yc_corrected = yc − {mean_dy:.1f}\n"
               f"         Then use fixed arena.  Step 02 will need this transform applied.")
    else:
        verdict = "RED"
        msg = (f"  [FAIL] Major coordinate divergence (worst |offset| {worst:.1f} px ≥ {YELLOW_OFFSET}).\n"
               f"         Fixed arena will not work — per-subject auto-arena required in step 02.")

    print(f"Verdict: {verdict}")
    print(msg)
    print()

    # ── 4. Conversion (only with --apply) ─────────────────────────────────────
    if not args.apply:
        print("Run with --apply to convert MA2/4/6/8 _res.mat → DLC-style CSV.")
        return

    if verdict == "RED" and not args.force:
        print("[!] Verdict RED — refusing to convert. Pass --force to override.")
        return
    if verdict == "YELLOW" and not args.force:
        print("[!] Verdict YELLOW — offset transform should be applied first.")
        print("    Pass --force to convert with raw xc/yc anyway.")
        return

    print("Converting MA2/4/6/8 _res.mat → DLC-style CSV...")
    n_converted = 0
    n_skipped_bad = 0
    bad_set = set(bad_tracking)
    for mat in find_all_mats():
        cohort, run = parse_subject(mat)
        if cohort not in {"MA2", "MA4", "MA6", "MA8"}:
            continue
        subject = f"{cohort}_{run}"
        if subject in bad_set:
            print(f"  [skip] {mat.name}: bad tracking, manual review needed")
            n_skipped_bad += 1
            continue
        try:
            xc, yc = load_xy_from_mat(mat)
        except Exception as e:
            print(f"  [skip] {mat.name}: {e}")
            continue
        folder  = COHORT_FOLDER[cohort]
        out_dir = CONVERTED / folder / f"OpenField{cohort}_{run}"
        out_csv = out_dir / f"OpenField{cohort}_{run}.csv"
        write_dlc_csv(out_csv, xc, yc)
        n_converted += 1
        print(f"  {mat.name}  →  {out_csv.relative_to(ROOT)}  ({len(xc)} frames)")

    print(f"\nConverted {n_converted} subjects ({n_skipped_bad} skipped due to bad tracking).")


if __name__ == "__main__":
    main()
