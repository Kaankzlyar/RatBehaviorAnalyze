"""
Batch analysis runner for Kare synthesized subjects (MA2 / MA4 / MA6 / MA8).

Runs the full pipeline for every synthesized DLC CSV produced by
src/synthesize_keypoints.py using the exact Kare arena pixel coordinates.
No video files are required.

Pipeline per subject:
  1. run_analysis.py  -> orbit, heatmaps, behavior detection
  2. oft_metrics.py   -> per-subject _oft_metrics.csv

Usage (from project root or from analysis/):
    python analysis/run_kare_batch.py
    python analysis/run_kare_batch.py --skip-behavior
    python analysis/run_kare_batch.py --dry-run
    python analysis/run_kare_batch.py --subjects MA2 MA8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Arena & inner-zone (confirmed from video annotation) — single source of
# truth in src/anxiety/config.py so spatial_rearing & predict_anxiety_v2
# stay consistent.
# ---------------------------------------------------------------------------
from src.anxiety.config import ARENA as _ARENA, INNER_ZONE as _INNER_ZONE, FPS as _FPS

ARENA      = tuple(int(v) for v in _ARENA)
INNER_ZONE = tuple(int(v) for v in _INNER_ZONE)
FPS        = int(_FPS)

COHORT_TO_GROUP = {
    2: "control",
    4: "ASP",
    6: "Greyfurt",
    8: "ASP ve Greyfurt",
}

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT         = ANALYSIS_DIR.parent
DLC_DIR      = ROOT / "data" / "DLCfiltered"


# ---------------------------------------------------------------------------

def find_kare_csvs(subject_filter: set[str] | None = None) -> list[tuple[int, int, Path]]:
    results = []
    for cohort_num, group in COHORT_TO_GROUP.items():
        if subject_filter and f"MA{cohort_num}" not in subject_filter:
            continue
        group_dir = DLC_DIR / group
        if not group_dir.exists():
            print(f"[warn] group dir not found: {group_dir.relative_to(ROOT)}")
            continue
        for sub_dir in sorted(group_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            name = sub_dir.name
            if not name.startswith(f"OpenFieldMA{cohort_num}_"):
                continue
            run_num = int(name.split("_")[-1])
            csv = sub_dir / f"{name}.csv"
            if csv.exists():
                results.append((cohort_num, run_num, csv))
            else:
                print(f"[warn] CSV missing: {csv.relative_to(ROOT)}")
    return sorted(results)


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> bool:
    print(f"  CMD: {' '.join(cmd)}")
    if dry_run:
        print("  [dry-run] skipped")
        return True
    result = subprocess.run(cmd, cwd=str(cwd))
    return result.returncode == 0


def run_subject(cohort_num: int, run_num: int, csv: Path,
                skip_behavior: bool, dry_run: bool) -> bool:
    rel_csv = os.path.relpath(csv, ANALYSIS_DIR)
    arena_args   = ["--arena",      *[str(x) for x in ARENA]]
    zone_args    = ["--inner-zone", *[str(x) for x in INNER_ZONE]]

    print(f"\n{'='*68}")
    print(f"  MA{cohort_num}-{run_num}  |  {csv.parent.relative_to(ROOT)}")
    print(f"{'='*68}")

    # Step 1: visualisations + behavior detection
    cmd1 = [
        sys.executable, "run_analysis.py",
        "--csv", rel_csv,
        *arena_args, *zone_args,
        "--fps", str(FPS),
    ]
    if skip_behavior:
        cmd1.append("--skip-behavior")
    if not _run(cmd1, ANALYSIS_DIR, dry_run):
        print(f"  [ERROR] run_analysis.py failed for MA{cohort_num}-{run_num}")
        return False

    # Step 2: quantitative OFT metrics
    cmd2 = [
        sys.executable, "oft_metrics.py",
        "--csv", rel_csv,
        *arena_args, *zone_args,
        "--fps", str(FPS),
    ]
    if not _run(cmd2, ANALYSIS_DIR, dry_run):
        print(f"  [ERROR] oft_metrics.py failed for MA{cohort_num}-{run_num}")
        return False

    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-behavior", action="store_true",
                   help="Skip behavior detection (rearing & grooming)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--subjects", nargs="+", default=None,
                   metavar="MAn",
                   help="Process only these cohorts, e.g. --subjects MA2 MA8")
    args = p.parse_args()

    subject_filter = {s.upper() for s in args.subjects} if args.subjects else None
    subjects = find_kare_csvs(subject_filter)

    if not subjects:
        print("[kare_batch] No subjects found — run src/synthesize_keypoints.py first.")
        sys.exit(1)

    print(f"[kare_batch] {len(subjects)} subjects to process")
    print(f"  arena      : {ARENA}")
    print(f"  inner-zone : {INNER_ZONE}")
    for c, r, csv in subjects:
        print(f"  MA{c}-{r}  {csv.relative_to(ROOT)}")

    ok, fail = 0, 0
    for cohort_num, run_num, csv in subjects:
        if run_subject(cohort_num, run_num, csv, args.skip_behavior, args.dry_run):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*68}")
    print(f"[kare_batch] done — {ok} OK, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
