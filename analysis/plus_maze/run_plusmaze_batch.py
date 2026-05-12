"""
run_plusmaze_batch.py
---------------------
Run the full plus-maze analysis pipeline for every subject across all
cohort folders (control / ASP / Greyfurt / ASP ve Greyfurt).

Zone coordinates are read from zones.json next to this script.
Each subject is skipped if its *_plus_maze_metrics.csv already exists
(unless --overwrite is passed).

Usage:
    python run_plusmaze_batch.py
    python run_plusmaze_batch.py --dry-run
    python run_plusmaze_batch.py --overwrite
    python run_plusmaze_batch.py --skip-metrics --skip-speed
    python run_plusmaze_batch.py --cohorts control ASP
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent.parent
DLC_DIR  = ROOT / "data" / "DLCfiltered"
ZONES_FILE = HERE / "zones.json"

COHORTS = ["control", "ASP", "Greyfurt", "ASP ve Greyfurt"]
FPS     = 30


def load_zones() -> dict:
    with open(ZONES_FILE) as f:
        return json.load(f)


def zone_args(zones: dict) -> list[str]:
    return [
        "--bottom-arm", *[str(v) for v in zones["bottom_arm"]],
        "--left-arm",   *[str(v) for v in zones["left_arm"]],
        "--right-arm",  *[str(v) for v in zones["right_arm"]],
        "--top-arm",    *[str(v) for v in zones["top_arm"]],
    ]


def find_subjects(cohort_filter: list[str] | None = None) -> list[tuple[str, Path]]:
    results = []
    for cohort in COHORTS:
        if cohort_filter and cohort not in cohort_filter:
            continue
        folder = DLC_DIR / cohort
        if not folder.exists():
            print(f"[warn] not found: {folder.relative_to(ROOT)}")
            continue
        for sub_dir in sorted(folder.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("PlusMaze"):
                continue
            csv = sub_dir / f"{sub_dir.name}.csv"
            if csv.exists():
                results.append((cohort, csv))
            else:
                print(f"[warn] CSV missing: {sub_dir.relative_to(ROOT)}")
    return results


def already_done(csv: Path) -> bool:
    return any(csv.parent.glob("*_plus_maze_metrics.csv"))


def run_subject(cohort: str, csv: Path, zones: dict,
                skip_metrics: bool, skip_orbit: bool,
                skip_heatmap: bool, skip_speed: bool,
                dry_run: bool) -> bool:
    rel = csv.relative_to(ROOT)
    print(f"\n{'='*68}")
    print(f"  {cohort}  |  {csv.parent.name}")
    print(f"{'='*68}")

    cmd = [
        sys.executable, str(HERE / "run_analysis.py"),
        "--csv", str(csv),
        "--fps", str(FPS),
        *zone_args(zones),
    ]
    if skip_metrics:
        cmd.append("--skip-metrics")
    if skip_orbit:
        cmd.append("--skip-orbit")
    if skip_heatmap:
        cmd.append("--skip-heatmap")
    if skip_speed:
        cmd.append("--skip-speed")

    print(f"  CMD: python run_analysis.py --csv {rel} ...")
    if dry_run:
        print("  [dry-run] skipped")
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohorts",       nargs="+", default=None,
                   help="Only process these cohorts, e.g. --cohorts control ASP")
    p.add_argument("--overwrite",     action="store_true",
                   help="Re-run even if metrics CSV already exists")
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--skip-metrics",  action="store_true", dest="skip_metrics")
    p.add_argument("--skip-orbit",    action="store_true", dest="skip_orbit")
    p.add_argument("--skip-heatmap",  action="store_true", dest="skip_heatmap")
    p.add_argument("--skip-speed",    action="store_true", dest="skip_speed")
    args = p.parse_args()

    zones    = load_zones()
    subjects = find_subjects(args.cohorts)

    if not subjects:
        print("No subjects found.")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects")
    print(f"Zones: bottom={zones['bottom_arm']}  left={zones['left_arm']}")
    print(f"       right={zones['right_arm']}  top={zones['top_arm']}")
    if args.dry_run:
        print("--- DRY RUN ---")

    ok = fail = skipped = 0
    for cohort, csv in subjects:
        if not args.overwrite and already_done(csv):
            print(f"  SKIP {csv.parent.name}: already done (--overwrite to re-run)")
            skipped += 1
            continue
        if run_subject(cohort, csv, zones,
                       args.skip_metrics, args.skip_orbit,
                       args.skip_heatmap, args.skip_speed,
                       args.dry_run):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*68}")
    print(f"Done — {ok} OK, {fail} failed, {skipped} skipped")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
