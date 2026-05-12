"""
run_analysis.py  (Plus/Cross Maze version)
-------------------------------------------
Master pipeline for plus maze analysis.

Steps:
  0. tmaze_metrics.py    — arm entries, alternation, zone occupancy
  1. orbit_plot.py       — zone-colored trajectory
  2. activity_heatmap.py — KDE density heatmap
  3. speed_analysis.py   — locomotion speed (shared, arena-agnostic)

Usage:
  # 1. Get zone coordinates interactively:
  python show_frame_coords.py --video <video.mp4>

  # 2. Run full pipeline with the printed coordinates:
  python run_analysis.py \\
      --csv ../../data/DLCfiltered/control/TmazeMA1_1/TmazeMA1_1.csv \\
      --bottom-arm 526 606 403 717 \\
      --left-arm   259 533 346 402 \\
      --right-arm  605 892 347 402 \\
      --top-arm    526 606   0 345
"""

import argparse
import os
import subprocess
import sys

DEFAULT_CSV = "../../data/DLCfiltered/control/PlusMazeMA1_1/PlusMazeMA1_1.csv"
DEFAULT_FPS = 30

# Absolute directory of this script — used to locate sibling scripts
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def run_command(script, args, description):
    cmd = [sys.executable, script] + args
    print(f"\n{'='*70}\n[*] {description}\n{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"[OK] {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description}\n{e}\n")
        return False


def zone_args(bottom, left, right, top):
    return [
        "--bottom-arm", *[str(v) for v in bottom],
        "--left-arm",   *[str(v) for v in left],
        "--right-arm",  *[str(v) for v in right],
        "--top-arm",    *[str(v) for v in top],
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Plus maze full analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv",         default=DEFAULT_CSV)
    parser.add_argument("--bottom-arm",  nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="bottom_arm")
    parser.add_argument("--left-arm",    nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="left_arm")
    parser.add_argument("--right-arm",   nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="right_arm")
    parser.add_argument("--top-arm",     nargs=4, type=float, required=True,
                        metavar=("XMIN","XMAX","YMIN","YMAX"), dest="top_arm")
    parser.add_argument("--fps",         type=float, default=DEFAULT_FPS)
    parser.add_argument("--likelihood",  type=float, default=0.6)
    parser.add_argument("--jump-thresh", type=float, default=60.0, dest="jump_thresh")
    parser.add_argument("--smooth",      type=int,   default=5)
    parser.add_argument("--cmap",        default="inferno")
    parser.add_argument("--sigma",       type=float, default=15.0)
    parser.add_argument("--skip-metrics", action="store_true", dest="skip_metrics")
    parser.add_argument("--skip-orbit",   action="store_true", dest="skip_orbit")
    parser.add_argument("--skip-heatmap", action="store_true", dest="skip_heatmap")
    parser.add_argument("--skip-speed",   action="store_true", dest="skip_speed")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}"); sys.exit(1)

    bottom = tuple(args.bottom_arm)
    left   = tuple(args.left_arm)
    right  = tuple(args.right_arm)
    top    = tuple(args.top_arm)
    zones  = zone_args(bottom, left, right, top)

    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    common  = [
        "--csv",         args.csv,
        "--out-dir",     csv_dir,
        "--likelihood",  str(args.likelihood),
        "--jump-thresh", str(args.jump_thresh),
        "--smooth",      str(args.smooth),
    ]

    print(f"\n{'='*70}\n[*] PLUS MAZE ANALYSIS PIPELINE\n{'='*70}")
    print(f"[*] CSV: {args.csv}")
    print(f"[*] Bottom: X {bottom[0]:.0f}-{bottom[1]:.0f}  Y {bottom[2]:.0f}-{bottom[3]:.0f}")
    print(f"[*] Left:   X {left[0]:.0f}-{left[1]:.0f}  Y {left[2]:.0f}-{left[3]:.0f}")
    print(f"[*] Right:  X {right[0]:.0f}-{right[1]:.0f}  Y {right[2]:.0f}-{right[3]:.0f}")
    print(f"[*] Top:    X {top[0]:.0f}-{top[1]:.0f}  Y {top[2]:.0f}-{top[3]:.0f}")

    results = {}

    if not args.skip_metrics:
        # tmaze_metrics saves next to CSV automatically; does not accept --out-dir
        metrics_args = [
            "--csv",         args.csv,
            "--likelihood",  str(args.likelihood),
            "--jump-thresh", str(args.jump_thresh),
            "--smooth",      str(args.smooth),
            "--fps",         str(args.fps),
        ]
        results["metrics"] = run_command(
            os.path.join(HERE, "tmaze_metrics.py"),
            metrics_args + zones,
            "Step 0/3: Plus Maze Metrics",
        )
    else:
        results["metrics"] = None

    if not args.skip_orbit:
        results["orbit"] = run_command(
            os.path.join(HERE, "orbit_plot.py"), common + zones,
            "Step 1/3: Zone Trajectory",
        )
    else:
        results["orbit"] = None

    if not args.skip_heatmap:
        results["heatmap"] = run_command(
            os.path.join(HERE, "activity_heatmap.py"),
            common + zones + ["--cmap", args.cmap, "--sigma", str(args.sigma)],
            "Step 2/3: Activity Heatmap",
        )
    else:
        results["heatmap"] = None

    if not args.skip_speed:
        results["speed"] = run_command(
            os.path.join(ROOT, "analysis", "speed_analysis.py"),
            ["--csv", args.csv, "--out-dir", csv_dir,
             "--likelihood", str(args.likelihood),
             "--jump-thresh", str(args.jump_thresh)],
            "Step 3/3: Speed Analysis",
        )
    else:
        results["speed"] = None

    print(f"\n{'='*70}\n[*] SUMMARY\n{'='*70}")
    info = {
        "metrics": ("Plus maze metrics",  "*_plus_maze_metrics.csv"),
        "orbit":   ("Trajectory plots",   "*_plus_maze_orbit.png, *_plus_maze_bodyparts.png"),
        "heatmap": ("Heatmaps",           "*_heatmap_kde.png, *_heatmap_histogram.png"),
        "speed":   ("Speed analysis",     "*_speed.csv, *_speed.png"),
    }
    for key, (label, outputs) in info.items():
        v = results.get(key)
        status = "[OK]  " if v is True else "[skip]" if v is None else "[FAIL]"
        print(f"{status} {label}  ->  {outputs}" if v is True else f"{status} {label}")

    if any(v is False for v in results.values()):
        sys.exit(1)
    print(f"\n[done] All outputs -> {csv_dir}\n{'='*70}\n")


if __name__ == "__main__":
    main()
