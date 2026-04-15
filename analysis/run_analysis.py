"""
run_analysis.py
---------------
Master automation script for complete open-field analysis.

Workflow:
  1. User defines arena boundaries (show_frame_coords.py or manual)
  2. Script automatically calculates inner-zone from arena bounds
  3. Generates all visualizations in sequence:
     - orbit_plot.py (trajectories)
     - activity_heatmap.py (KDE heatmap - primary)
     - bodypart_heatmaps.py (per-bodypart grid)

Usage (RECOMMENDED):
    # Interactive: define boundaries with tool
    python show_frame_coords.py --video ../data/DLCfiltered/OpenFieldMA1_2.mp4
    # Copy the --arena and --inner-zone from output, then:
    python run_analysis.py --arena 396 776 153 530 --inner-zone 422 747 177 502

Or (QUICK - auto-generate inner-zone):
    python run_analysis.py --arena 396 776 153 530

The script will automatically calculate inner-zone as:
    inner_zone = arena_bounds_with_20%_margin
"""

import argparse
import subprocess
import sys
import os

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULT_CSV = "../data/DLCfiltered/OpenFieldMA1_2.csv"
DEFAULT_MARGIN = 0.20  # 20% margin = thigmotaxis zone


def compute_inner_zone(arena: tuple, margin: float = 0.20) -> tuple:
    """Calculate inner zone from arena bounds + margin fraction.

    Args:
        arena: (x_min, x_max, y_min, y_max)
        margin: fraction of width/height to use as border (default 0.20 = 20%)

    Returns:
        (ix_min, ix_max, iy_min, iy_max)
    """
    x_min, x_max, y_min, y_max = arena
    w = x_max - x_min
    h = y_max - y_min
    ix_min = x_min + margin * w
    ix_max = x_max - margin * w
    iy_min = y_min + margin * h
    iy_max = y_max - margin * h
    return (ix_min, ix_max, iy_min, iy_max)


def run_command(script: str, args: list, description: str) -> bool:
    """Execute a Python script with args.

    Args:
        script: Python script filename (e.g., 'orbit_plot.py')
        args: List of command arguments
        description: Human-readable description for logging

    Returns:
        True if successful, False otherwise
    """
    cmd = [sys.executable, script] + args
    print(f"\n{'='*70}")
    print(f"📊 {description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✅ {description} — COMPLETE\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} — FAILED")
        print(f"Error: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automated open-field analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With manual inner-zone definition
  python run_analysis.py --arena 396 776 153 530 --inner-zone 422 747 177 502

  # Auto-calculate inner-zone (20%% margin)
  python run_analysis.py --arena 396 776 153 530

  # Custom margin and filtering
  python run_analysis.py --arena 396 776 153 530 --margin 0.25 --likelihood 0.8
        """
    )

    parser.add_argument("--arena", nargs=4, type=float, required=True,
                       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                       help="Arena boundaries (get from show_frame_coords.py)")
    parser.add_argument("--inner-zone", nargs=4, type=float,
                       metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                       help="Inner zone bounds (optional; auto-calculated if not provided)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                       help=f"Border margin fraction if auto-calculating inner-zone (default {DEFAULT_MARGIN})")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                       help=f"Path to DLC CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--likelihood", type=float, default=0.6,
                       help="Likelihood threshold (default: 0.6)")
    parser.add_argument("--jump-thresh", type=float, default=60.0,
                       help="Jump threshold in px (default: 60)")
    parser.add_argument("--smooth", type=int, default=5,
                       help="Rolling median window (default: 5)")
    parser.add_argument("--cmap", default="inferno",
                       help="KDE colormap (default: inferno)")
    parser.add_argument("--skip-orbit", action="store_true",
                       help="Skip orbit_plot.py (trajectory visualization)")
    parser.add_argument("--skip-heatmap", action="store_true",
                       help="Skip activity_heatmap.py (KDE heatmap)")
    parser.add_argument("--skip-bodypart", action="store_true",
                       help="Skip bodypart_heatmaps.py (per-bodypart grid)")

    args = parser.parse_args()

    # Parse arena bounds
    arena = tuple(args.arena)

    # Calculate or use provided inner-zone
    if args.inner_zone:
        inner_zone = tuple(args.inner_zone)
        print(f"Using provided inner-zone: {inner_zone}")
    else:
        inner_zone = compute_inner_zone(arena, args.margin)
        print(f"Auto-calculated inner-zone (margin={args.margin*100:.0f}%): {inner_zone}")

    # Verify CSV exists
    if not os.path.isfile(args.csv):
        print(f"❌ ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    # Common arguments for all scripts
    common_args = [
        "--csv", args.csv,
        "--arena", *[str(x) for x in arena],
        "--inner-zone", *[str(x) for x in inner_zone],
        "--likelihood", str(args.likelihood),
        "--jump-thresh", str(args.jump_thresh),
        "--smooth", str(args.smooth),
    ]

    # Track success
    results = {}

    print(f"\n{'='*70}")
    print(f"🔧 OPEN-FIELD ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"\n📍 Arena bounds:  X {arena[0]:.0f}–{arena[1]:.0f}  Y {arena[2]:.0f}–{arena[3]:.0f}")
    print(f"📍 Inner zone:    X {inner_zone[0]:.0f}–{inner_zone[1]:.0f}  Y {inner_zone[2]:.0f}–{inner_zone[3]:.0f}")
    print(f"📊 CSV: {args.csv}")
    print(f"⚙️  Likelihood: {args.likelihood}  |  Jump-thresh: {args.jump_thresh}px  |  Smooth: {args.smooth}")

    # ── Step 1: Orbit trajectories ────────────────────────────────────────────
    if not args.skip_orbit:
        results["orbit"] = run_command(
            "orbit_plot.py",
            common_args,
            "Step 1/3: Orbit Trajectories & Thigmotaxis"
        )
    else:
        print("\n⊘ Skipping orbit_plot.py")
        results["orbit"] = None

    # ── Step 2: Activity heatmap (KDE) ───────────────────────────────────────
    if not args.skip_heatmap:
        heatmap_args = common_args + ["--cmap", args.cmap]
        results["heatmap"] = run_command(
            "activity_heatmap.py",
            heatmap_args,
            "Step 2/3: Activity Heatmap (KDE) — THESIS PRIMARY"
        )
    else:
        print("\n⊘ Skipping activity_heatmap.py")
        results["heatmap"] = None

    # ── Step 3: Per-bodypart heatmap grid ────────────────────────────────────
    if not args.skip_bodypart:
        results["bodypart"] = run_command(
            "bodypart_heatmaps.py",
            common_args,
            "Step 3/3: Per-Bodypart Heatmap Grid"
        )
    else:
        print("\n⊘ Skipping bodypart_heatmaps.py")
        results["bodypart"] = None

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📋 PIPELINE SUMMARY")
    print(f"{'='*70}")

    completed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)

    if results.get("orbit") is True:
        print("✅ Orbit trajectories generated")
        print("   → *_orbit_grid.png (per-bodypart trajectories)")
        print("   → *_thigmotaxis.png (body_center + walls)")

    if results.get("heatmap") is True:
        print("✅ Activity heatmap generated (THESIS PRIMARY)")
        print("   → *_heatmap_histogram.png (discrete bin density)")
        print("   → *_heatmap_kde.png (smooth density, inferno colormap)")

    if results.get("bodypart") is True:
        print("✅ Per-bodypart heatmap grid generated")
        print("   → *_bodypart_heatmaps.png (4×3 grid)")

    if skipped > 0:
        print(f"\n⊘ {skipped} script(s) skipped")

    if failed > 0:
        print(f"\n❌ {failed} script(s) FAILED")
        sys.exit(1)

    print(f"\n✨ All outputs saved to: ../data/DLCfiltered/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
