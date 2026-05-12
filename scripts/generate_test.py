"""
generate_test.py
----------------
Two synthetic DLC-format pose CSVs for testing predict_anxiety_v2.py:

  SyntheticControl_2026.csv  — control profile (center-dwelling, low thigmotaxis)
  SyntheticTreated_2026.csv  — treated profile  (wall-hugging, high thigmotaxis)

Each CSV follows the exact 3-row DLC header that load_dlc_csv() expects.
Rearing and grooming geometry is set to reliably trigger the behavior_detection
thresholds used by classify_frames().

Usage:
  python scripts/generate_test.py
  python scripts/generate_test.py --out reports/test_inputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ── Arena constants (must match src/anxiety/config.py) ───────────────────────
AX0, AX1, AY0, AY1 = 397.0, 777.0, 156.0, 535.0
IX0, IX1, IY0, IY1 = 473.0, 701.0, 232.0, 459.0   # 20 % margin inner zone
ARENA_CX = (AX0 + AX1) / 2   # 587
ARENA_CY = (AY0 + AY1) / 2   # 345.5

# ── DLC header (must match what load_dlc_csv column-flattening produces) ──────
SCORER = "DLC_Resnet50_rat_behavior_openfieldApr1shuffle1_snapshot_best-180"
BODYPARTS = [
    "nose", "head", "left_ear", "right_ear", "body_center",
    "left_forepaw", "right_forepaw", "left_hindpaw", "right_hindpaw", "tail_base",
]
COORDS = ["x", "y", "likelihood"]

N_FRAMES = 6000   # 200 s at 30 fps
FPS = 30.0


# ── Low-level pose builders ───────────────────────────────────────────────────

def _walk_pose(cx: float, cy: float, hdeg: float, rng: np.random.Generator) -> dict:
    """Normal locomotion pose centred at (cx, cy) facing hdeg degrees."""
    hrad = np.deg2rad(hdeg)
    dx, dy = np.cos(hrad), np.sin(hrad)

    # (forward_px, lateral_px) in body frame; lateral is left = −, right = +
    layout = {
        "nose":          (22,  0),
        "head":          (15,  0),
        "left_ear":      (10, -10),
        "right_ear":     (10,  10),
        "body_center":   (0,   0),
        "left_forepaw":  (8,  -12),
        "right_forepaw": (8,   12),
        "left_hindpaw":  (-12, -11),
        "right_hindpaw": (-12,  11),
        "tail_base":     (-22,  0),
    }
    pose: dict[str, tuple[float, float, float]] = {}
    for bp, (fwd, lat) in layout.items():
        px = cx + fwd * dx - lat * dy + rng.normal(0, 1.5)
        py = cy + fwd * dy + lat * dx + rng.normal(0, 1.5)
        pose[bp] = (float(px), float(py), 0.99)
    return pose


def _compact_rear_pose(cx: float, cy: float, rng: np.random.Generator) -> dict:
    """
    Compact center rearing (R1).

    Triggers:
      htdist  = dist(head, tail_base) ≈ 40 px  < REAR_COMPACT_HTDIST (75)
      nose2fp = dist(nose, avg_forepaw) ≈ 50 px > GROOM_NOSE2FP (35)  → not grooming
      body_center drifts slightly (body_still = False)  → sitting_upright gate does NOT fire
    """
    # Body compressed vertically: head above body_center, tail below
    head_y  = cy - 22.0
    tail_y  = cy + 18.0
    # htdist ≈ sqrt(0 + 40^2) = 40 ✓
    pose = {
        "body_center":   (cx,          cy,          0.99),
        "head":          (cx + rng.normal(0, 1), head_y, 0.99),
        "nose":          (cx + rng.normal(0, 1), head_y - 12.0, 0.99),
        "left_ear":      (cx - 7,      head_y + 4,  0.99),
        "right_ear":     (cx + 7,      head_y + 4,  0.99),
        # Forepaws stay at body level so nose2fp ≈ 46 px > 35
        "left_forepaw":  (cx - 12,     cy + 12,     0.99),
        "right_forepaw": (cx + 12,     cy + 12,     0.99),
        "left_hindpaw":  (cx - 10,     tail_y - 5,  0.99),
        "right_hindpaw": (cx + 10,     tail_y - 5,  0.99),
        "tail_base":     (cx + rng.normal(0, 1), tail_y, 0.99),
    }
    return pose


def _wall_press_rear_pose(cx: float, cy: float,
                          wall: str, rng: np.random.Generator) -> dict:
    """
    Wall-press rearing (R6).

    Triggers:
      nose pushed outside arena boundary (nose_x < 397 or > 775)
      htdist = dist(head, tail_base) ≈ 45 px  < REAR_WALL_PRESS_HTDIST (115)
    """
    if wall == "left":
        nose_x   = AX0 - 5.0     # 392 — outside left boundary
        head_x   = AX0 - 1.0     # 396
        fp_x     = AX0 + 6.0     # 403
        hp_x     = cx
        tail_x   = cx + 15.0
    else:
        nose_x   = AX1 + 5.0     # 782 — outside right boundary
        head_x   = AX1 + 1.0     # 778
        fp_x     = AX1 - 6.0     # 771
        hp_x     = cx
        tail_x   = cx - 15.0

    # htdist = dist(head, tail_base)
    # left case: sqrt((396 - (cx+15))^2 + 0^2)
    # if cx≈415: sqrt((396-430)^2) = 34 ✓ < 115

    pose = {
        "body_center":   (cx,                cy,                     0.99),
        "nose":          (nose_x,            cy + rng.normal(0, 3),  0.99),
        "head":          (head_x,            cy + rng.normal(0, 3),  0.99),
        "left_ear":      (head_x + (4 if wall == "right" else -4), cy - 7, 0.99),
        "right_ear":     (head_x + (4 if wall == "right" else -4), cy + 7, 0.99),
        "left_forepaw":  (fp_x,              cy - 10,                0.99),
        "right_forepaw": (fp_x,              cy + 10,                0.99),
        "left_hindpaw":  (hp_x,              cy - 10,                0.99),
        "right_hindpaw": (hp_x,              cy + 10,                0.99),
        "tail_base":     (tail_x,            cy + rng.normal(0, 2),  0.99),
    }
    return pose


def _grooming_pose(cx: float, cy: float, rng: np.random.Generator) -> dict:
    """
    Grooming (tight branch).

    Triggers:
      nose2fp < GROOM_NOSE2FP_TIGHT (22)
      fp_hp_vert = hp_y - fp_y ≈ 7   < GROOM_MAX_FPHP (10)
      body_vel ≈ 0                    < GROOM_MAX_VEL  (25)
      nose_inside = True
    """
    fp_y = cy - 2.0
    hp_y = fp_y + 7.0    # fp_hp_vert = 7 ✓
    pose = {
        "body_center":   (cx,                cy,       0.99),
        "nose":          (cx + rng.normal(0, 2), fp_y + rng.normal(0, 2), 0.99),
        "head":          (cx + rng.normal(0, 2), fp_y - 8,                0.99),
        "left_ear":      (cx - 7,            fp_y - 5, 0.99),
        "right_ear":     (cx + 7,            fp_y - 5, 0.99),
        "left_forepaw":  (cx - 5,            fp_y,     0.99),
        "right_forepaw": (cx + 5,            fp_y,     0.99),
        "left_hindpaw":  (cx - 9,            hp_y,     0.99),
        "right_hindpaw": (cx + 9,            hp_y,     0.99),
        "tail_base":     (cx + rng.normal(0, 2), cy + 25, 0.99),
    }
    return pose


# ── CSV writer ────────────────────────────────────────────────────────────────

def _write_csv(path: Path, frames: list[dict]) -> None:
    """Write frames (list of bodypart → (x, y, lik) dicts) as a DLC CSV."""
    n_cols = len(BODYPARTS) * len(COORDS)
    header1 = ["scorer"]  + [SCORER] * n_cols
    header2 = ["bodyparts"] + [bp for bp in BODYPARTS for _ in COORDS]
    header3 = ["coords"]   + COORDS * len(BODYPARTS)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(",".join(header1) + "\n")
        fh.write(",".join(header2) + "\n")
        fh.write(",".join(header3) + "\n")
        for i, pose in enumerate(frames):
            vals = [str(i)]
            for bp in BODYPARTS:
                x, y, lik = pose[bp]
                vals += [f"{x:.5f}", f"{y:.5f}", f"{lik:.5f}"]
            fh.write(",".join(vals) + "\n")


# ── Movement generators ───────────────────────────────────────────────────────

def _clip_arena(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x, AX0 + 5, AX1 - 5)),
        float(np.clip(y, AY0 + 5, AY1 - 5)),
    )


def _generate_frames(profile: str, rng: np.random.Generator) -> list[dict]:
    """
    Build N_FRAMES pose dicts for 'control' or 'treated'.

    Control: biased toward center zone, compact center rearing, grooming bouts.
    Treated: biased toward periphery, wall-press rearing, little grooming.
    """
    frames: list[dict] = []
    # Event schedules (start_frame, duration_frames)
    if profile == "control":
        step_std    = 2.5    # low locomotion
        center_pull = 0.08   # strong pull toward center
        wall_pull   = 0.00
        rear_events  = _schedule_events(
            n=14, min_start=300, max_end=N_FRAMES - 100,
            dur_lo=40, dur_hi=80, rng=rng)
        groom_events = _schedule_events(
            n=7, min_start=200, max_end=N_FRAMES - 100,
            dur_lo=60, dur_hi=150, rng=rng)
        rear_type = "center"
        # Start inside the center zone
        cx, cy = float(ARENA_CX), float(ARENA_CY)
    else:  # treated
        step_std    = 3.5    # more locomotion
        center_pull = 0.00
        wall_pull   = 0.10   # drift toward nearest wall
        rear_events  = _schedule_events(
            n=20, min_start=300, max_end=N_FRAMES - 100,
            dur_lo=35, dur_hi=70, rng=rng)
        groom_events = _schedule_events(
            n=3, min_start=500, max_end=N_FRAMES - 200,
            dur_lo=40, dur_hi=80, rng=rng)
        rear_type = "wall"
        # Start near left wall
        cx, cy = AX0 + 20.0, float(ARENA_CY)

    # Build event lookup: frame → event type
    event_map: dict[int, str] = {}
    for (s, e) in rear_events:
        for f in range(s, e):
            event_map[f] = "rear"
    for (s, e) in groom_events:
        for f in range(s, e):
            if f not in event_map:
                event_map[f] = "groom"

    heading = rng.uniform(0, 360)
    # Which wall side for treated rearing (alternate left/right)
    wall_sides = ["left", "right"] * (len(rear_events) // 2 + 1)
    wall_by_start = {s: wall_sides[i] for i, (s, _) in enumerate(rear_events)}
    active_wall = "left"

    for f in range(N_FRAMES):
        ev = event_map.get(f)

        if ev == "rear":
            # Determine which rear event this frame belongs to (for wall side)
            for s, e in rear_events:
                if s <= f < e:
                    active_wall = wall_by_start[s]
                    # Snap body toward appropriate position for the bout
                    if rear_type == "wall":
                        target_x = AX0 + 20.0 if active_wall == "left" else AX1 - 20.0
                        cx = cx * 0.85 + target_x * 0.15
                    else:
                        # Center rearing: drift body slowly so body_still=False
                        cx += rng.normal(0, 0.6)
                        cy += rng.normal(0, 0.6)
                    break

            if rear_type == "wall":
                pose = _wall_press_rear_pose(cx, cy, active_wall, rng)
            else:
                pose = _compact_rear_pose(cx, cy, rng)

        elif ev == "groom":
            # Grooming: body stays still
            pose = _grooming_pose(cx, cy, rng)

        else:
            # Normal locomotion
            heading += rng.normal(0, 12)

            # Attraction toward target zone
            if center_pull > 0:
                cx += center_pull * (ARENA_CX - cx)
                cy += center_pull * (ARENA_CY - cy)
            if wall_pull > 0:
                # Pull toward nearest wall
                dist_left  = cx - AX0
                dist_right = AX1 - cx
                dist_top   = cy - AY0
                dist_bot   = AY1 - cy
                min_d = min(dist_left, dist_right, dist_top, dist_bot)
                if min_d == dist_left:
                    cx -= wall_pull * dist_left
                elif min_d == dist_right:
                    cx += wall_pull * dist_right
                elif min_d == dist_top:
                    cy -= wall_pull * dist_top
                else:
                    cy += wall_pull * dist_bot

            cx += rng.normal(0, step_std)
            cy += rng.normal(0, step_std)
            cx, cy = _clip_arena(cx, cy)
            pose = _walk_pose(cx, cy, heading, rng)

        frames.append(pose)

    return frames


def _schedule_events(
    n: int, min_start: int, max_end: int,
    dur_lo: int, dur_hi: int, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Return n non-overlapping (start, end) event intervals."""
    events: list[tuple[int, int]] = []
    cursor = min_start
    total_span = max_end - min_start
    gap_per_event = total_span // (n + 1)

    for _ in range(n):
        dur = int(rng.integers(dur_lo, dur_hi))
        jitter = int(rng.integers(0, max(1, gap_per_event - dur)))
        start = cursor + jitter
        end   = min(start + dur, max_end)
        events.append((start, end))
        cursor = end + int(rng.integers(20, 60))
        if cursor >= max_end:
            break

    return events


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic DLC CSVs for model testing.")
    ap.add_argument("--out", type=Path, default=Path("test_data"),
                    help="Output directory (default: test_data/)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    profiles = [
        ("control", "SyntheticControl_2026"),
        ("treated", "SyntheticTreated_2026"),
    ]

    for profile, name in profiles:
        out_path = args.out / f"{name}.csv"
        print(f"[{profile}] generating {N_FRAMES} frames -> {out_path}")
        frames = _generate_frames(profile, rng)
        _write_csv(out_path, frames)

        # Quick sanity check: count expected behaviors in event map
        print(f"  written: {out_path} ({out_path.stat().st_size // 1024} KB)")

    print("\nTamam. Test etmek için:")
    print(f"  python scripts/predict_anxiety_v2.py {args.out}/SyntheticControl_2026.csv")
    print(f"  python scripts/predict_anxiety_v2.py {args.out}/SyntheticTreated_2026.csv")


if __name__ == "__main__":
    main()
