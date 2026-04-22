"""
behavior_detection.py
---------------------
Detects rearing and grooming from DLC-filtered tracking data.

Detection logic
---------------
GROOMING posture is computed FIRST (before rearing) so it can disambiguate
compact rearing from grooming – both compress the body in 2-D projection,
but grooming has nose close to forepaws while rearing does not.

GROOMING – three sufficient postural branches (disjunction), all requiring
     body_vel < GROOM_MAX_VEL:

       (a) Tight posture  (nose2fp < GROOM_NOSE2FP_TIGHT, fp_hp_vert low)
           Classic face-washing: nose held against the forepaw cluster.

       (b) Loose posture + stillness  (fp_hp_vert low)
           Low / splayed body-grooming where the nose is farther from the
           paws in 2-D projection but the animal is nearly stationary.

       (c) Upright / sit-up posture  (fp_hp_vert elevated, nose_y not near
           the top wall)
           Rat sits on haunches with forepaws raised to the face. Forepaws
           are ABOVE hindpaws in image (fp_hp_vert > GROOM_MAX_FPHP), so
           branches (a)/(b) miss this phenotype. The nose_y gate keeps
           real top-wall rearing from being reclassified as grooming.

     Confirmed ground-truth windows:
       MA1_2  — frames 2181-2196 (brief face-wash), 4795-5034 (2:39-2:47).
       MA5_1  — ~65-67, 92-103, 106-121, 131-145, 145-180 s (long bouts).
       MA7_1  — ~160-170 s (upright / sit-up grooming).
     Note: the MA5_1 78-81 s bout is unrecoverable from DLC because the
     nose is occluded by paws (likelihood < 0.6 in 85/90 frames).

REARING – five complementary cues (any is sufficient):

  1. Compact rearing  (htdist < REAR_COMPACT_HTDIST  AND  NOT grooming_posture)
     Body compressed in 2-D projection. Grooming posture is excluded
     because a grooming rat also compresses its body but keeps its nose
     near the forepaws. Other rearing rules (R2-R5) have wall-specific
     gates and need no grooming exclusion.

  2. Top-wall extended rearing  (fp_hp_vert > REAR_EXTEND_FPHP  AND  nose_y < REAR_NOSE_Y_MAX)
     Forepaws clearly above hindpaws + nose near the top wall.

  3-4. Bottom-wall rearing  (strong: fp_hp < -80  OR  compact: fp_hp < -45 + htdist guard)
       Forepaws well below hindpaws + nose near the bottom wall.

  5. Side-wall rearing  (htd_y in 40-60 + nose at left/right boundary)

  6. Wall-press rearing  (nose beyond arena boundary + htdist < 115)
     Catches rearing missed by R5 when the rat presses flat against a
     side wall with low htd_y. Nose outside the arena = wall interaction.

The final grooming label = grooming_posture AND NOT rearing.

Outputs
-------
  <name>_behavior_timeline.png   – colour-coded timeline
  <name>_behavior_bouts.csv      – per-bout summary table
  Printed summary to stdout

Usage
-----
    python behavior_detection.py
    python behavior_detection.py --csv data/DLCfiltered/OpenFieldMA1_2.csv
    python behavior_detection.py --csv data/DLCfiltered/OpenFieldMA1_2.csv \\
        --fps 30 --out-dir data/DLCfiltered
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── THRESHOLDS ───────────────────────────────────────────────────────────────
# Rearing – compact (body compressed against wall in 2-D projection)
REAR_COMPACT_HTDIST   = 75     # px: head-to-tail distance below this → rearing (adjusted from 55 for labeled data)

# Rearing – extended (body upright near the TOP wall of the arena)
REAR_EXTEND_FPHP      = 45     # px: hindpaw_y - forepaw_y must exceed this
REAR_NOSE_Y_MAX       = 165    # px: nose must be above this y-threshold (smaller y = higher)

# Rearing – side walls (rat stands against the LEFT or RIGHT wall)
REAR_SIDE_NOSE_X_RIGHT = 760   # px: nose beyond this x → near right wall  (arena right edge ≈ 775)
REAR_SIDE_NOSE_X_LEFT  = 430   # px: nose below this x → near left wall   (arena left edge  ≈ 397)
REAR_SIDE_HTD_Y_MIN    = 40    # px: head-tail y-separation lower bound  (< 40 = horizontal thigmotaxis, not rearing)
REAR_SIDE_HTD_Y_MAX    = 60    # px: head-tail y-separation upper bound  (> 60 = normal oblique locomotion)

# Rearing – bottom wall (rat stands against the BOTTOM wall; forepaws are lower than hindpaws)
# Two sub-types with separate conditions so a borderline "walking near wall" bout (high velocity,
# body not compressed) does not count:
#   Strong:  very large fp_hp_vert signal — catches early-session rearing (frames ~22-58)
#   Compact: moderate signal but body is compressed (htdist small) — catches frames ~1016-1038
REAR_BOTTOM_STRONG_FPHP   = -80   # strong: forepaws very far below hindpaws
REAR_BOTTOM_COMPACT_FPHP  = -45   # compact: moderate forepaw depression + body compressed
REAR_BOTTOM_HTDIST        = 105   # compact: head-to-tail distance must be below this (body squished)
REAR_BOTTOM_NOSE_Y        = 500   # both: nose must be near the bottom wall

# Grooming – disjunctive posture rule (see module docstring).
#   Tight branch covers upright face-washing (MA1_2): nose held against paws.
#   Loose branch covers low/splayed body-grooming (MA5_1): wider nose2fp
#     window, gated by near-zero body velocity to exclude locomotion.
#   Both branches require a velocity gate: during wall-rearing the nose can
#   be close to the forepaws in 2-D projection (both pressed on the wall),
#   so velocity separates stationary grooming from active wall-rearing.
GROOM_NOSE2FP_TIGHT   = 22     # px: tight-posture grooming
GROOM_NOSE2FP         = 35     # px: loose-posture grooming
GROOM_MAX_FPHP        = 10     # forepaw must NOT be elevated above hindpaw (excludes rearing postures)
GROOM_MAX_VEL         = 25     # px/s: body_center speed (grooming ≈ stationary)
GROOM_VEL_WINDOW      = 5      # frames: rolling mean window for velocity smoothing

# Arena bounds — nose outside these limits means the rat is pressed against
# (or past) the wall and cannot be grooming.  Also used for wall-press
# rearing (R6): nose beyond boundary + compressed body → rearing.
ARENA_X_LEFT   = 397    # px: left wall
ARENA_X_RIGHT  = 775    # px: right wall
ARENA_Y_TOP    = 158    # px: top wall
ARENA_Y_BOTTOM = 532    # px: bottom wall

# Rearing – wall-press (nose pushed beyond any arena boundary + body compressed)
# Catches side-wall rearing with low htd_y that R5 misses (e.g. MA5_2 ~17 s).
REAR_WALL_PRESS_HTDIST = 115   # px: body must be compressed (walking along wall ≈ 130+)

# Tracking quality filter
LIKELIHOOD_THRESH     = 0.6

# Bout grouping
INTER_BOUT_GAP        = 15     # frames: gap ≤ this is bridged into a single bout
MIN_BOUT_FRAMES       = 10     # frames: bouts shorter than this are discarded

# Defaults
DEFAULT_CSV     = "../data/DLCfiltered/OpenFieldMA1_2.csv"
DEFAULT_OUT_DIR = None   # None = same directory as the input CSV
DEFAULT_FPS     = 30

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_dlc_csv(path: str) -> pd.DataFrame:
    """Load a DLC filtered CSV and return a flat DataFrame with short column names."""
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = ["_".join(c[-2:]).strip() for c in df.columns.values]
    df.index = df.index.astype(int)
    return df


def mask_low_likelihood(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """Replace x/y with NaN for any keypoint whose likelihood < thresh."""
    df = df.copy()
    bodyparts = {c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_x")}
    for bp in bodyparts:
        mask = df[f"{bp}_likelihood"] < thresh
        df.loc[mask, f"{bp}_x"] = np.nan
        df.loc[mask, f"{bp}_y"] = np.nan
    return df


def compute_features(
    raw_df: pd.DataFrame,
    masked_df: pd.DataFrame,
    fps: float = DEFAULT_FPS,
) -> pd.DataFrame:
    """Compute per-frame postural features used for behaviour classification.

    raw_df    – original DLC output (no likelihood masking): used for htdist so
                tail occlusion during wall-rearing does not create NaN values.
    masked_df – coordinates set to NaN where likelihood < threshold: used for
                positional/distance features that rely on accurate keypoint locations.
    fps       – video frame rate; needed to express body velocity in px/s.
    """
    # Use .mean(axis=1) so a single occluded (NaN) paw does not blank the pair
    fp_x = masked_df[["left_forepaw_x", "right_forepaw_x"]].mean(axis=1)
    fp_y = masked_df[["left_forepaw_y", "right_forepaw_y"]].mean(axis=1)
    hp_y = masked_df[["left_hindpaw_y", "right_hindpaw_y"]].mean(axis=1)

    feat = pd.DataFrame(index=raw_df.index)

    # Head-to-tail distance computed on RAW (unmasked) data: tail_base is often
    # occluded when the rat rears against a wall, so we must not discard it.
    feat["htdist"] = np.sqrt(
        (raw_df["head_x"] - raw_df["tail_base_x"]) ** 2
        + (raw_df["head_y"] - raw_df["tail_base_y"]) ** 2
    )

    # Hindpaw minus forepaw vertical separation (positive = forepaws elevated)
    feat["fp_hp_vert"] = hp_y - fp_y

    # Nose height in image (smaller y = higher in the frame)
    feat["nose_y"] = masked_df["nose_y"]

    # Nose-to-forepaw cluster distance (grooming cue)
    feat["nose2fp"] = np.sqrt(
        (masked_df["nose_x"] - fp_x) ** 2 + (masked_df["nose_y"] - fp_y) ** 2
    )

    # Side-wall rearing cues (raw data: nose near left/right boundary + body runs horizontally)
    feat["htd_y"]  = (raw_df["head_y"] - raw_df["tail_base_y"]).abs()
    feat["nose_x"] = masked_df["nose_x"]

    # Body-center speed in px/frame, smoothed and converted to px/s.
    # Assumes unit frame spacing (consecutive DLC rows are consecutive frames).
    bc_dx = masked_df["body_center_x"].diff()
    bc_dy = masked_df["body_center_y"].diff()
    bc_speed_per_frame = np.sqrt(bc_dx ** 2 + bc_dy ** 2)
    feat["body_vel"] = (
        bc_speed_per_frame
        .rolling(GROOM_VEL_WINDOW, min_periods=1, center=True)
        .mean()
        * fps
    )

    return feat


def classify_frames(feat: pd.DataFrame) -> pd.Series:
    """Return a Series of string labels: 'rearing', 'grooming', or 'other'."""
    # ── Grooming posture (computed BEFORE rearing so it can gate R1) ──────
    # Three sufficient branches cover the observed grooming phenotypes:
    #   tight   — upright face-wash with nose held against paws (MA1_2)
    #   loose   — low / splayed body-grooming, stationary (MA5_1)
    #   upright — sit-up grooming on haunches, forepaws elevated (MA7_1).
    #             Distinguished from top-wall rearing by the nose_y gate:
    #             real top-wall rearing has nose_y < REAR_NOSE_Y_MAX (near
    #             the top of the frame); sit-up grooming happens in the
    #             middle of the arena.
    # All three require a velocity gate: during wall-rearing the nose can
    # be close to the forepaws in 2-D projection (both pressed on the
    # wall), so velocity separates stationary grooming from active rearing.
    # NaN in any feature → condition evaluates False (safe).
    nose_inside = (
        (feat["nose_x"] > ARENA_X_LEFT) & (feat["nose_x"] < ARENA_X_RIGHT)
        & (feat["nose_y"] > ARENA_Y_TOP) & (feat["nose_y"] < ARENA_Y_BOTTOM)
    )
    groom_tight = (
        (feat["nose2fp"] < GROOM_NOSE2FP_TIGHT)
        & (feat["body_vel"] < GROOM_MAX_VEL)
        & (feat["fp_hp_vert"] < GROOM_MAX_FPHP)
    )
    groom_loose = (
        (feat["nose2fp"] < GROOM_NOSE2FP)
        & (feat["body_vel"] < GROOM_MAX_VEL)
        & (feat["fp_hp_vert"] < GROOM_MAX_FPHP)
    )
    groom_upright = (
        (feat["nose2fp"] < GROOM_NOSE2FP)
        & (feat["body_vel"] < GROOM_MAX_VEL)
        & (feat["fp_hp_vert"] > GROOM_MAX_FPHP)        # forepaws elevated
        & (feat["nose_y"] > REAR_NOSE_Y_MAX)           # not near top wall
    )
    grooming_posture = (groom_tight | groom_loose | groom_upright) & nose_inside

    # ── Rearing rules ────────────────────────────────────────────────────
    # R1 – Compact rearing (body compressed in 2-D projection).
    #   Grooming also compresses the body (htdist drops), so we exclude
    #   frames that show clear grooming posture (nose near forepaws,
    #   forepaws not elevated, low velocity).  Other rearing rules
    #   (R2-R5) have wall-specific gates and are unaffected.
    rear_compact  = (feat["htdist"] < REAR_COMPACT_HTDIST) & ~grooming_posture

    # R2 – Top-wall extended rearing
    rear_top_wall = (feat["fp_hp_vert"] > REAR_EXTEND_FPHP) & (feat["nose_y"] < REAR_NOSE_Y_MAX)
    # R3/R4 – Bottom-wall rearing (strong signal OR compact body)
    rear_bot_strong  = (feat["fp_hp_vert"] < REAR_BOTTOM_STRONG_FPHP) & (feat["nose_y"] > REAR_BOTTOM_NOSE_Y)
    rear_bot_compact = (feat["fp_hp_vert"] < REAR_BOTTOM_COMPACT_FPHP) & (feat["nose_y"] > REAR_BOTTOM_NOSE_Y) & (feat["htdist"] < REAR_BOTTOM_HTDIST)
    # R5 – Side-wall rearing: nose at left/right boundary + body axis partially horizontal
    rear_side_wall = (
        (feat["htd_y"] > REAR_SIDE_HTD_Y_MIN) & (feat["htd_y"] < REAR_SIDE_HTD_Y_MAX)
        & ((feat["nose_x"] > REAR_SIDE_NOSE_X_RIGHT) | (feat["nose_x"] < REAR_SIDE_NOSE_X_LEFT))
        & (feat["fp_hp_vert"] < 25)
    )
    # R6 – Wall-press rearing: nose pushed beyond any arena boundary + body
    #   compressed.  Catches wall rearing that R5 misses (e.g. htd_y too low
    #   for R5's band when the rat presses flat against a side wall).
    nose_at_wall = ~nose_inside & feat["nose_x"].notna()
    rear_wall_press = nose_at_wall & (feat["htdist"] < REAR_WALL_PRESS_HTDIST)

    rearing = rear_compact | rear_top_wall | rear_bot_strong | rear_bot_compact | rear_side_wall | rear_wall_press

    # ── Final grooming label: grooming posture that wasn't claimed by rearing ─
    grooming = grooming_posture & ~rearing

    labels = pd.Series("other", index=feat.index, dtype=str)
    labels[grooming] = "grooming"
    labels[rearing]  = "rearing"
    return labels


def frames_to_bouts(
    frames: np.ndarray,
    gap: int = INTER_BOUT_GAP,
    min_dur: int = MIN_BOUT_FRAMES,
) -> list[tuple[int, int]]:
    """Merge consecutive detected frames into bouts; discard short ones."""
    if len(frames) == 0:
        return []
    bouts: list[tuple[int, int]] = []
    start = int(frames[0])
    prev  = int(frames[0])
    for f in frames[1:]:
        f = int(f)
        if f - prev > gap:
            if prev - start + 1 >= min_dur:  # +1: inclusive frame count
                bouts.append((start, prev))
            start = f
        prev = f
    if prev - start + 1 >= min_dur:
        bouts.append((start, prev))
    return bouts


def bouts_to_dataframe(bouts: list[tuple[int, int]], label: str, fps: float) -> pd.DataFrame:
    rows = []
    for i, (s, e) in enumerate(bouts, 1):
        rows.append(
            {
                "behaviour":  label,
                "bout":       i,
                "start_frame": s,
                "end_frame":   e,
                "start_s":    round(s / fps, 2),
                "end_s":      round(e / fps, 2),
                "duration_s": round((e - s) / fps, 2),
            }
        )
    return pd.DataFrame(rows)


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

BEHAVIOUR_COLORS = {
    "rearing":  "#E74C3C",
    "grooming": "#2ECC71",
    "other":    "#D0D0D0",
}

GROUND_TRUTH_BY_SUBJECT: dict[str, dict[str, list[tuple[int, int]]]] = {
    "OpenFieldMA1_2": {
        # Confirmed rearing windows (frames at 30 fps)
        "rearing": [
            (21, 57),          # 0.7–1.9 s   (bottom-wall compact)
            (323, 371),        # 10.8–12.4 s (top-wall extended)
            (685, 729),        # 22.8–24.3 s (compact)
            (783, 801),        # 26.1–26.7 s (side-wall right)
            (876, 945),        # 29.2–31.5 s (side-wall left / bottom-wall)
            (990, 1040),       # 33.0–34.7 s (bottom-wall compact)
            (1236, 1290),      # 41.2–43.0 s (compact)
            (1800, 1893),      # 59.7–63.1 s (top-wall extended)
            (2025, 2061),      # 67.5–68.7 s (compact)
        ],
        # Confirmed grooming windows
        "grooming": [
            (2181, 2196),      # 72.7–73.2 s  (brief face-washing)
            (4806, 5031),      # 160.2–167.7 s (extended grooming)
        ],
    },
    "OpenFieldMA5_1": {
        "rearing": [],
        # Grooming windows reported from video review (seconds → frames @ 30 fps).
        # 78-81 s is intentionally omitted: DLC nose likelihood fails in 85/90
        # frames of that window, so it is unrecoverable from the current data.
        "grooming": [
            (1950, 2010),      # 65-67 s
            (2760, 3090),      # 92-103 s
            (3180, 3630),      # 106-121 s
            (3930, 4350),      # 131-145 s
            (4350, 5400),      # 145-180 s (grooming-like continuation)
        ],
    },
    "OpenFieldMA7_1": {
        "rearing": [],
        # Upright / sit-up grooming confirmed by video review.
        # User-reported window is 158–180 s, but the tracked CSV ends at
        # frame 5115 (170.5 s), so GT is capped at the data boundary.
        "grooming": [
            (4740, 5115),      # 158–170.5 s (upright grooming)
        ],
    },
}


def ground_truth_for(base: str) -> dict[str, list[tuple[int, int]]]:
    """Return the GT dict for a given CSV basename, or empty if unknown."""
    return GROUND_TRUTH_BY_SUBJECT.get(base, {"rearing": [], "grooming": []})


def plot_timeline(
    labels: pd.Series,
    rear_bouts: list[tuple[int, int]],
    groom_bouts: list[tuple[int, int]],
    fps: float,
    out_path: str,
    ground_truth: dict[str, list[tuple[int, int]]] | None = None,
) -> None:
    if ground_truth is None:
        ground_truth = {"rearing": [], "grooming": []}
    n_frames = len(labels)
    time_s   = np.arange(n_frames) / fps

    fig, axes = plt.subplots(3, 1, figsize=(18, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1]})
    fig.suptitle("Behaviour Timeline — Rearing & Grooming Detection", fontsize=13, y=1.01)

    row_labels   = ["Ground Truth", "Detected", "Frame Labels"]
    row_data     = [
        # ground truth (from the confirmed windows above)
        None,
        # detected bouts
        None,
        # per-frame colour strip
        None,
    ]

    # ── Row 0: ground truth ──
    ax0 = axes[0]
    ax0.set_ylabel("Ground\nTruth", rotation=0, ha="right", va="center", fontsize=9)
    ax0.set_yticks([])
    ax0.set_ylim(0, 1)
    for beh, color in [("rearing", BEHAVIOUR_COLORS["rearing"]),
                        ("grooming", BEHAVIOUR_COLORS["grooming"])]:
        for s, e in ground_truth.get(beh, []):
            ax0.axvspan(s / fps, e / fps, ymin=0, ymax=1,
                        color=color, alpha=0.6, label=beh)

    # ── Row 1: detected bouts ──
    ax1 = axes[1]
    ax1.set_ylabel("Detected\nBouts", rotation=0, ha="right", va="center", fontsize=9)
    ax1.set_yticks([])
    ax1.set_ylim(0, 1)
    for s, e in rear_bouts:
        ax1.axvspan(s / fps, e / fps, ymin=0, ymax=1,
                    color=BEHAVIOUR_COLORS["rearing"], alpha=0.6)
    for s, e in groom_bouts:
        ax1.axvspan(s / fps, e / fps, ymin=0, ymax=1,
                    color=BEHAVIOUR_COLORS["grooming"], alpha=0.6)

    # ── Row 2: per-frame colour strip ──
    ax2 = axes[2]
    ax2.set_ylabel("Per-Frame\nLabel", rotation=0, ha="right", va="center", fontsize=9)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    color_arr = np.array([BEHAVIOUR_COLORS[lb] for lb in labels])
    for i, c in enumerate(color_arr):
        if c != BEHAVIOUR_COLORS["other"]:
            ax2.axvspan(time_s[i], time_s[i] + 1 / fps,
                        ymin=0, ymax=1, color=c, alpha=0.8, linewidth=0)

    # Shared x axis
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_xlim(0, n_frames / fps)

    # Legend
    legend_handles = [
        mpatches.Patch(color=BEHAVIOUR_COLORS["rearing"],  label="Rearing"),
        mpatches.Patch(color=BEHAVIOUR_COLORS["grooming"], label="Grooming"),
        mpatches.Patch(color=BEHAVIOUR_COLORS["other"],    label="Other"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right",
                   fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect rearing and grooming from DLC CSV")
    parser.add_argument("--csv",            default=DEFAULT_CSV,      help="Path to DLC filtered CSV")
    parser.add_argument("--fps",            type=float, default=DEFAULT_FPS, help="Video frame rate")
    parser.add_argument("--out-dir",        default=DEFAULT_OUT_DIR,  help="Output directory")
    parser.add_argument("--min-bout-frames", type=int, default=MIN_BOUT_FRAMES,
                        dest="min_bout_frames",
                        help=f"Minimum frames to count as a bout (default {MIN_BOUT_FRAMES}; "
                             "use 1 for sparse labeled data)")
    parser.add_argument("--inter-bout-gap", type=int, default=INTER_BOUT_GAP,
                        dest="inter_bout_gap",
                        help=f"Max frame gap to merge into one bout (default {INTER_BOUT_GAP})")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    out_dir  = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    fps  = args.fps

    print(f"Loading {csv_path}")
    raw_df    = load_dlc_csv(csv_path)
    masked_df = mask_low_likelihood(raw_df, LIKELIHOOD_THRESH)
    feat      = compute_features(raw_df, masked_df, fps=fps)

    print("Classifying frames...")
    labels = classify_frames(feat)

    rear_frames  = np.where(labels == "rearing")[0]
    groom_frames = np.where(labels == "grooming")[0]

    rear_bouts  = frames_to_bouts(rear_frames,  gap=args.inter_bout_gap, min_dur=args.min_bout_frames)
    groom_bouts = frames_to_bouts(groom_frames, gap=args.inter_bout_gap, min_dur=args.min_bout_frames)

    # ── Console summary ──────────────────────────────────────────────────────
    total_s = len(raw_df) / fps
    print(f"\nVideo length : {total_s:.1f} s  ({len(raw_df)} frames @ {fps} fps)")
    print(f"Likelihood filter  : > {LIKELIHOOD_THRESH}")
    print(f"Rearing thresholds : htdist<{REAR_COMPACT_HTDIST}"
          f" | (fp_hp>{REAR_EXTEND_FPHP} & nose_y<{REAR_NOSE_Y_MAX})"
          f" | (fp_hp<{REAR_BOTTOM_STRONG_FPHP} & nose_y>{REAR_BOTTOM_NOSE_Y})"
          f" | (fp_hp<{REAR_BOTTOM_COMPACT_FPHP} & nose_y>{REAR_BOTTOM_NOSE_Y} & htdist<{REAR_BOTTOM_HTDIST})")
    print(f"Grooming thresholds : (nose2fp<{GROOM_NOSE2FP_TIGHT} & fp_hp<{GROOM_MAX_FPHP})"
          f" OR (nose2fp<{GROOM_NOSE2FP} & fp_hp<{GROOM_MAX_FPHP})"
          f" OR (nose2fp<{GROOM_NOSE2FP} & fp_hp>{GROOM_MAX_FPHP} & nose_y>{REAR_NOSE_Y_MAX})"
          f"   [all & body_vel<{GROOM_MAX_VEL} px/s & not rearing]")
    print(f"Minimum bout duration : {args.min_bout_frames} frames ({args.min_bout_frames/fps:.2f} s)\n")

    print("REARING BOUTS:")
    for i, (s, e) in enumerate(rear_bouts, 1):
        print(f"  Bout {i:2d}:  frames {s:5d}–{e:5d}"
              f"  ({s/fps:6.2f} s – {e/fps:6.2f} s,"
              f"  duration {(e-s)/fps:.2f} s)")

    print(f"\nGROOMING BOUTS:")
    for i, (s, e) in enumerate(groom_bouts, 1):
        print(f"  Bout {i:2d}:  frames {s:5d}–{e:5d}"
              f"  ({s/fps:6.2f} s – {e/fps:6.2f} s,"
              f"  duration {(e-s)/fps:.2f} s)")

    rear_time_s  = sum(e - s for s, e in rear_bouts)  / fps
    groom_time_s = sum(e - s for s, e in groom_bouts) / fps
    print(f"\nTotal rearing time  : {rear_time_s:.1f} s"
          f"  ({100*rear_time_s/total_s:.1f}% of session)")
    print(f"Total grooming time : {groom_time_s:.1f} s"
          f"  ({100*groom_time_s/total_s:.1f}% of session)")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    _BOUT_COLS = ["behaviour", "bout", "start_frame", "end_frame", "start_s", "end_s", "duration_s"]
    _parts = [bouts_to_dataframe(rear_bouts, "rearing", fps),
              bouts_to_dataframe(groom_bouts, "grooming", fps)]
    _parts = [p for p in _parts if not p.empty]
    bout_df = pd.concat(_parts) if _parts else pd.DataFrame(columns=_BOUT_COLS)
    if not bout_df.empty:
        bout_df = bout_df.sort_values("start_frame")
    bout_df = bout_df.reset_index(drop=True)

    csv_out = os.path.join(out_dir, f"{base}_behavior_bouts.csv")
    bout_df.to_csv(csv_out, index=False)
    print(f"\n  Saved: {csv_out}")

    # Also save a per-frame label CSV
    frame_labels = pd.DataFrame({
        "frame":     raw_df.index,
        "time_s":    raw_df.index / fps,
        "behaviour": labels.values,
    })
    frame_csv = os.path.join(out_dir, f"{base}_behavior_frames.csv")
    frame_labels.to_csv(frame_csv, index=False)
    print(f"  Saved: {frame_csv}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\nPlotting timeline...")
    img_out = os.path.join(out_dir, f"{base}_behavior_timeline.png")
    plot_timeline(labels, rear_bouts, groom_bouts, fps, img_out,
                  ground_truth=ground_truth_for(base))

    print("\nDone.")


if __name__ == "__main__":
    main()
