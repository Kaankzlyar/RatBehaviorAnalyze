"""
dlc_inference.py
----------------
Step 5: Run trained DLC model on all videos, filter predictions,
        generate labeled videos for QC, and export clean tracking CSVs.
Run after training is complete in dlc_train.py.

Outputs are separated by arena:
    data/dlc_output/rectangle/  — open field videos (Part0)
    data/dlc_output/tmaze/      — T-maze videos (Part1, Part2)
"""

import os
import glob
import numpy as np
import pandas as pd
import deeplabcut

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CONFIG_PATH   = "D:/ProjectsD/ThesisWork/models/dlc_model/rat_behavior-kaank-YYYY-MM-DD/config.yaml"
VIDEO_DIR     = "D:/ProjectsD/ThesisWork/data/raw_videos"
OUTPUT_BASE   = "D:/ProjectsD/ThesisWork/data/dlc_output"
LABELED_DIR   = "D:/ProjectsD/ThesisWork/data/dlc_output/labeled_videos"
VIDEO_EXT     = "avi"
SHUFFLE       = 1
GPU_ID        = 0
LIKELIHOOD_THRESH = 0.6   # frames below this confidence are set to NaN

# Arena type determined by which Part subfolder the video lives in
ARENA_MAP = {
    "Part0": "rectangle",
    "Part1": "tmaze",
    "Part2": "tmaze",
}

os.makedirs(LABELED_DIR, exist_ok=True)
for arena in set(ARENA_MAP.values()):
    os.makedirs(os.path.join(OUTPUT_BASE, arena), exist_ok=True)

# ─── COLLECT VIDEOS ───────────────────────────────────────────────────────────

def get_arena_type(video_path: str) -> str:
    """Return 'rectangle' or 'tmaze' based on which Part subfolder the video lives in."""
    normalized = video_path.replace("\\", "/")
    for part, arena in ARENA_MAP.items():
        if f"/{part}/" in normalized:
            return arena
    return "unknown"

def get_videos() -> list[str]:
    """Recursively collect all .avi videos from Part0/, Part1/, Part2/."""
    videos = glob.glob(os.path.join(VIDEO_DIR, "**", f"*.{VIDEO_EXT}"), recursive=True)
    if not videos:
        raise FileNotFoundError(f"No .{VIDEO_EXT} videos found under {VIDEO_DIR}")
    for part, arena in ARENA_MAP.items():
        part_vids = [v for v in videos if part in v.replace("\\", "/")]
        print(f"  {part} ({arena}): {len(part_vids)} video(s)")
    print(f"Total: {len(videos)} video(s)")
    return videos

def get_videos_by_arena() -> dict[str, list[str]]:
    """Return videos grouped by arena type."""
    all_videos = get_videos()
    grouped: dict[str, list[str]] = {}
    for v in all_videos:
        arena = get_arena_type(v)
        grouped.setdefault(arena, []).append(v)
    return grouped

# ─── RUN INFERENCE ────────────────────────────────────────────────────────────

def analyze_videos(videos: list[str], arena: str) -> None:
    """Run DLC inference and write results to the arena-specific output folder."""
    out_dir = os.path.join(OUTPUT_BASE, arena)
    print(f"\nRunning DLC inference on {len(videos)} {arena} video(s) → {out_dir}")
    deeplabcut.analyze_videos(
        CONFIG_PATH,
        videos,
        videotype=VIDEO_EXT,
        shuffle=SHUFFLE,
        save_as_csv=True,
        destfolder=out_dir,
        gputouse=GPU_ID,
    )
    print(f"Tracking data saved to: {out_dir}")

# ─── FILTER PREDICTIONS ───────────────────────────────────────────────────────

def filter_predictions(videos: list[str], arena: str) -> None:
    """Smooth tracking with median filter; results written to arena output folder."""
    out_dir = os.path.join(OUTPUT_BASE, arena)
    deeplabcut.filterpredictions(
        CONFIG_PATH,
        videos,
        videotype=VIDEO_EXT,
        shuffle=SHUFFLE,
        filtertype="median",
        windowlength=5,
        destfolder=out_dir,
    )
    print(f"Filtered predictions saved for {arena}.")

# ─── CREATE LABELED VIDEOS (QC) ───────────────────────────────────────────────

def create_labeled_videos(videos: list[str]) -> None:
    """Generate annotated videos to visually verify tracking quality."""
    deeplabcut.create_labeled_video(
        CONFIG_PATH,
        videos,
        videotype=VIDEO_EXT,
        shuffle=SHUFFLE,
        filtered=True,
        draw_skeleton=True,
        destfolder=LABELED_DIR,
    )
    print(f"Labeled QC videos saved to: {LABELED_DIR}")

# ─── LOAD & CLEAN TRACKING DATA ───────────────────────────────────────────────

def load_tracking(csv_path: str) -> dict[str, dict[str, np.ndarray]]:
    """
    Load a DLC CSV and apply likelihood threshold.
    Returns dict: { bodypart: { 'x': array, 'y': array, 'likelihood': array } }
    Low-confidence frames are set to NaN (interpolated in next step).
    """
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(0).unique().tolist()

    tracking = {}
    for bp in bodyparts:
        x          = df[bp]["x"].values.astype(float)
        y          = df[bp]["y"].values.astype(float)
        likelihood = df[bp]["likelihood"].values.astype(float)

        mask      = likelihood < LIKELIHOOD_THRESH
        x[mask]   = np.nan
        y[mask]   = np.nan

        tracking[bp] = {"x": x, "y": y, "likelihood": likelihood}

    pct_valid = {
        bp: np.sum(~np.isnan(tracking[bp]["x"])) / len(tracking[bp]["x"]) * 100
        for bp in bodyparts
    }
    print(f"  {os.path.basename(csv_path)}")
    for bp, pct in pct_valid.items():
        print(f"    {bp}: {pct:.1f}% frames above likelihood threshold")

    return tracking

def interpolate_gaps(tracking: dict) -> dict:
    """
    Linear interpolation for short NaN gaps (dropped frames).
    Gaps longer than 30 frames are left as NaN.
    """
    MAX_GAP = 30
    for bp in tracking:
        for coord in ["x", "y"]:
            arr = pd.Series(tracking[bp][coord])
            arr = arr.interpolate(method="linear", limit=MAX_GAP, limit_direction="both")
            tracking[bp][coord] = arr.values
    return tracking

def load_sessions_for_arena(arena: str) -> dict[str, dict]:
    """
    Load all filtered DLC CSVs for a given arena from its output folder.
    Returns dict keyed by video name.
    """
    out_dir   = os.path.join(OUTPUT_BASE, arena)
    csv_files = glob.glob(os.path.join(out_dir, "*filtered*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(out_dir, "*.csv"))

    sessions = {}
    print(f"\nLoading {len(csv_files)} {arena} tracking file(s)...")
    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        tracking = load_tracking(csv_path)
        tracking = interpolate_gaps(tracking)
        sessions[name] = tracking

    return sessions

# ─── EXPORT CLEAN CSV ─────────────────────────────────────────────────────────

def export_clean_csv(sessions: dict, arena: str) -> None:
    """
    Export a clean, flat CSV per session:
    columns = frame, nose_x, nose_y, head_x, head_y, ...
    Saved to data/dlc_output/clean/<arena>/
    """
    clean_dir = os.path.join(OUTPUT_BASE, "clean", arena)
    os.makedirs(clean_dir, exist_ok=True)

    for name, tracking in sessions.items():
        bodyparts = list(tracking.keys())
        n_frames  = len(tracking[bodyparts[0]]["x"])

        rows = {"frame": np.arange(n_frames)}
        for bp in bodyparts:
            rows[f"{bp}_x"]           = tracking[bp]["x"]
            rows[f"{bp}_y"]           = tracking[bp]["y"]
            rows[f"{bp}_likelihood"]  = tracking[bp]["likelihood"]

        df = pd.DataFrame(rows)
        out_path = os.path.join(clean_dir, f"{name}_clean.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print(f"\nClean CSVs saved to: {clean_dir}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    videos_by_arena = get_videos_by_arena()

    for arena, videos in videos_by_arena.items():
        print(f"\n{'='*50}")
        print(f"Arena: {arena.upper()}  ({len(videos)} videos)")
        print(f"{'='*50}")

        analyze_videos(videos, arena)
        filter_predictions(videos, arena)

    # Visual QC — labeled videos for spot-checking
    all_videos = [v for vlist in videos_by_arena.values() for v in vlist]
    create_labeled_videos(all_videos)
    print("\nCheck labeled videos in:", LABELED_DIR)
    print("If tracking looks poor, go back to dlc_train.py → extract_outliers_and_refine()")

    # Load, clean, and export per arena
    for arena in videos_by_arena:
        sessions = load_sessions_for_arena(arena)
        export_clean_csv(sessions, arena)

    print("\nNext step: run heatmap.py / path_analysis.py with the clean CSVs.")
