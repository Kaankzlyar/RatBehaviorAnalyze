"""
dlc_inference.py
----------------
Step 5: Run trained DLC model on all videos, filter predictions,
        generate labeled videos for QC, and export clean tracking CSVs.
Run after training is complete in dlc_train.py.
"""

import os
import glob
import numpy as np
import pandas as pd
import deeplabcut

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CONFIG_PATH   = "D:/ProjectsD/ThesisWork/models/dlc_model/rat_tmaze-kaank-YYYY-MM-DD/config.yaml"
VIDEO_DIR     = "D:/ProjectsD/ThesisWork/data/raw_videos"
OUTPUT_DIR    = "D:/ProjectsD/ThesisWork/data/dlc_output"
LABELED_DIR   = "D:/ProjectsD/ThesisWork/data/dlc_output/labeled_videos"
VIDEO_EXT     = "mp4"
SHUFFLE       = 1
GPU_ID        = 0
LIKELIHOOD_THRESH = 0.6   # frames below this confidence are set to NaN

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABELED_DIR, exist_ok=True)

# ─── COLLECT VIDEOS ───────────────────────────────────────────────────────────

def get_videos() -> list[str]:
    videos = glob.glob(os.path.join(VIDEO_DIR, f"*.{VIDEO_EXT}"))
    if not videos:
        raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")
    print(f"Found {len(videos)} video(s) for inference.")
    return videos

# ─── RUN INFERENCE ────────────────────────────────────────────────────────────

def analyze_videos(videos: list[str]) -> None:
    print("Running DLC inference...")
    deeplabcut.analyze_videos(
        CONFIG_PATH,
        videos,
        videotype=VIDEO_EXT,
        shuffle=SHUFFLE,
        save_as_csv=True,
        destfolder=OUTPUT_DIR,
        gputouse=GPU_ID,
    )
    print(f"Tracking data saved to: {OUTPUT_DIR}")

# ─── FILTER PREDICTIONS ───────────────────────────────────────────────────────

def filter_predictions(videos: list[str]) -> None:
    """Smooth tracking with median filter and flag low-confidence frames."""
    deeplabcut.filterpredictions(
        CONFIG_PATH,
        videos,
        videotype=VIDEO_EXT,
        shuffle=SHUFFLE,
        filtertype="median",
        windowlength=5,      # 5-frame median window
        destfolder=OUTPUT_DIR,
    )
    print("Filtered predictions saved.")

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

    pct_valid = {bp: np.sum(~np.isnan(tracking[bp]["x"])) / len(tracking[bp]["x"]) * 100
                 for bp in bodyparts}
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
            arr  = pd.Series(tracking[bp][coord])
            arr  = arr.interpolate(method="linear", limit=MAX_GAP, limit_direction="both")
            tracking[bp][coord] = arr.values
    return tracking

def load_all_sessions() -> dict[str, dict]:
    """
    Load all filtered DLC CSVs from OUTPUT_DIR.
    Returns dict keyed by video name.
    """
    # Filtered CSVs have '_filtered' suffix
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*filtered*.csv"))
    if not csv_files:
        # Fall back to unfiltered
        csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))

    sessions = {}
    print(f"\nLoading {len(csv_files)} tracking file(s)...")
    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        tracking = load_tracking(csv_path)
        tracking = interpolate_gaps(tracking)
        sessions[name] = tracking

    return sessions

# ─── EXPORT CLEAN CSV ─────────────────────────────────────────────────────────

def export_clean_csv(sessions: dict, out_dir: str = OUTPUT_DIR) -> None:
    """
    Export a clean, flat CSV per session:
    columns = frame, nose_x, nose_y, head_x, head_y, ...
    """
    clean_dir = os.path.join(out_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)

    for name, tracking in sessions.items():
        bodyparts = list(tracking.keys())
        n_frames  = len(tracking[bodyparts[0]]["x"])

        rows = {"frame": np.arange(n_frames)}
        for bp in bodyparts:
            rows[f"{bp}_x"] = tracking[bp]["x"]
            rows[f"{bp}_y"] = tracking[bp]["y"]
            rows[f"{bp}_likelihood"] = tracking[bp]["likelihood"]

        df = pd.DataFrame(rows)
        out_path = os.path.join(clean_dir, f"{name}_clean.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print(f"\nClean CSVs saved to: {clean_dir}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    videos = get_videos()

    # Run tracking
    analyze_videos(videos)
    filter_predictions(videos)

    # Visual QC — check a few labeled videos before proceeding
    create_labeled_videos(videos)
    print("\nCheck labeled videos in:", LABELED_DIR)
    print("If tracking looks good, continue. If poor, go back to dlc_train.py → extract_outliers_and_refine()")

    # Load and clean all tracking data
    sessions = load_all_sessions()
    export_clean_csv(sessions)

    print("\nNext step: run heatmap.py with the clean CSVs.")
