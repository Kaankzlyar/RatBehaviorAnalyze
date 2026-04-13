"""
dlc_setup.py
------------
Step 1 & 2: Create DeepLabCut project and extract frames for labeling.
Run this once at the start of your project.

Requirements:
    pip install deeplabcut[torch]
    CUDA 11.8 or 12.1 + PyTorch 2.x (RTX 3060)
"""

import os
import glob
import deeplabcut

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_NAME   = "rat_behavior"
EXPERIMENTER   = "kaank"
VIDEO_DIR      = "data/raw_videos"
WORKING_DIR    = "models/dlc_model"
VIDEO_EXT      = "avi"
FRAMES_PER_VID = 20   # frames to extract per video for labeling

# Arena mapping — used downstream for analysis routing
# Part0 = rectangle open field arena
# Part1 = T-maze (MA1, MA3)
# Part2 = T-maze (MA5, MA7)
ARENA_MAP = {
    "Part0": "rectangle",
    "Part1": "tmaze",
    "Part2": "tmaze",
}

# ─── COLLECT VIDEOS ───────────────────────────────────────────────────────────

def get_videos(video_dir: str, ext: str = "avi") -> list[str]:
    """
    Recursively collect all videos from Part0/, Part1/, Part2/ subdirectories.
    All parts are included in the DLC project so the model trains on diverse
    appearances from both rectangle and T-maze arenas.
    """
    videos = glob.glob(os.path.join(video_dir, "**", f"*.{ext}"), recursive=True)
    if not videos:
        raise FileNotFoundError(f"No .{ext} files found recursively under {video_dir}")
    videos = [os.path.abspath(v) for v in videos]
    # Group by arena for informational output
    for part, arena in ARENA_MAP.items():
        part_vids = [v for v in videos if part in v.replace("\\", "/")]
        print(f"  {part} ({arena}): {len(part_vids)} video(s)")
    print(f"Total: {len(videos)} video(s) across all arenas")
    return videos

# ─── CREATE PROJECT ───────────────────────────────────────────────────────────

def create_project(videos: list[str]) -> str:
    config_path = deeplabcut.create_new_project(
        PROJECT_NAME,
        EXPERIMENTER,
        videos,
        working_directory=WORKING_DIR,
        copy_videos=False,
        videotype=VIDEO_EXT,
    )
    print(f"\nProject created.\nConfig: {config_path}")
    return config_path

# ─── PATCH CONFIG ─────────────────────────────────────────────────────────────

def patch_config(config_path: str) -> None:
    """
    Update config.yaml with rat-specific body parts, skeleton, and settings.
    DeepLabCut uses ruamel.yaml internally — edit via its own API to be safe.
    """
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["bodyparts"] = ["nose", "head", "body_center", "tail_base"]

    cfg["skeleton"] = [
        ["nose", "head"],
        ["head", "body_center"],
        ["body_center", "tail_base"],
    ]

    cfg["numframes2pick"] = FRAMES_PER_VID

    # Marker size for GUI labeling
    cfg["dotsize"] = 6
    cfg["alphavalue"] = 0.9

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print("config.yaml patched with rat body parts and skeleton.")

# ─── EXTRACT FRAMES ───────────────────────────────────────────────────────────

def extract_frames(config_path: str) -> None:
    deeplabcut.extract_frames(
        config_path,
        mode="automatic",
        algo="kmeans",       # diverse frame selection
        userfeedback=False,
        crop=False,
    )
    print("Frame extraction complete. Ready for labeling.")

# ─── LAUNCH LABELING GUI ──────────────────────────────────────────────────────

def label_frames(config_path: str) -> None:
    print("\nLaunching labeling GUI...")
    print("Shortcuts: A/D = prev/next frame | Ctrl+S = save | Left click = place label")
    deeplabcut.label_frames(config_path)

# ─── VERIFY LABELS ────────────────────────────────────────────────────────────

def check_labels(config_path: str) -> None:
    deeplabcut.check_labels(config_path, visualizeindividuals=True)
    print("Label check images saved to labeled-data/ folders.")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    videos = get_videos(VIDEO_DIR, VIDEO_EXT)

    # ── Step 1: Create project (skip if already exists) ───────────────────────
    config_path = os.path.join(
        os.path.abspath(WORKING_DIR),
        f"{PROJECT_NAME}-{EXPERIMENTER}-2026-04-01",
        "config.yaml",
    )
    if os.path.exists(config_path):
        print(f"Project already exists, using: {config_path}")
    else:
        config_path = create_project(videos)
        patch_config(config_path)

    # ── Step 2: Extract frames (skip if already done) ─────────────────────────
    labeled_data_dir = os.path.join(os.path.dirname(config_path), "labeled-data")
    existing_frames = glob.glob(os.path.join(labeled_data_dir, "**", "*.png"), recursive=True)
    if existing_frames:
        print(f"Frames already extracted ({len(existing_frames)} found). Skipping extraction.")
    else:
        extract_frames(config_path)

    # ── Step 3: Launch labeling GUI ───────────────────────────────────────────
    print(f"\nFrames are in: {labeled_data_dir}")
    print("To label frames, run in a new terminal:")
    print(f'  python -c "import deeplabcut; deeplabcut.label_frames(r\'{config_path}\')"')
    label_frames(config_path)

    # After labeling — verify visually
    check_labels(config_path)

    print(f"\nNext step: run dlc_train.py with config_path = '{config_path}'")
