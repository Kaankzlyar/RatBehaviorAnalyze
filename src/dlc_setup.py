"""
dlc_setup.py
------------
Stage 2.1–2.3: Create DeepLabCut project, extract frames, and label them.

Steps covered:
    2.1  Project creation — register all 24 videos, patch config.yaml
    2.2  Frame extraction — kmeans-based diverse sampling (~20 per video)
    2.3  Frame labeling  — GUI labeling + visual verification

Usage:
    # Full pipeline (create → extract → label → verify):
    python dlc_setup.py

    # Resume labeling on an existing project:
    python dlc_setup.py --config <path/to/config.yaml>

    # Only verify labels (after labeling is done):
    python dlc_setup.py --config <path/to/config.yaml> --check-only

Requirements:
    pip install deeplabcut[torch]
    CUDA 11.8 or 12.1 + PyTorch 2.x (RTX 3060)
"""

import argparse
import os
import glob
import sys

import deeplabcut

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_NAME   = "rat_behavior"
EXPERIMENTER   = "kaank"
VIDEO_DIR      = "data/raw_videos"
WORKING_DIR    = "models/dlc_model"
VIDEO_EXT      = "avi"
FRAMES_PER_VID = 20   # frames to extract per video for labeling

EXPECTED_VIDEO_COUNT = 24

BODY_PARTS = [
    "nose", "head", "neck",
    "left_ear", "right_ear",
    "body_center",
    "left_forepaw", "right_forepaw",
    "left_hindpaw", "right_hindpaw",
    "tail_base",
]
SKELETON = [
    ["nose", "head"],
    ["head", "neck"],
    ["head", "left_ear"],
    ["head", "right_ear"],
    ["neck", "body_center"],
    ["neck", "left_forepaw"],
    ["neck", "right_forepaw"],
    ["body_center", "left_hindpaw"],
    ["body_center", "right_hindpaw"],
    ["body_center", "tail_base"],
]

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
    arena_counts = {}
    for part, arena in ARENA_MAP.items():
        part_vids = [v for v in videos if part in v.replace("\\", "/")]
        arena_counts[part] = len(part_vids)
        print(f"  {part} ({arena}): {len(part_vids)} video(s)")
    print(f"Total: {len(videos)} video(s) across all arenas")

    # Validate expected count
    if len(videos) != EXPECTED_VIDEO_COUNT:
        print(f"WARNING: Expected {EXPECTED_VIDEO_COUNT} videos, found {len(videos)}.")

    # Ensure both arena types are represented
    rect_count = arena_counts.get("Part0", 0)
    tmaze_count = sum(c for p, c in arena_counts.items() if p != "Part0")
    if rect_count == 0:
        print("WARNING: No rectangle arena (Part0) videos found.")
    if tmaze_count == 0:
        print("WARNING: No T-maze (Part1/Part2) videos found.")

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
    """
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["bodyparts"] = [
        "nose", "head", "neck",
        "left_ear", "right_ear",
        "body_center",
        "left_forepaw", "right_forepaw",
        "left_hindpaw", "right_hindpaw",
        "tail_base"
    ]

    cfg["skeleton"] = [
        ["nose", "head"],
        ["head", "neck"],
        ["head", "left_ear"],
        ["head", "right_ear"],
        ["neck", "body_center"],
        ["neck", "left_forepaw"],
        ["neck", "right_forepaw"],
        ["body_center", "left_hindpaw"],
        ["body_center", "right_hindpaw"],
        ["body_center", "tail_base"],
    ]
    cfg["numframes2pick"] = FRAMES_PER_VID

    # Marker size for GUI labeling
    cfg["dotsize"] = 6
    cfg["alphavalue"] = 0.9

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print("config.yaml patched with rat body parts and skeleton.")

# ─── VERIFY CONFIG ───────────────────────────────────────────────────────────

def verify_config(config_path: str) -> bool:
    """
    Stage 2.1 verification: confirm videos registered, body parts, and skeleton
    are correctly saved in config.yaml.
    """
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ok = True

    # Check body parts
    saved_parts = cfg.get("bodyparts", [])
    if saved_parts != BODY_PARTS:
        print(f"FAIL: bodyparts mismatch.\n  Expected: {BODY_PARTS}\n  Got:      {saved_parts}")
        ok = False
    else:
        print(f"OK: bodyparts = {saved_parts}")

    # Check skeleton
    saved_skel = cfg.get("skeleton", [])
    if saved_skel != SKELETON:
        print(f"FAIL: skeleton mismatch.\n  Expected: {SKELETON}\n  Got:      {saved_skel}")
        ok = False
    else:
        print(f"OK: skeleton = {len(saved_skel)} links")

    # Check video list
    video_sets = cfg.get("video_sets", {})
    n_videos = len(video_sets)
    if n_videos != EXPECTED_VIDEO_COUNT:
        print(f"WARNING: {n_videos} videos in config (expected {EXPECTED_VIDEO_COUNT})")
    else:
        print(f"OK: {n_videos} videos registered in config.yaml")

    return ok

# ─── EXTRACT FRAMES ──────────────────────────────────────────────────────────

def extract_frames(config_path: str) -> None:
    deeplabcut.extract_frames(
        config_path,
        mode="automatic",
        algo="kmeans",       # diverse frame selection
        userfeedback=False,
        crop=False,
    )
    print("Frame extraction complete.")
    _report_extracted_frames(config_path)

def _report_extracted_frames(config_path: str) -> None:
    """Report how many frames were extracted per arena type."""
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    project_dir = os.path.dirname(config_path)
    labeled_data_dir = os.path.join(project_dir, "labeled-data")

    if not os.path.isdir(labeled_data_dir):
        print("  labeled-data/ directory not found — skipping frame count.")
        return

    total = 0
    arena_totals = {"rectangle": 0, "tmaze": 0}
    for folder in sorted(os.listdir(labeled_data_dir)):
        folder_path = os.path.join(labeled_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        n_frames = len([f for f in os.listdir(folder_path) if f.endswith(".png")])
        total += n_frames
        # Determine arena type from folder name
        arena = "unknown"
        for part, a in ARENA_MAP.items():
            if part in folder:
                arena = a
                break
        if arena in arena_totals:
            arena_totals[arena] += n_frames

    print(f"\nExtracted frames summary:")
    print(f"  Rectangle: {arena_totals['rectangle']} frames")
    print(f"  T-maze:    {arena_totals['tmaze']} frames")
    print(f"  Total:     {total} frames")

    if arena_totals["rectangle"] == 0 or arena_totals["tmaze"] == 0:
        print("WARNING: Missing frames from one arena type — model needs both.")

# ─── LAUNCH LABELING GUI ────────────────────────────────────────────────────

def label_frames(config_path: str) -> None:
    print("\nLaunching labeling GUI...")
    print("Label all 5 body parts per frame: nose, head, neck, body_center, tail_base")
    print("Target: >= 10 frames from rectangle + >= 10 frames from T-maze")
    print("Shortcuts: A/D = prev/next frame | Ctrl+S = save | Left click = place label")
    deeplabcut.label_frames(config_path)

# ─── VERIFY LABELS ──────────────────────────────────────────────────────────

def check_labels(config_path: str) -> None:
    deeplabcut.check_labels(config_path, visualizeindividuals=True)
    print("Label check images saved to labeled-data/ folders.")
    print("Visually inspect the overlay images to confirm labels are correct.")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2.1–2.3: DLC project setup, frame extraction, and labeling.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to existing config.yaml (skip project creation, resume labeling).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run label verification (requires --config).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.check_only:
        if not args.config:
            print("ERROR: --check-only requires --config <path/to/config.yaml>")
            sys.exit(1)
        check_labels(args.config)
        sys.exit(0)

    if args.config:
        # Resume on existing project
        config_path = args.config
        if not os.path.isfile(config_path):
            print(f"ERROR: config not found: {config_path}")
            sys.exit(1)
        print(f"Resuming with existing project: {config_path}")
        verify_config(config_path)
    else:
        # Full setup: create project from scratch
        videos = get_videos(VIDEO_DIR, VIDEO_EXT)
        config_path = create_project(videos)
        patch_config(config_path)
        verify_config(config_path)
        extract_frames(config_path)
    label_frames(config_path)

    # Post-labeling verification
    check_labels(config_path)

    print(f"\nNext step: run dlc_train.py with config_path = '{config_path}'")
