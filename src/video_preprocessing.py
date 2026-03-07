"""
video_preprocessing.py
----------------------
Stage 01: Video Collection & Preprocessing

Tasks:
    1.1  Video inventory — scan all .avi files, verify they open, record properties
    1.2  Metadata CSV — parse filenames, map arena type, export data/metadata.csv
    1.3  Quality checks — resolution/fps consistency, brightness, blur, dropped frames
    1.4  Trim trial windows — optional, requires manual start/end frame markers
    1.5  Condition labels — assign experimental group via CONDITION_MAP

Usage:
    python src/video_preprocessing.py

Outputs:
    data/metadata.csv         — full video inventory with properties
    data/quality_report.csv   — per-video quality flags
    data/quality_plots/       — brightness histograms per video
"""

import os
import re
import glob
import json
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless rendering — no display needed
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────

VIDEO_DIR   = "D:/ProjectsD/ThesisWork/data/raw_videos"
DATA_DIR    = "D:/ProjectsD/ThesisWork/data"
PLOTS_DIR   = "D:/ProjectsD/ThesisWork/data/quality_plots"
VIDEO_EXT   = "avi"

# Arena type by subfolder
ARENA_MAP = {
    "Part0": "rectangle",
    "Part1": "tmaze",
    "Part2": "tmaze",
}

# ── Assign experimental conditions here ──────────────────────────────────────
# Key: rat_id (e.g. "MA1"), Value: condition label
# Update this once your experimental groups are defined.
CONDITION_MAP = {
    "MA1": "unknown",
    "MA3": "unknown",
    "MA5": "unknown",
    "MA7": "unknown",
}

# Expected video properties (used for consistency checks)
EXPECTED_FPS         = None   # set to e.g. 25 to flag deviations; None = auto-detect from first video
EXPECTED_RESOLUTION  = None   # set to e.g. (1280, 720); None = auto-detect
BLUR_THRESHOLD       = 80.0   # Laplacian variance below this → flagged as blurry
BRIGHTNESS_LOW       = 30     # Mean pixel value below this → flagged as too dark
BRIGHTNESS_HIGH      = 220    # Mean pixel value above this → flagged as overexposed
SAMPLE_FRAMES        = 10     # Number of evenly spaced frames sampled per video for QC

# ─── FILENAME PARSING ─────────────────────────────────────────────────────────

# Expects: MA{rat_id}-{session}_res.avi   e.g. MA1-2_res.avi
_FILENAME_RE = re.compile(r"MA(\d+)-(\d+)_res\.avi", re.IGNORECASE)

def parse_filename(filename: str) -> tuple[str, int] | tuple[None, None]:
    """Return (rat_id, session) parsed from filename, or (None, None) if unrecognised."""
    m = _FILENAME_RE.match(os.path.basename(filename))
    if m:
        return f"MA{m.group(1)}", int(m.group(2))
    return None, None

def get_part(filepath: str) -> str | None:
    """Return 'Part0', 'Part1', or 'Part2' based on which subfolder the file lives in."""
    normalized = filepath.replace("\\", "/")
    for part in ARENA_MAP:
        if f"/{part}/" in normalized:
            return part
    return None

# ─── VIDEO PROPERTIES ─────────────────────────────────────────────────────────

def get_video_properties(filepath: str) -> dict:
    """
    Open video with OpenCV and extract properties.
    Returns a dict with keys: width, height, fps, frame_count, duration_s,
    file_size_mb, readable.
    """
    props = {
        "width": None, "height": None, "fps": None,
        "frame_count": None, "duration_s": None,
        "file_size_mb": None, "readable": False,
    }

    props["file_size_mb"] = round(os.path.getsize(filepath) / (1024 ** 2), 2)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return props

    props["readable"]     = True
    props["fps"]          = cap.get(cv2.CAP_PROP_FPS)
    props["frame_count"]  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    props["width"]        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    props["height"]       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if props["fps"] and props["fps"] > 0:
        props["duration_s"] = round(props["frame_count"] / props["fps"], 2)

    cap.release()
    return props

# ─── QUALITY CHECKS ───────────────────────────────────────────────────────────

def sample_frames(filepath: str, n: int = SAMPLE_FRAMES) -> list[np.ndarray]:
    """Return n evenly spaced frames from the video as grayscale numpy arrays."""
    cap = cv2.VideoCapture(filepath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames

def check_blur(frames: list[np.ndarray]) -> float:
    """
    Mean Laplacian variance across sampled frames.
    Low value = blurry video / out-of-focus camera.
    """
    if not frames:
        return 0.0
    variances = [cv2.Laplacian(f, cv2.CV_64F).var() for f in frames]
    return float(np.mean(variances))

def check_brightness(frames: list[np.ndarray]) -> tuple[float, float]:
    """Return (mean_brightness, std_brightness) across sampled frames."""
    if not frames:
        return 0.0, 0.0
    means = [f.mean() for f in frames]
    return float(np.mean(means)), float(np.std(means))

def check_dropped_frames(props: dict) -> float:
    """
    Estimate dropped frame rate.
    Compares actual frame_count vs expected count from duration × fps.
    Returns fraction of dropped frames (0.0 = no drops).
    """
    if not props["fps"] or not props["duration_s"] or props["fps"] == 0:
        return 0.0
    expected = props["duration_s"] * props["fps"]
    actual   = props["frame_count"]
    if expected == 0:
        return 0.0
    return max(0.0, round((expected - actual) / expected, 4))

def save_brightness_histogram(frames: list[np.ndarray], label: str, out_dir: str) -> str:
    """Save a brightness histogram plot for a video; returns the saved path."""
    os.makedirs(out_dir, exist_ok=True)
    safe_label = label.replace("/", "_").replace("\\", "_")
    out_path   = os.path.join(out_dir, f"{safe_label}_brightness.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    for f in frames:
        ax.hist(f.ravel(), bins=64, range=(0, 255), alpha=0.3, color="steelblue", density=True)
    ax.set_xlim(0, 255)
    ax.axvline(BRIGHTNESS_LOW,  color="red",    linestyle="--", linewidth=1, label=f"Low ({BRIGHTNESS_LOW})")
    ax.axvline(BRIGHTNESS_HIGH, color="orange", linestyle="--", linewidth=1, label=f"High ({BRIGHTNESS_HIGH})")
    ax.set_title(f"Brightness distribution — {label}")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path

# ─── CONSISTENCY CHECKS ───────────────────────────────────────────────────────

def check_consistency(records: list[dict]) -> list[str]:
    """
    Compare all videos against the expected/majority resolution and fps.
    Returns a list of warning strings.
    """
    warnings_list = []

    resolutions = [(r["width"], r["height"]) for r in records if r["readable"]]
    fps_values  = [r["fps"] for r in records if r["readable"]]

    if resolutions:
        majority_res = max(set(resolutions), key=resolutions.count)
        for r in records:
            if r["readable"] and (r["width"], r["height"]) != majority_res:
                warnings_list.append(
                    f"  RESOLUTION MISMATCH: {r['filename']} is {r['width']}x{r['height']}"
                    f", expected {majority_res[0]}x{majority_res[1]}"
                )

    if fps_values:
        majority_fps = max(set(fps_values), key=fps_values.count)
        for r in records:
            if r["readable"] and r["fps"] != majority_fps:
                warnings_list.append(
                    f"  FPS MISMATCH: {r['filename']} is {r['fps']:.1f} fps"
                    f", expected {majority_fps:.1f} fps"
                )

    return warnings_list

# ─── TRIM TRIAL WINDOWS ───────────────────────────────────────────────────────

def trim_video(
    input_path: str,
    output_path: str,
    start_frame: int,
    end_frame: int,
) -> None:
    """
    Trim a video to [start_frame, end_frame] and save to output_path.
    Requires a markers CSV (see load_trial_markers()).
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def load_trial_markers(markers_csv: str) -> pd.DataFrame:
    """
    Load manually defined trial start/end markers.
    Expected columns: filename, start_frame, end_frame
    Returns DataFrame; empty if file does not exist.
    """
    if not os.path.exists(markers_csv):
        return pd.DataFrame(columns=["filename", "start_frame", "end_frame"])
    return pd.read_csv(markers_csv)

def batch_trim(metadata: pd.DataFrame, markers_csv: str) -> None:
    """
    Trim all videos that have an entry in markers_csv.
    Writes trimmed files to data/raw_videos_trimmed/.
    """
    markers   = load_trial_markers(markers_csv)
    if markers.empty:
        print("No trial markers found — skipping trim step.")
        print(f"  Create '{markers_csv}' with columns: filename, start_frame, end_frame")
        return

    trim_dir = os.path.join(DATA_DIR, "raw_videos_trimmed")
    os.makedirs(trim_dir, exist_ok=True)

    merged = metadata.merge(markers, on="filename", how="inner")
    print(f"\nTrimming {len(merged)} video(s)...")

    for _, row in merged.iterrows():
        part_subdir = os.path.join(trim_dir, row["part"])
        os.makedirs(part_subdir, exist_ok=True)
        out_path = os.path.join(part_subdir, row["filename"])
        print(f"  {row['filename']}  frames {row['start_frame']}→{row['end_frame']}")
        trim_video(row["file_path"], out_path, int(row["start_frame"]), int(row["end_frame"]))

    print(f"Trimmed videos saved to: {trim_dir}")

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def build_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all videos, run quality checks, and return:
        (metadata_df, quality_report_df)
    """
    video_paths = glob.glob(
        os.path.join(VIDEO_DIR, "**", f"*.{VIDEO_EXT}"), recursive=True
    )
    if not video_paths:
        raise FileNotFoundError(f"No .{VIDEO_EXT} videos found under {VIDEO_DIR}")

    print(f"Found {len(video_paths)} video(s)\n")

    metadata_rows = []
    quality_rows  = []

    global EXPECTED_FPS, EXPECTED_RESOLUTION

    for filepath in sorted(video_paths):
        filename        = os.path.basename(filepath)
        rat_id, session = parse_filename(filename)
        part            = get_part(filepath)
        arena           = ARENA_MAP.get(part, "unknown") if part else "unknown"
        condition       = CONDITION_MAP.get(rat_id, "unknown") if rat_id else "unknown"

        props = get_video_properties(filepath)

        # Set expected values from first readable video if not configured
        if props["readable"]:
            if EXPECTED_FPS is None:
                EXPECTED_FPS = props["fps"]
            if EXPECTED_RESOLUTION is None:
                EXPECTED_RESOLUTION = (props["width"], props["height"])

        # Quality analysis
        frames          = sample_frames(filepath) if props["readable"] else []
        blur_score      = check_blur(frames)
        mean_bright, std_bright = check_brightness(frames)
        drop_rate       = check_dropped_frames(props)

        # Flags
        flag_unreadable  = not props["readable"]
        flag_empty       = props["file_size_mb"] is not None and props["file_size_mb"] < 1.0
        flag_blur        = blur_score < BLUR_THRESHOLD and bool(frames)
        flag_dark        = mean_bright < BRIGHTNESS_LOW and bool(frames)
        flag_overexposed = mean_bright > BRIGHTNESS_HIGH and bool(frames)
        flag_drops       = drop_rate > 0.01   # > 1% frames missing
        flag_parse       = rat_id is None

        any_flag = any([flag_unreadable, flag_empty, flag_blur,
                        flag_dark, flag_overexposed, flag_drops, flag_parse])

        resolution_str = (
            f"{props['width']}x{props['height']}" if props["readable"] else "N/A"
        )

        metadata_rows.append({
            "rat_id":       rat_id,
            "session":      session,
            "arena":        arena,
            "part":         part,
            "condition":    condition,
            "filename":     filename,
            "duration_s":   props["duration_s"],
            "fps":          props["fps"],
            "resolution":   resolution_str,
            "frame_count":  props["frame_count"],
            "file_size_mb": props["file_size_mb"],
            "file_path":    filepath.replace("\\", "/"),
        })

        quality_rows.append({
            "filename":        filename,
            "rat_id":          rat_id,
            "part":            part,
            "readable":        props["readable"],
            "blur_score":      round(blur_score, 2),
            "mean_brightness": round(mean_bright, 2),
            "std_brightness":  round(std_bright, 2),
            "drop_rate":       drop_rate,
            "flag_unreadable": flag_unreadable,
            "flag_empty":      flag_empty,
            "flag_blur":       flag_blur,
            "flag_dark":       flag_dark,
            "flag_overexposed":flag_overexposed,
            "flag_drops":      flag_drops,
            "flag_parse_error":flag_parse,
            "any_flag":        any_flag,
        })

        # Save brightness histogram
        if frames:
            label = f"{part}_{os.path.splitext(filename)[0]}"
            save_brightness_histogram(frames, label, PLOTS_DIR)

        status = "OK" if not any_flag else "FLAG"
        print(
            f"  [{status}] {filename:25s} | {resolution_str:10s} | "
            f"{props['fps'] or 0:.1f}fps | {props['duration_s'] or 0:.1f}s | "
            f"blur={blur_score:.0f} | bright={mean_bright:.0f}"
        )

    metadata_df = pd.DataFrame(metadata_rows).sort_values(["part", "rat_id", "session"])
    quality_df  = pd.DataFrame(quality_rows).sort_values(["part", "filename"])

    return metadata_df, quality_df


def print_summary(metadata_df: pd.DataFrame, quality_df: pd.DataFrame) -> None:
    """Print a structured summary report to the console."""
    total       = len(metadata_df)
    readable    = quality_df["readable"].sum()
    flagged     = quality_df["any_flag"].sum()

    print("\n" + "=" * 60)
    print("VIDEO INVENTORY SUMMARY")
    print("=" * 60)
    print(f"  Total videos found   : {total}")
    print(f"  Readable             : {readable}")
    print(f"  Flagged for review   : {flagged}")

    print("\n  By Part:")
    for part, group in metadata_df.groupby("part"):
        arena = ARENA_MAP.get(part, "?")
        print(f"    {part} ({arena:10s}) : {len(group)} videos")

    print("\n  By Rat:")
    for rat, group in metadata_df.groupby("rat_id"):
        print(f"    {rat} : {len(group)} sessions across {group['part'].nunique()} part(s)")

    resolutions = metadata_df["resolution"].dropna().unique()
    fps_values  = metadata_df["fps"].dropna().unique()
    print(f"\n  Resolutions seen : {sorted(resolutions)}")
    print(f"  FPS values seen  : {sorted(fps_values)}")

    if flagged > 0:
        print(f"\n  ⚠  Flagged videos ({int(flagged)}):")
        flagged_rows = quality_df[quality_df["any_flag"]]
        for _, row in flagged_rows.iterrows():
            reasons = [
                col.replace("flag_", "")
                for col in ["flag_unreadable", "flag_empty", "flag_blur",
                            "flag_dark", "flag_overexposed", "flag_drops",
                            "flag_parse_error"]
                if row[col]
            ]
            print(f"    {row['filename']:30s} → {', '.join(reasons)}")
    else:
        print("\n  All videos passed quality checks.")

    print("=" * 60)

    consistency_warnings = check_consistency(
        metadata_df.to_dict(orient="records")
    )
    if consistency_warnings:
        print("\n  Consistency warnings:")
        for w in consistency_warnings:
            print(w)


def save_outputs(metadata_df: pd.DataFrame, quality_df: pd.DataFrame) -> None:
    """Write metadata.csv and quality_report.csv to data/."""
    os.makedirs(DATA_DIR, exist_ok=True)

    meta_path    = os.path.join(DATA_DIR, "metadata.csv")
    quality_path = os.path.join(DATA_DIR, "quality_report.csv")

    metadata_df.to_csv(meta_path,    index=False)
    quality_df.to_csv(quality_path,  index=False)

    print(f"\n  Metadata saved   : {meta_path}")
    print(f"  Quality report   : {quality_path}")
    print(f"  Brightness plots : {PLOTS_DIR}/")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Stage 01 — Video Preprocessing\n")

    # 1.1 + 1.2 + 1.3 Build inventory and run quality checks
    metadata_df, quality_df = build_inventory()

    # Print console summary
    print_summary(metadata_df, quality_df)

    # Save outputs
    save_outputs(metadata_df, quality_df)

    # 1.4 Trim trial windows (optional)
    # Create data/trial_markers.csv with columns: filename, start_frame, end_frame
    # Then uncomment the line below:
    # batch_trim(metadata_df, os.path.join(DATA_DIR, "trial_markers.csv"))

    # 1.5 Condition labels
    # Edit CONDITION_MAP at the top of this file with your experimental group assignments.
    undefined = [r for r, c in CONDITION_MAP.items() if c == "unknown"]
    if undefined:
        print(f"\n  NOTE: Condition labels not yet assigned for: {undefined}")
        print("  Edit CONDITION_MAP in this script and re-run to update metadata.csv")

    print("\nNext step: run src/dlc_setup.py")
