"""
show_frame_coords.py  (Plus/Cross Maze version)
------------------------------------------------
Four-phase interactive zone selector for plus maze videos.

Phase 1 — Click 4 corners of the BOTTOM arm  (start arm / stem)
Phase 2 — Click 4 corners of the LEFT arm
Phase 3 — Click 4 corners of the RIGHT arm
Phase 4 — Click 4 corners of the TOP arm

Plus maze layout (top-down view):
          [TOP ARM]
              ||
[LEFT ARM] ==JUNCTION== [RIGHT ARM]
              ||
         [BOTTOM ARM]

Usage:
    python show_frame_coords.py --video ../../data/raw_videos/Part1/MA1-1.avi
"""

import argparse
import json
import pathlib
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ZONES = ["bottom_arm", "left_arm", "right_arm", "top_arm"]
ZONE_COLORS = {
    "bottom_arm": "#4FC3F7",
    "left_arm":   "#A5D6A7",
    "right_arm":  "#FFCC80",
    "top_arm":    "#CE93D8",
}
ZONE_LABELS = {
    "bottom_arm": "PHASE 1 — BOTTOM arm (start arm)",
    "left_arm":   "PHASE 2 — LEFT arm",
    "right_arm":  "PHASE 3 — RIGHT arm",
    "top_arm":    "PHASE 4 — TOP arm",
}

phase_idx    = 0
zone_pts     = {z: [] for z in ZONES}
scatter_objs = []
rect_artists = []


def current_zone():
    return ZONES[phase_idx]


def update_title():
    z = current_zone()
    color = ZONE_COLORS[z]
    remaining = 4 - len(zone_pts[z])
    ax.set_title(
        f"{ZONE_LABELS[z]}  ({len(zone_pts[z])}/4 clicked, {remaining} remaining)\n"
        f"Click the 4 zone corners  |  Mouse coords shown dynamically",
        color=color, fontsize=11,
    )
    fig.canvas.draw_idle()


def draw_rect(pts, color, label=""):
    if len(pts) < 4:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2.0, edgecolor=color,
        facecolor=color, alpha=0.15, linestyle="--", zorder=10,
    )
    ax.add_patch(rect)
    rect_artists.append(rect)
    if label:
        ax.text(x0 + 4, y0 + 18, label, color=color,
                fontsize=9, fontweight="bold", zorder=11)
    fig.canvas.draw_idle()


def onmove(event):
    if event.inaxes is None or event.xdata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    z = current_zone()
    ax.set_title(
        f"{ZONE_LABELS[z]}  |  Mouse: ({x}, {y})\n"
        f"Click the 4 zone corners",
        color=ZONE_COLORS[z], fontsize=11,
    )
    fig.canvas.draw_idle()


def onclick(event):
    global phase_idx
    if event.inaxes is None or event.xdata is None:
        return
    if event.button != 1:
        return

    z   = current_zone()
    pts = zone_pts[z]
    if len(pts) >= 4:
        return

    x, y = int(event.xdata), int(event.ydata)
    pts.append((x, y))
    color = ZONE_COLORS[z]
    dot = ax.scatter(x, y, color=color, s=80, zorder=15,
                     edgecolors="black", linewidths=0.8)
    scatter_objs.append(dot)
    print(f"  {z} — Point {len(pts)}: ({x}, {y})")

    if len(pts) == 4:
        draw_rect(pts, color, label=z.replace("_", " "))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print(f"\n  {z} bounds: xmin={min(xs)}  xmax={max(xs)}  "
              f"ymin={min(ys)}  ymax={max(ys)}")
        if phase_idx < len(ZONES) - 1:
            phase_idx += 1
            print(f"\n>>> Phase {phase_idx + 1}: "
                  f"Click 4 corners for {ZONE_LABELS[ZONES[phase_idx]]}.\n")
        else:
            print("\n>>> All zones selected. Close the window to see your command.\n")

    update_title()
    fig.canvas.draw_idle()


def extract_first_frame(video_path: str) -> np.ndarray:
    import os
    if not os.path.exists(video_path):
        base = os.path.splitext(video_path)[0]
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            alt = base + ext
            if os.path.exists(alt):
                video_path = alt
                break
        else:
            raise ValueError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame from: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bounds_str(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return f"{min(xs)} {max(xs)} {min(ys)} {max(ys)}"


ROOT         = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON = ROOT / "data" / "arm_coords.json"


def main():
    parser = argparse.ArgumentParser(
        description="Plus maze zone selector (bottom + left + right + top)"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--save-json", default=str(DEFAULT_JSON), dest="save_json",
                        help=f"arm_coords.json kayit yolu (varsayilan: {DEFAULT_JSON})")
    args = parser.parse_args()

    print(f"Loading: {args.video}")
    frame = extract_first_frame(args.video)
    h, w = frame.shape[:2]
    print(f"Frame size: {w} x {h} px\n")
    print("PHASE 1: Click the 4 corners of the BOTTOM arm (start arm).")

    global fig, ax
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#1a1a1a")
    ax.set_facecolor("#0a0a0a")
    ax.imshow(frame)
    ax.set_xlabel("X (pixels)", color="#AAAAAA")
    ax.set_ylabel("Y (pixels)", color="#AAAAAA")
    ax.tick_params(colors="#666666")

    update_title()
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("motion_notify_event", onmove)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    all_complete = all(len(zone_pts[z]) == 4 for z in ZONES)

    for z in ZONES:
        pts = zone_pts[z]
        if len(pts) == 4:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            print(f"{z:>12}: xmin={min(xs)}  xmax={max(xs)}  "
                  f"ymin={min(ys)}  ymax={max(ys)}")
        else:
            print(f"{z:>12}: NOT FULLY SELECTED ({len(pts)}/4)")

    if not all_complete:
        print("\nRun again and click all 4 corners for each zone.")
        sys.exit(1)

    # ── JSON olarak kaydet ────────────────────────────────────────────────────
    coords = {}
    for z in ZONES:
        pts = zone_pts[z]
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        coords[z] = [min(xs), max(xs), min(ys), max(ys)]

    out_path = pathlib.Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(coords, fh, indent=2)
    print(f"\n[ok] Koordinatlar kaydedildi -> {out_path}")
    print("     orbit_plot.py ve ethological_features.py bu dosyayi otomatik okur.\n")

    b = bounds_str(zone_pts["bottom_arm"])
    l = bounds_str(zone_pts["left_arm"])
    r = bounds_str(zone_pts["right_arm"])
    t = bounds_str(zone_pts["top_arm"])

    print(f"tmaze_metrics.py icin komut:\n")
    print(f"  python analysis/plus_maze/tmaze_metrics.py \\")
    print(f"    --batch-dir data/DLCfiltered \\")
    print(f"    --bottom-arm {b} \\")
    print(f"    --left-arm   {l} \\")
    print(f"    --right-arm  {r} \\")
    print(f"    --top-arm    {t}")
    print("=" * 70)


if __name__ == "__main__":
    main()
