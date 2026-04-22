"""
show_frame_coords.py
-------------------
Two-phase interactive arena boundary selector.

Phase 1 — Click 4 arena corners  (outer wall)
Phase 2 — Click 4 inner zone corners  (thigmotaxis boundary)

At the end prints the exact orbit_plot.py command with your selections.

Usage:
    python show_frame_coords.py --video ../data/DLCfiltered/OpenFieldMA5_1/MA5-1_res.avi
"""

import argparse
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# --- STATE --------------------------------------------------------------------

phase        = 1          # 1 = arena corners, 2 = inner zone corners
arena_pts    = []         # up to 4 (x, y)
inner_pts    = []         # up to 4 (x, y)
scatter_objs = []         # dot artists for removal
rect_artists = []         # rectangle artists


def current_pts():
    return arena_pts if phase == 1 else inner_pts


def update_title():
    if phase == 1:
        remaining = 4 - len(arena_pts)
        msg = (
            f"PHASE 1 — Arena corners  ({len(arena_pts)}/4 clicked, {remaining} remaining)\n"
            f"Click the 4 arena wall corners  |  Mouse: move to read coords"
        )
        color = "white"
    else:
        remaining = 4 - len(inner_pts)
        msg = (
            f"PHASE 2 — Inner zone corners  ({len(inner_pts)}/4 clicked, {remaining} remaining)\n"
            f"Click the 4 thigmotaxis inner boundary corners"
        )
        color = "#FFD700"
    ax.set_title(msg, color=color, fontsize=11)
    fig.canvas.draw_idle()


def draw_rect(pts, color, linestyle="--", lw=2.0, label=""):
    """Draw a rectangle from a list of (x,y) points."""
    if len(pts) < 4:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=lw, edgecolor=color,
        facecolor="none", linestyle=linestyle, zorder=10,
    )
    ax.add_patch(rect)
    rect_artists.append(rect)
    if label:
        ax.text(x0 + 4, y0 + 14, label, color=color, fontsize=8, zorder=11)
    fig.canvas.draw_idle()


# --- EVENTS -------------------------------------------------------------------

def onmove(event):
    if event.inaxes is None or event.xdata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    phase_str = "Phase 1 (Arena)" if phase == 1 else "Phase 2 (Inner zone)"
    ax.set_title(
        f"{phase_str}  |  Mouse: ({x}, {y})\n"
        + ("Click 4 arena wall corners" if phase == 1
           else "Click 4 inner thigmotaxis corners"),
        color="white" if phase == 1 else "#FFD700",
        fontsize=11,
    )
    fig.canvas.draw_idle()


def onclick(event):
    global phase

    if event.inaxes is None or event.xdata is None:
        return
    if event.button != 1:  # left click only
        return

    x, y = int(event.xdata), int(event.ydata)
    pts = current_pts()

    if len(pts) >= 4:
        return  # already have 4 for this phase

    pts.append((x, y))
    color = "white" if phase == 1 else "#FFD700"
    dot = ax.scatter(x, y, color=color, s=80, zorder=15,
                     edgecolors="black", linewidths=0.8)
    scatter_objs.append(dot)
    print(f"  Phase {phase} — Point {len(pts)}: ({x}, {y})")

    # Draw box once 4 corners selected
    if len(pts) == 4:
        if phase == 1:
            draw_rect(arena_pts, color="white",   linestyle="--", label="arena")
            print(f"\n  Arena bounds: xmin={min(p[0] for p in arena_pts)}, "
                  f"xmax={max(p[0] for p in arena_pts)}, "
                  f"ymin={min(p[1] for p in arena_pts)}, "
                  f"ymax={max(p[1] for p in arena_pts)}")
            print("\n>>> Phase 1 done. Now click 4 inner zone corners (thigmotaxis boundary).\n")
            phase = 2
        else:
            draw_rect(inner_pts, color="#FFD700", linestyle=":",  label="inner zone")
            print(f"\n  Inner zone: xmin={min(p[0] for p in inner_pts)}, "
                  f"xmax={max(p[0] for p in inner_pts)}, "
                  f"ymin={min(p[1] for p in inner_pts)}, "
                  f"ymax={max(p[1] for p in inner_pts)}")
            print("\n>>> Phase 2 done. Close the window to get your command.\n")

    update_title()
    fig.canvas.draw_idle()


# --- MAIN ---------------------------------------------------------------------

def extract_first_frame(video_path: str) -> np.ndarray:
    import os
    from pathlib import Path
    
    # Check if file exists
    if not os.path.exists(video_path):
        # Try alternative extensions
        base_path = os.path.splitext(video_path)[0]
        alternatives = [
            base_path + ".mp4",
            base_path + ".avi",
            base_path + ".mov",
            base_path + ".mkv",
        ]
        
        print(f"\n❌ File not found: {video_path}")
        print(f"\nLooking for alternatives in: {os.path.dirname(video_path)}")
        
        found = False
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"   ✓ Found: {alt}")
                video_path = alt
                found = True
                break
            else:
                print(f"   ✗ Not found: {alt}")
        
        if not found:
            # List all files in the directory
            dir_path = os.path.dirname(video_path)
            if os.path.isdir(dir_path):
                print(f"\nAvailable files in {dir_path}:")
                try:
                    files = os.listdir(dir_path)
                    for f in sorted(files):
                        print(f"   {f}")
                except Exception as e:
                    print(f"   Error listing directory: {e}")
            raise ValueError(f"Could not find video file: {video_path}\n"
                           f"Tried extensions: {', '.join(['.mp4', '.avi', '.mov', '.mkv'])}")
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame from: {video_path}\n"
                        f"The file exists but OpenCV cannot open it.\n"
                        f"Check if it's a supported video format or corrupted.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="Interactive arena + inner zone selector")
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    print(f"Loading: {args.video}")
    frame = extract_first_frame(args.video)
    h, w = frame.shape[:2]
    print(f"Frame size: {w} × {h} px\n")
    print("PHASE 1: Click the 4 arena wall corners (any order).")

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

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 65)

    if len(arena_pts) < 4:
        print("Arena corners not fully selected. Run again and click 4 corners.")
        sys.exit(1)

    ax_min = min(p[0] for p in arena_pts)
    ax_max = max(p[0] for p in arena_pts)
    ay_min = min(p[1] for p in arena_pts)
    ay_max = max(p[1] for p in arena_pts)

    print(f"Arena bounds:     xmin={ax_min}  xmax={ax_max}  ymin={ay_min}  ymax={ay_max}")

    if len(inner_pts) == 4:
        ix_min = min(p[0] for p in inner_pts)
        ix_max = max(p[0] for p in inner_pts)
        iy_min = min(p[1] for p in inner_pts)
        iy_max = max(p[1] for p in inner_pts)
        print(f"Inner zone:       xmin={ix_min}  xmax={ix_max}  ymin={iy_min}  ymax={iy_max}")
        print(f"\nRun orbit_plot.py with these exact bounds:")
        print(f"\n  python orbit_plot.py \\\n"
              f"    --arena {ax_min} {ax_max} {ay_min} {ay_max} \\\n"
              f"    --inner-zone {ix_min} {ix_max} {iy_min} {iy_max}")
    else:
        print("Inner zone not selected.")
        print(f"\n  python orbit_plot.py --arena {ax_min} {ax_max} {ay_min} {ay_max}")

    print("=" * 65)


if __name__ == "__main__":
    main()
