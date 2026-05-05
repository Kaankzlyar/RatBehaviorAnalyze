"""
Interactive bout-labelling tool for ground-truth construction.

Recommended workflow
--------------------
1. Open with a video that has weak labels (rule-based bouts) already.
2. Press ``A`` to accept ALL weak labels as a starting point.
3. Scrub through the timeline strip; for each suspect bout, jump to it,
   press ``u`` to undo if wrong, or extend with a fresh manual bout.
4. Add any missing rearing/grooming bouts that the rule-based detector missed.
5. Press ``s`` periodically to save; ``q`` to save and quit.

Usage
-----
    python analysis/label_bouts.py \\
        --subject OpenFieldMA1_1 \\
        --video /path/to/OpenFieldMA1_1.avi

Auto-resolves the DLC CSV under ``data/DLCfiltered/<group>/<subject>/<subject>.csv``
and writes/appends bouts to ``data/behavior_ground_truth.csv``.

Keys
----
    SPACE          play / pause
    .   /  ,       +1 / −1 frame
    >   /  <       +30 / −30 frames (1 s)
    j              jump to typed frame number
    b              start a new bout at current frame (label set with r/g/l/i)
    r / g / l / i  start a bout (or set label on in-progress) —
                   rearing / grooming / locomotion / immobile
    e              end the in-progress bout at current frame
    a              accept the weak (rule-based) bout overlapping cursor
    A              accept ALL weak bouts at once
    u              undo last saved bout
    s              save GT CSV
    q              save and quit
    h              print this help

Output schema (data/behavior_ground_truth.csv)
----------------------------------------------
    subject_id, start_frame, end_frame, label, source
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "DLCfiltered"
GT_PATH = ROOT / "data" / "behavior_ground_truth.csv"

LABEL_KEYS = {
    ord("r"): "rearing",
    ord("g"): "grooming",
    ord("l"): "locomotion",
    ord("i"): "immobile",
}
LABEL_COLORS = {
    "rearing": (0, 0, 255),
    "grooming": (0, 200, 0),
    "locomotion": (255, 200, 0),
    "immobile": (180, 180, 180),
    "other": (80, 80, 80),
}

HELP_TEXT = (
    "SPACE play/pause | . , step ±1 | > < step ±30 | j jump | "
    "b start | r/g/l/i label | e end | a accept weak | A accept ALL weak | "
    "u undo | s save | q quit | h help"
)


# ── path resolution ──────────────────────────────────────────────────────────

def find_subject_dir(subject: str) -> Path:
    for group in DATA_DIR.iterdir():
        if not group.is_dir():
            continue
        cand = group / subject
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"subject {subject!r} not found under {DATA_DIR}")


# ── DLC keypoints ────────────────────────────────────────────────────────────

def load_dlc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = [f"{bp}_{coord}" for _, bp, coord in df.columns]
    return df


def overlay_keypoints(frame: np.ndarray, dlc_row: pd.Series, thresh: float = 0.6) -> None:
    bps = sorted({c.rsplit("_", 1)[0] for c in dlc_row.index})
    for bp in bps:
        x = dlc_row.get(f"{bp}_x")
        y = dlc_row.get(f"{bp}_y")
        lik = dlc_row.get(f"{bp}_likelihood", 1.0)
        if pd.isna(x) or pd.isna(y) or lik < thresh:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(frame, bp[:3], (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


# ── ground truth I/O ─────────────────────────────────────────────────────────

GT_COLS = ["subject_id", "start_frame", "end_frame", "label", "source"]


def load_existing_gt(subject: str) -> pd.DataFrame:
    if not GT_PATH.exists():
        return pd.DataFrame(columns=GT_COLS)
    gt = pd.read_csv(GT_PATH)
    if "subject_id" not in gt.columns:
        return pd.DataFrame(columns=GT_COLS)
    return gt[gt["subject_id"] == subject].copy().reset_index(drop=True)


def save_gt(subject: str, bouts: pd.DataFrame) -> None:
    if GT_PATH.exists():
        full = pd.read_csv(GT_PATH)
        full = full[full["subject_id"] != subject]
    else:
        full = pd.DataFrame(columns=GT_COLS)
    out = pd.concat([full, bouts], ignore_index=True)
    out = out.sort_values(["subject_id", "start_frame"]).reset_index(drop=True)
    out.to_csv(GT_PATH, index=False)
    print(f"[save] {GT_PATH.relative_to(ROOT)}: {len(out)} rows total ({len(bouts)} for {subject})")


def append_bout(bouts: pd.DataFrame, subject: str, start: int, end: int,
                label: str, source: str) -> pd.DataFrame:
    if end < start:
        start, end = end, start
    row = pd.DataFrame([{
        "subject_id": subject,
        "start_frame": int(start),
        "end_frame": int(end),
        "label": label,
        "source": source,
    }])
    return pd.concat([bouts, row], ignore_index=True)


# ── timeline strip ───────────────────────────────────────────────────────────

def render_timeline(width: int, n_frames: int, cur_frame: int,
                    bouts: pd.DataFrame, weak: pd.DataFrame | None) -> np.ndarray:
    h_manual, h_weak, h_cursor = 18, 10, 4
    strip = np.full((h_manual + h_weak + h_cursor, width, 3), 30, dtype=np.uint8)

    def draw_band(ax_y0, ax_h, df, alpha=1.0):
        for _, row in df.iterrows():
            x0 = int(row["start_frame"] / max(n_frames, 1) * width)
            x1 = int(row["end_frame"] / max(n_frames, 1) * width)
            color = (np.array(LABEL_COLORS.get(row["label"], (200, 200, 200))) * alpha).astype(int)
            cv2.rectangle(strip, (x0, ax_y0), (max(x1, x0 + 1), ax_y0 + ax_h),
                          color.tolist(), -1)

    if len(bouts):
        draw_band(0, h_manual, bouts, alpha=1.0)
    if weak is not None and len(weak):
        draw_band(h_manual, h_weak, weak, alpha=0.5)

    cx = int(cur_frame / max(n_frames - 1, 1) * width)
    cv2.line(strip, (cx, 0), (cx, strip.shape[0]), (255, 255, 255), 1)
    return strip


# ── main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g. OpenFieldMA1_1")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--csv", type=Path, default=None,
                   help="DLC CSV (auto-resolved if omitted)")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip DLC keypoint overlay (faster)")
    args = p.parse_args()

    subject = args.subject
    if not args.video.exists():
        print(f"[err] video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.csv is None:
        subj_dir = find_subject_dir(subject)
        args.csv = subj_dir / f"{subject}.csv"
    if not args.csv.exists():
        print(f"[err] DLC csv not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] video={args.video.name}  csv={args.csv.name}")
    dlc = load_dlc(args.csv) if not args.no_overlay else None

    bouts_csv = args.csv.parent / f"{subject}_behavior_bouts.csv"
    weak = pd.DataFrame()
    if bouts_csv.exists():
        wraw = pd.read_csv(bouts_csv)
        if {"start_frame", "end_frame"}.issubset(wraw.columns):
            label_col = "behavior" if "behavior" in wraw.columns else (
                "label" if "label" in wraw.columns else None)
            if label_col:
                weak = wraw.rename(columns={label_col: "label"})[["start_frame", "end_frame", "label"]].copy()
        print(f"[load] weak labels from {bouts_csv.name}: {len(weak)} bouts")

    bouts = load_existing_gt(subject)
    print(f"[load] existing GT for {subject}: {len(bouts)} bouts")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[err] could not open video", file=sys.stderr)
        sys.exit(1)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[load] frames={n_frames} fps={fps:.1f} size={w}x{h}")
    print(HELP_TEXT)

    cur = 0
    playing = False
    in_progress: dict | None = None  # {"start": int, "label": str|None}
    win = "label_bouts (h for help)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def grab(idx: int) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        return fr if ok else None

    while True:
        frame = grab(cur)
        if frame is None:
            cur = max(0, min(cur, n_frames - 1))
            frame = grab(cur)
            if frame is None:
                break

        if dlc is not None and cur < len(dlc):
            overlay_keypoints(frame, dlc.iloc[cur])

        ip_str = "-" if in_progress is None else f"{in_progress['label'] or '?'} from {in_progress['start']}"
        hud = f"frame {cur}/{n_frames - 1}  t={cur/fps:6.2f}s  bouts={len(bouts)}  in-progress={ip_str}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        timeline = render_timeline(frame.shape[1], n_frames, cur, bouts, weak)
        canvas = np.vstack([frame, timeline])
        cv2.imshow(win, canvas)

        key_wait = max(1, int(1000 / fps)) if playing else 0
        k = cv2.waitKey(key_wait) & 0xFFFF

        # quit/save
        if k in (27, ord("q")):
            save_gt(subject, bouts)
            break
        elif k == ord("h"):
            print(HELP_TEXT)
            continue
        elif k == ord(" "):
            playing = not playing
            continue

        # navigation: ASCII keys are unambiguous across platforms
        if k == ord("."):
            cur = min(cur + 1, n_frames - 1); playing = False
        elif k == ord(","):
            cur = max(cur - 1, 0); playing = False
        elif k == ord(">"):
            cur = min(cur + 30, n_frames - 1); playing = False
        elif k == ord("<"):
            cur = max(cur - 30, 0); playing = False
        elif k == ord("j"):
            try:
                target = int(input("jump to frame: ").strip())
                cur = max(0, min(target, n_frames - 1))
            except Exception:
                pass
            playing = False

        # bout construction
        elif k == ord("b"):
            in_progress = {"start": cur, "label": None}
            print(f"[bout] start={cur}, awaiting r/g/l/i then e")
        elif k in LABEL_KEYS:
            label = LABEL_KEYS[k]
            if in_progress is None:
                in_progress = {"start": cur, "label": label}
                print(f"[bout] {label} start={cur}  (press 'e' to end)")
            else:
                in_progress["label"] = label
                print(f"[bout] label set to {label}")
        elif k == ord("e"):
            if in_progress is None or in_progress["label"] is None:
                print("[warn] no labelled bout in progress")
            else:
                bouts = append_bout(bouts, subject, in_progress["start"], cur,
                                    in_progress["label"], "manual")
                last = bouts.iloc[-1]
                print(f"[bout] saved: {last['label']} {last['start_frame']}-{last['end_frame']} (n={last['end_frame']-last['start_frame']+1})")
                in_progress = None

        # undo / save
        elif k == ord("u"):
            if len(bouts):
                last = bouts.iloc[-1]
                bouts = bouts.iloc[:-1].reset_index(drop=True)
                print(f"[undo] removed {last['label']} {last['start_frame']}-{last['end_frame']}")
        elif k == ord("s"):
            save_gt(subject, bouts)

        # weak-label acceptance
        elif k == ord("a"):
            if not len(weak):
                print("[warn] no weak labels available")
            else:
                hit = weak[(weak["start_frame"] <= cur) & (weak["end_frame"] >= cur)]
                if not len(hit):
                    print(f"[warn] no weak label at frame {cur}")
                else:
                    row = hit.iloc[0]
                    bouts = append_bout(bouts, subject, row["start_frame"],
                                        row["end_frame"], row["label"], "weak_accepted")
                    print(f"[bout] accepted weak: {row['label']} {row['start_frame']}-{row['end_frame']}")
        elif k == ord("A"):
            if not len(weak):
                print("[warn] no weak labels available")
            else:
                added = 0
                for _, row in weak.iterrows():
                    bouts = append_bout(bouts, subject, row["start_frame"],
                                        row["end_frame"], row["label"], "weak_accepted")
                    added += 1
                print(f"[bout] accepted ALL {added} weak labels — review and edit as needed")

        if playing:
            cur = min(cur + 1, n_frames - 1)
            if cur == n_frames - 1:
                playing = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
