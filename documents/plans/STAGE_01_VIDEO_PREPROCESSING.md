# Stage 01 — Video Collection & Preprocessing
## Plan & Roadmap

---

## Objective

Organize raw experiment videos into a clean, structured dataset with consistent quality and metadata, ready for DeepLabCut analysis.

---

## Status

- [ ] In progress

---

## Dataset Summary

| Part | Arena | Rats | Sessions | Files | Format |
|------|-------|------|----------|-------|--------|
| Part0 | Rectangle (Open Field) | MA1, MA3, MA5, MA7 | 3 each | 12 | .avi |
| Part1 | T-Maze | MA1, MA3 | 3 each | 6 | .avi |
| Part2 | T-Maze | MA5, MA7 | 3 each | 6 | .avi |

**Naming convention:** `MA{rat_id}-{session}_res.avi`

---

## Tasks

### 1.1 — Video Inventory
- [ ] List all 24 videos and confirm file sizes are non-zero
- [ ] Verify all videos open correctly (no corruption)
- [ ] Record duration, resolution, and frame rate for each video

```python
import cv2, os, glob

video_dir = "D:/ProjectsD/ThesisWork/data/raw_videos"
videos = glob.glob(os.path.join(video_dir, "**", "*.avi"), recursive=True)

for v in videos:
    cap = cv2.VideoCapture(v)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w      = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h      = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    dur    = frames / fps if fps > 0 else 0
    print(f"{os.path.basename(v):30s} | {int(w)}x{int(h)} | {fps:.1f}fps | {dur:.1f}s | {frames:.0f} frames")
    cap.release()
```

### 1.2 — Metadata CSV
- [ ] Create `data/metadata.csv` with one row per video

| Column | Description |
|--------|-------------|
| `rat_id` | MA1, MA3, MA5, MA7 |
| `session` | 1, 2, 3 |
| `arena` | rectangle / tmaze |
| `part` | Part0, Part1, Part2 |
| `condition` | e.g., control / stressed (to be defined) |
| `filename` | `MA1-1_res.avi` |
| `duration_s` | Total video duration in seconds |
| `fps` | Frame rate |
| `resolution` | e.g., 1280x720 |
| `file_path` | Full path |

### 1.3 — Video Quality Check
- [ ] Confirm consistent resolution across all sessions
- [ ] Check for lighting inconsistencies (histogram per video)
- [ ] Flag any videos with motion blur or dropped frames
- [ ] Confirm top-down camera angle is consistent

### 1.4 — Trim Trial Windows (Optional)
- [ ] Identify trial start and end markers in each video
- [ ] If manual markers are available, trim videos to trial only
- [ ] Export trimmed copies to `data/raw_videos_trimmed/` (keep originals)

### 1.5 — Define Experimental Groups
- [ ] Assign `condition` labels (control, stressed, drug-treated, etc.)
- [ ] Document group assignment rationale
- [ ] Ensure group labels are stored in `metadata.csv`

---

## Deliverables

| File | Description |
|------|-------------|
| `data/metadata.csv` | Master inventory of all 24 videos |
| `data/raw_videos/` | Original .avi files untouched |

---

## Acceptance Criteria

- All 24 videos confirmed playable
- Metadata CSV populated with resolution, fps, duration, condition for every video
- No resolution or frame rate mismatches between sessions

---

## Next Step

→ **Stage 02:** `STAGE_02_DEEPLABCUT.md`
