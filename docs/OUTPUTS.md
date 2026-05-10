# Outputs Guide — What the Pipeline Produces

The `data/DLCfiltered/` folder holds the per-subject outputs of the open-field
behavioural-analysis pipeline. If you have just arrived at the project, start
here: this document explains **what the project does, what each file in a
subject folder means, how it was produced, and what to look for when
interpreting it.**

For the high-level project description and pipeline diagram, see
[`README.md`](README.md).

---

## 1. What is the project doing?

We study **rat open-field behaviour** from video. Each subject (e.g. `MA1_2`,
`MA5_3`) was recorded for ~3 minutes in a rectangular arena while exploring
freely. The raw videos were processed by **DeepLabCut (DLC)** to produce
per-frame coordinates of 9 body parts (nose, head, body-centre, left/right
forepaws, left/right hindpaws, tail-base, tail-tip).

Starting from those DLC-filtered coordinates, this pipeline produces two
complementary kinds of analysis per subject:

| Analysis | Question it answers | Main outputs |
|---|---|---|
| **Behaviour detection** | *When* does the rat rear or groom? | `*_behavior_timeline.png`, `*_behavior_bouts.csv`, `*_behavior_frames.csv` |
| **Spatial/locomotion analysis** | *Where* does the rat spend its time? (open-field / thigmotaxis metrics) | `*_orbit_grid.png`, `*_thigmotaxis.png`, `*_heatmap_kde.png`, `*_heatmap_histogram.png`, `*_bodypart_heatmaps.png` |

The end goal (thesis) is a per-subject behavioural profile that can be
compared across cohorts (MA1 / MA3 / MA5 / MA7).

---

## 2. Subject naming

Every CSV and its outputs follow the pattern:

```
OpenField<COHORT>_<RUN>        e.g. OpenFieldMA5_2
            │         │
            │         └─ session index (1, 2, 3)
            └─ cohort label (MA1, MA3, MA5, MA7)
```

Cohorts MA1/MA3/MA5/MA7 correspond to four experimental groups (3 subjects
each → 12 open-field recordings total).

---

## 3. Folder contents

Every subject folder (e.g. `OpenFieldMA5_2/`) contains the same 9 files:

```
OpenFieldMA5_2/
├── OpenFieldMA5_2.csv                       ← DLC-filtered input (source of truth)
│
│   Behaviour detection (src/behavior_detection.py)
├── OpenFieldMA5_2_behavior_timeline.png
├── OpenFieldMA5_2_behavior_bouts.csv
├── OpenFieldMA5_2_behavior_frames.csv
│
│   Spatial / locomotion (analysis/*.py)
├── OpenFieldMA5_2_orbit_grid.png
├── OpenFieldMA5_2_thigmotaxis.png
├── OpenFieldMA5_2_heatmap_kde.png
├── OpenFieldMA5_2_heatmap_histogram.png
└── OpenFieldMA5_2_bodypart_heatmaps.png
```

Each file is described below.

---

## 4. Input: the DLC CSV

**File**: `OpenField<ID>.csv`

Multi-header CSV produced by DeepLabCut. Each body part has three columns —
`x`, `y`, `likelihood` — at 30 fps. All downstream scripts load this file as
their only input.

Key conventions used throughout the pipeline:

- **Frame rate**: 30 fps (so frame `f` = time `f / 30` seconds).
- **Coordinate system**: image-space pixels. `y` increases **downward**
  (smaller `y` = higher in the frame, important for rearing logic).
- **Likelihood filter**: points with likelihood < 0.6 are set to NaN before
  use, except for `head`/`tail_base` when computing head-to-tail distance
  (the tail is often occluded during wall rearing — see
  `src/behavior_detection.py` docstring).

---

## 5. Behaviour-detection outputs

Produced by `src/behavior_detection.py` (rule-based; no ML training).
Classifies every frame into one of `rearing`, `grooming`, or `other`.

### 5.1 `*_behavior_timeline.png`

Three stacked rows sharing a time axis:

| Row | What it shows |
|---|---|
| **Ground Truth** | Manually validated rearing/grooming windows from video review. Only populated for subjects listed in `GROUND_TRUTH_BY_SUBJECT` in `src/behavior_detection.py`. A blank row means no GT has been entered for that subject yet. |
| **Detected Bouts** | Algorithm output after merging adjacent frames into bouts (gap ≤ 15 frames, min duration 10 frames). |
| **Per-Frame Label** | Raw per-frame classification before bout smoothing. Useful for spotting flicker or marginal detections. |

Colour code: **red = rearing**, **green = grooming**, grey/empty = other.

Use this plot to (a) spot-check detection against ground truth, and (b) see
how bout smoothing bridges short gaps.

### 5.2 `*_behavior_bouts.csv`

One row per detected bout. Columns:

| Column | Meaning |
|---|---|
| `behaviour` | `rearing` or `grooming` |
| `bout` | Sequential index within its behaviour |
| `start_frame`, `end_frame` | Inclusive frame range |
| `start_s`, `end_s` | Same, in seconds (frame / 30) |
| `duration_s` | Bout duration |

This is the main table used for statistical summaries (total time rearing,
bout counts, mean bout duration, etc.).

### 5.3 `*_behavior_frames.csv`

One row per frame: `frame, time_s, behaviour`. Useful when you need
per-frame labels for downstream alignment (e.g. correlating behaviour with
spatial position from the heatmap scripts).

### 5.4 How the rules work (summary)

Full details live in `src/behavior_detection.py`. The short version:

- **Grooming** is computed *first*, because grooming and compact rearing
  both compress the body in 2-D. Grooming requires nose close to forepaws,
  forepaws not elevated, low body velocity, and nose inside the arena.
- **Rearing** has six complementary rules (compact, top-wall extended,
  bottom-wall strong/compact, side-wall, wall-press). Any one firing labels
  the frame as rearing. Compact rearing explicitly excludes grooming
  posture.
- Final grooming label = grooming posture AND NOT rearing.

---

## 6. Spatial / locomotion outputs

Produced by the scripts in `analysis/` (entry point: `analysis/run_analysis.py`).
These describe *where* the rat went, not *what* it was doing.

All of them apply the same 4-step cleaning pipeline to the DLC CSV:
likelihood filter → arena-bounds filter → jump threshold → rolling-median
smoothing. See `docs/WORKFLOW_SUMMARY.md` for parameters.

### 6.1 `*_orbit_grid.png`

Grid of trajectory plots, one panel per body part. Shows the raw path each
body part took over the whole session, overlaid on the arena rectangle.

**Use it to**: eyeball tracking quality (a clean trajectory = reliable DLC
output; scattered dots = many rejected frames) and get a qualitative sense
of exploration pattern.

### 6.2 `*_thigmotaxis.png`

Single-panel trajectory of `body_center` with the **inner zone** drawn
inside the arena (default: 20 % margin from each wall).

**Thigmotaxis** = wall-hugging behaviour, a classical anxiety-like index in
open-field assays. A rat that stays in the outer ring is thigmotactic;
one that crosses into the centre is more exploratory. The plot title
usually reports the % time spent in the inner zone.

### 6.3 `*_heatmap_kde.png`

Kernel-density estimate of `body_center` position over the whole session.
Smooth, publication-ready spatial-occupancy map. **This is the primary
thesis figure for spatial preference.**

Hot spots = frequently occupied locations. Strong hot ring along the walls
= thigmotaxis; central hot spot = centre-seeking.

### 6.4 `*_heatmap_histogram.png`

Same underlying data as the KDE, but as a 2-D histogram (binned counts
instead of smoothed density). Useful as a sanity check — the histogram
shows raw sample distribution, the KDE shows the smoothed version.

### 6.5 `*_bodypart_heatmaps.png`

Grid of density heatmaps, one per body part. Highlights which parts of the
arena each body part visited. Can reveal, for example, that the nose maps
further along the walls than the body-centre (nose-to-wall sniffing during
thigmotaxis).

---

## 7. How to regenerate these files

From the repo root:

```bash
# Behaviour detection for one subject
python src/behavior_detection.py --csv data/DLCfiltered/OpenFieldMA5_2/OpenFieldMA5_2.csv

# Spatial analysis for one subject (arena coords come from show_frame_coords.py)
python analysis/run_analysis.py --arena 396 776 153 530
```

See `analysis/README.md` for the full spatial-analysis workflow and
`docs/behavior_detection_documentation.legacy.md` for the behaviour-detection
design rationale (Turkish).

---

## 8. Where to look when something looks wrong

| Symptom | Likely cause | Where to check |
|---|---|---|
| Ground-truth row in timeline is empty | No GT entry for this subject | `GROUND_TRUTH_BY_SUBJECT` dict in `src/behavior_detection.py` |
| Trajectory has big gaps | Many frames rejected by likelihood filter | Run with `--likelihood 0.5` or inspect the bodypart heatmap grid for quality |
| Rearing detected during fast locomotion | Likely a compact-rearing false positive while passing near a wall | Thresholds are in `src/behavior_detection.py` (`REAR_COMPACT_HTDIST` etc.) |
| Grooming detected during wall-rearing | Velocity gate not triggering | `GROOM_MAX_VEL` in `src/behavior_detection.py` |

---

## 9. Cohort status at a glance

| Subject | Behaviour outputs | Spatial outputs | Ground truth in code |
|---|---|---|---|
| MA1_1, MA1_2, MA1_3 | yes | yes | only MA1_2 |
| MA3_1, MA3_2, MA3_3 | yes | yes | no |
| MA5_1, MA5_2, MA5_3 | yes | yes | only MA5_1 (grooming) |
| MA7_1, MA7_2, MA7_3 | CSV only — not yet analysed | CSV only — not yet analysed | no |

Adding ground truth for a new subject: append an entry to
`GROUND_TRUTH_BY_SUBJECT` in `src/behavior_detection.py` with
`(start_frame, end_frame)` tuples at 30 fps.
