# DeepLabCut Training Pipeline
## Dual-Arena Rat Behavioral Analysis

---

## Overview

This document covers the full DeepLabCut (DLC) workflow for the rat behavioral project. Videos span two arena types (**rectangle open field** and **T-maze**) stored across three subfolders. A **single DLC model** is trained on frames from all arenas — the rat body parts are the same regardless of the arena, and diverse training data improves generalization.

| Part | Arena | Rats | Videos |
|------|-------|------|--------|
| Part0 | Rectangle (Open Field) | MA1, MA3, MA5, MA7 | 12 × .avi |
| Part1 | T-Maze | MA1, MA3 | 6 × .avi |
| Part2 | T-Maze | MA5, MA7 | 6 × .avi |

---

## Prerequisites

### Install DeepLabCut

```bash
conda create -n dlc python=3.10 -y
conda activate dlc

# PyTorch backend (DLC 3.x+, recommended for RTX 3060)
pip install deeplabcut[torch]

python -c "import deeplabcut; print(deeplabcut.__version__)"
```

### GPU Setup
```bash
# PyTorch backend
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Stage 1 — Project Creation

```python
import deeplabcut, glob, os

project_name = "rat_behavior"
experimenter = "kaank"

# Collect all .avi videos recursively from Part0/, Part1/, Part2/
video_dir  = "D:/ProjectsD/ThesisWork/data/raw_videos"
video_paths = glob.glob(os.path.join(video_dir, "**", "*.avi"), recursive=True)
print(f"Found {len(video_paths)} videos")  # expected: 24

config_path = deeplabcut.create_new_project(
    project_name,
    experimenter,
    video_paths,
    working_directory="D:/ProjectsD/ThesisWork/models/dlc_model",
    copy_videos=False,
    videotype="avi"
)

print(f"Config path: {config_path}")
```

This creates:
```
models/dlc_model/
└── rat_behavior-kaank-YYYY-MM-DD/
    ├── config.yaml
    ├── videos/
    ├── labeled-data/
    ├── training-datasets/
    └── dlc-models/
```

---

## Stage 2 — Configure config.yaml

```yaml
bodyparts:
  - nose
  - head
  - neck
  - body_center
  - tail_base

skeleton:
  - [nose, head]
  - [head, neck]
  - [neck, body_center]
  - [body_center, tail_base]

numframes2pick: 20   # 20 per video × 24 videos = up to 480 frames

dotsize: 6
alphavalue: 0.9
```

---

## Stage 3 — Frame Extraction

```python
deeplabcut.extract_frames(
    config_path,
    mode="automatic",
    algo="kmeans",        # diverse frame selection
    userfeedback=False,
    crop=False
)
```

### Extraction Strategy
- `kmeans` captures diverse postures across both arena types
- Target: **200–400 total labeled frames**
- Include frames from: straight running, turning, stopping, rearing
- Make sure frames from **both rectangle and T-maze sessions** are represented

---

## Stage 4 — Labeling Frames

```python
deeplabcut.label_frames(config_path)
```

### Labeling GUI Shortcuts
| Action | Shortcut |
|--------|----------|
| Next frame | `D` |
| Previous frame | `A` |
| Save labels | `Ctrl+S` |
| Place label | Left click |
| Remove label | Right click |

### Body Part Guidelines
- **nose**: tip of the snout
- **head**: between the ears, top of skull
- **neck**: base of skull / top of shoulders
- **body_center**: midpoint of the torso
- **tail_base**: where tail meets the body

> If a body part is not visible (occluded), skip it — DLC handles missing labels.

### Verify Labels
```python
deeplabcut.check_labels(config_path, visualizeindividuals=True)
```

---

## Stage 5 — Create Training Dataset

```python
deeplabcut.create_training_dataset(
    config_path,
    num_shuffles=1,
    augmenter_type="imgaug"
)
```

---

## Stage 6 — Train the Model

```python
deeplabcut.train_network(
    config_path,
    shuffle=1,
    trainingsetindex=0,
    gputouse=0,              # RTX 3060
    max_snapshots_to_keep=5,
    autotune=False,
    displayiters=500,
    saveiters=5000,
    maxiters=50000
)
```

### Training Recommendations
| Dataset Size | Recommended maxiters |
|---|---|
| < 200 frames | 30,000 |
| 200–400 frames | 50,000 |
| 400+ frames | 100,000 |

**RTX 3060 (6GB VRAM):** set `batch_size=8` in `pose_cfg.yaml` via `dlc_train.py`.

---

## Stage 7 — Evaluate the Model

```python
deeplabcut.evaluate_network(
    config_path,
    shuffle=[1],
    plotting=True
)
```

### Target Metrics
| Metric | Acceptable | Good |
|--------|-----------|------|
| Train RMSE | < 5 px | < 3 px |
| Test RMSE | < 8 px | < 5 px |
| p-cutoff ratio | > 0.85 | > 0.95 |

If test error is high on one arena type, extract outlier frames from those videos and refine.

---

## Stage 8 — Run Inference on All Videos

Outputs are **separated by arena** into `rectangle/` and `tmaze/` subfolders.

```python
import glob, os
import deeplabcut

config_path = "D:/ProjectsD/ThesisWork/models/dlc_model/rat_behavior-kaank-YYYY-MM-DD/config.yaml"

ARENA_MAP = {"Part0": "rectangle", "Part1": "tmaze", "Part2": "tmaze"}

for part, arena in ARENA_MAP.items():
    videos = glob.glob(
        f"D:/ProjectsD/ThesisWork/data/raw_videos/{part}/*.avi"
    )
    out_dir = f"D:/ProjectsD/ThesisWork/data/dlc_output/{arena}"
    os.makedirs(out_dir, exist_ok=True)

    deeplabcut.analyze_videos(
        config_path,
        videos,
        videotype="avi",
        shuffle=1,
        save_as_csv=True,
        destfolder=out_dir
    )
```

### Output Files
```
data/dlc_output/
├── rectangle/
│   └── MA1-1_res_DLC_resnet50_rat_behaviorMMDDshuffle1_50000.csv
├── tmaze/
│   └── MA1-1_res_DLC_resnet50_rat_behaviorMMDDshuffle1_50000.csv
└── labeled_videos/
```

### CSV Column Structure
```
scorer         | DLC_resnet50_...
bodyparts      | nose  nose  nose  head  head  head  ...
coords         | x     y     likelihood  x  y  likelihood ...
frame_index
0              | 312.4  210.1  0.998  ...
```

---

## Stage 9 — Filter & Create Labeled Videos

```python
# Filter predictions (smooth tracking)
deeplabcut.filterpredictions(
    config_path,
    videos,
    videotype="avi",
    shuffle=1,
    filtertype="median",
    windowlength=5,
    destfolder=out_dir
)

# Create labeled video for visual QC
deeplabcut.create_labeled_video(
    config_path,
    videos,
    videotype="avi",
    shuffle=1,
    filtered=True,
    draw_skeleton=True,
    destfolder="D:/ProjectsD/ThesisWork/data/dlc_output/labeled_videos/"
)
```

---

## Stage 10 — Load Tracking Data for Analysis

```python
import pandas as pd
import numpy as np

def load_dlc_csv(csv_path, likelihood_threshold=0.6):
    """Load DLC CSV and filter low-confidence predictions."""
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(0).unique()
    cleaned = {}

    for bp in bodyparts:
        x          = df[bp]["x"].values.astype(float)
        y          = df[bp]["y"].values.astype(float)
        likelihood = df[bp]["likelihood"].values.astype(float)

        x[likelihood < likelihood_threshold] = np.nan
        y[likelihood < likelihood_threshold] = np.nan

        cleaned[bp] = {"x": x, "y": y, "likelihood": likelihood}

    return cleaned

# Rectangle arena session
oft_tracking = load_dlc_csv("D:/ProjectsD/ThesisWork/data/dlc_output/clean/rectangle/MA1-1_res_..._clean.csv")

# T-maze session
tmaze_tracking = load_dlc_csv("D:/ProjectsD/ThesisWork/data/dlc_output/clean/tmaze/MA1-1_res_..._clean.csv")
```

---

## Iterative Improvement (Active Learning)

If tracking quality is poor on certain videos:

```python
# Extract frames where DLC was uncertain
deeplabcut.extract_outlier_frames(
    config_path,
    ["path/to/problem_video.avi"],
    outlieralgorithm="uncertain",
    epsilon=20,
    automatic=True
)

deeplabcut.refine_labels(config_path)
deeplabcut.merge_datasets(config_path)
deeplabcut.train_network(config_path, shuffle=1, maxiters=75000)
```

---

## Video Format Notes

- **Format**: `.avi` (H.264 or MJPEG)
- **Camera angle**: Top-down (bird's eye) for both arenas
- **Lighting**: Uniform, no flickering; rat should contrast with floor
- **Resolution**: Consistent across all sessions

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| High test RMSE on one arena | Underrepresented in training frames | Add more frames from that arena, re-label |
| Body parts jumping | Low video quality or fast motion | Add frames from fast-motion segments |
| GPU out of memory | batch_size too large | Reduce `batch_size` in pose_cfg.yaml (try 4) |
| Tracking drifts | Model undertrained | Increase `maxiters` |
| Low likelihood everywhere | Wrong video type or resolution | Check `videotype="avi"` and crop settings |
| No videos found | Script searching for .mp4 | Confirm `VIDEO_EXT = "avi"` in all scripts |

---

## Next Step

Once DLC tracking CSVs are ready, proceed to:
**`RAT_TMAZE_PROJECT.md` → Stage 3A (OFT Analysis) and Stage 3B (T-Maze Heatmaps)**
