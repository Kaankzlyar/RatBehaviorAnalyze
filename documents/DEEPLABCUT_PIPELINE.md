# DeepLabCut Training Pipeline
## Rat T-Maze Pose Estimation Setup

---

## Overview

This document covers the full DeepLabCut (DLC) workflow: from serving your rat T-maze videos to training a pose estimation model and exporting tracking data for downstream analysis.

---

## Prerequisites

### Install DeepLabCut

```bash
# Create a dedicated conda environment (recommended)
conda create -n dlc python=3.10 -y
conda activate dlc

# Install DeepLabCut with GPU support
pip install deeplabcut[tf]

# Or for PyTorch backend (DLC 3.x+)
pip install deeplabcut[torch]

# Verify installation
python -c "import deeplabcut; print(deeplabcut.__version__)"
```

### GPU Setup (recommended for training)
```bash
# Check CUDA availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# or for PyTorch backend:
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Stage 1 — Project Creation

```python
import deeplabcut

# Define your project
project_name = "rat_tmaze"
experimenter  = "your_name"

# List of video paths to include in the project
video_paths = [
    "D:/ProjectsD/ThesisWork/data/raw_videos/rat01_session01.mp4",
    "D:/ProjectsD/ThesisWork/data/raw_videos/rat02_session01.mp4",
    # Add more videos...
]

# Create project — generates config.yaml in the project folder
config_path = deeplabcut.create_new_project(
    project_name,
    experimenter,
    video_paths,
    working_directory="D:/ProjectsD/ThesisWork/models/dlc_model",
    copy_videos=False,   # Set True to copy videos into project folder
    videotype="mp4"
)

print(f"Config path: {config_path}")
```

This creates the directory structure:
```
dlc_model/
└── rat_tmaze-your_name-YYYY-MM-DD/
    ├── config.yaml          # Main config — edit this
    ├── videos/              # Symlinks or copies of your videos
    ├── labeled-data/        # Where your labeled frames will go
    ├── training-datasets/   # Generated training data
    └── dlc-models/          # Trained model weights
```

---

## Stage 2 — Configure config.yaml

Open `config.yaml` and edit the following key fields:

```yaml
# Body parts to track on the rat
bodyparts:
  - nose
  - head
  - neck
  - body_center
  - tail_base

# Skeleton connections (for visualization)
skeleton:
  - [nose, head]
  - [head, neck]
  - [neck, body_center]
  - [body_center, tail_base]

# Number of frames to extract per video for labeling
numframes2pick: 20   # 20 per video — adjust based on video count

# Cropping (optional — useful if maze occupies only part of frame)
cropleftwidth: 0
cropright: 1920
croptop: 0
cropbottom: 1080
```

---

## Stage 3 — Frame Extraction

Extract representative frames from your videos for manual labeling.

```python
# Extract frames using kmeans clustering (diverse frames)
deeplabcut.extract_frames(
    config_path,
    mode="automatic",       # "automatic" or "manual"
    algo="kmeans",          # "kmeans" or "uniform"
    userfeedback=False,
    crop=False
)
```

### Extraction Strategy for T-maze
- Use `kmeans` to get diverse postures
- Target: **200-400 total frames** across all videos
- Include frames from: straight running, turning, stopping, rearing

---

## Stage 4 — Labeling Frames

Launch the labeling GUI to annotate body parts on extracted frames.

```python
deeplabcut.label_frames(config_path)
```

### Labeling GUI Tips
| Action | Shortcut |
|---|---|
| Next frame | `D` |
| Previous frame | `A` |
| Save labels | `Ctrl+S` |
| Zoom in | Scroll wheel |
| Place label | Left click |
| Remove label | Right click |

### Labeling Guidelines for Rats
- **nose**: tip of the snout
- **head**: between the ears, top of skull
- **neck**: base of skull / top of shoulders
- **body_center**: midpoint of the torso
- **tail_base**: where tail meets the body

> If a body part is not visible (occluded), skip it — DLC handles missing labels.

### Verify Labels
```python
deeplabcut.check_labels(config_path, visualizeindividuals=True)
# Outputs labeled images to labeled-data/ for visual inspection
```

---

## Stage 5 — Create Training Dataset

```python
deeplabcut.create_training_dataset(
    config_path,
    num_shuffles=1,          # Number of train/test splits
    augmenter_type="imgaug"  # Data augmentation backend
)
```

This generates the TFRecords / numpy arrays used during training.

---

## Stage 6 — Train the Model

```python
deeplabcut.train_network(
    config_path,
    shuffle=1,
    trainingsetindex=0,
    gputouse=0,              # GPU index (0 for first GPU, None for CPU)
    max_snapshots_to_keep=5,
    autotune=False,
    displayiters=100,        # Print loss every N iterations
    saveiters=1000,          # Save checkpoint every N iterations
    maxiters=50000           # Total training iterations
)
```

### Training Recommendations
| Dataset Size | Recommended maxiters |
|---|---|
| < 200 frames | 30,000 |
| 200-400 frames | 50,000 |
| 400+ frames | 100,000 |

Monitor the loss in the terminal — it should decrease and plateau. Stop training when loss stabilizes.

---

## Stage 7 — Evaluate the Model

```python
deeplabcut.evaluate_network(
    config_path,
    shuffle=[1],
    plotting=True            # Saves evaluation plots
)
```

### Target Metrics
| Metric | Acceptable | Good |
|---|---|---|
| Train RMSE | < 5 px | < 3 px |
| Test RMSE | < 8 px | < 5 px |
| p-cutoff ratio | > 0.85 | > 0.95 |

If test error is high:
- Add more labeled frames (especially failure cases)
- Run more training iterations
- Check label consistency

---

## Stage 8 — Run Inference on All Videos

```python
# Analyze a single video
deeplabcut.analyze_videos(
    config_path,
    ["D:/ProjectsD/ThesisWork/data/raw_videos/rat01_session01.mp4"],
    videotype="mp4",
    shuffle=1,
    save_as_csv=True,        # Export as CSV in addition to H5
    destfolder="D:/ProjectsD/ThesisWork/data/dlc_output/"
)

# Analyze all videos in a folder
import os
video_folder = "D:/ProjectsD/ThesisWork/data/raw_videos/"
videos = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]

deeplabcut.analyze_videos(
    config_path,
    videos,
    videotype="mp4",
    shuffle=1,
    save_as_csv=True,
    destfolder="D:/ProjectsD/ThesisWork/data/dlc_output/"
)
```

### Output Files
For each video, DLC generates:
```
data/dlc_output/
├── rat01_session01DLC_resnet50_rat_tmazeJan01shuffle1_50000.h5   # Full data
├── rat01_session01DLC_resnet50_rat_tmazeJan01shuffle1_50000.csv  # Human-readable
└── rat01_session01DLC_resnet50_rat_tmazeJan01shuffle1_50000_meta.pickle
```

### CSV Column Structure
```
scorer         | DLC_resnet50_...
bodyparts      | nose  nose  nose  head  head  head  ...
coords         | x     y     likelihood  x  y  likelihood ...
frame_index
0              | 312.4  210.1  0.998  ...
1              | 313.1  210.5  0.997  ...
```

---

## Stage 9 — Filter & Create Labeled Videos

```python
# Filter predictions (smooth tracking, remove low-confidence frames)
deeplabcut.filterpredictions(
    config_path,
    ["D:/ProjectsD/ThesisWork/data/raw_videos/rat01_session01.mp4"],
    videotype="mp4",
    shuffle=1,
    filtertype="median",
    windowlength=5
)

# Create labeled video for visual QC
deeplabcut.create_labeled_video(
    config_path,
    ["D:/ProjectsD/ThesisWork/data/raw_videos/rat01_session01.mp4"],
    videotype="mp4",
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
        x = df[bp]["x"].values.astype(float)
        y = df[bp]["y"].values.astype(float)
        likelihood = df[bp]["likelihood"].values.astype(float)

        # Mask low-confidence frames
        x[likelihood < likelihood_threshold] = np.nan
        y[likelihood < likelihood_threshold] = np.nan

        cleaned[bp] = {"x": x, "y": y, "likelihood": likelihood}

    return cleaned

# Example usage
tracking = load_dlc_csv(
    "D:/ProjectsD/ThesisWork/data/dlc_output/rat01_session01...csv"
)
nose_x = tracking["nose"]["x"]
nose_y = tracking["nose"]["y"]
```

---

## Iterative Improvement (Active Learning)

If tracking quality is poor on certain videos:

```python
# Extract frames where DLC was uncertain
deeplabcut.extract_outlier_frames(
    config_path,
    ["path/to/problem_video.mp4"],
    outlieralgorithm="uncertain",  # Flags low-likelihood frames
    epsilon=20,                    # Pixel threshold for outlier detection
    automatic=True
)

# Re-label the outlier frames
deeplabcut.refine_labels(config_path)

# Merge new labels with existing dataset
deeplabcut.merge_datasets(config_path)

# Retrain from existing checkpoint
deeplabcut.train_network(config_path, shuffle=1, maxiters=75000)
```

---

## Video Serving Notes

When serving videos for DLC analysis:

- **Format**: MP4 (H.264) recommended
- **Resolution**: Consistent across all sessions (e.g., 1920x1080 or 1280x720)
- **Frame rate**: 25-30 fps standard; 60+ fps for fast movements
- **Camera angle**: Top-down (bird's eye) for T-maze — minimizes occlusion
- **Lighting**: Uniform, no flickering; rat should contrast with maze floor
- **Compression**: Avoid heavy compression — artefacts degrade DLC accuracy

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| High test RMSE | Too few labels or inconsistent labeling | Add 50-100 more frames, re-check labels |
| Body parts jumping | Low video quality or fast motion | Add more frames from fast-motion segments |
| GPU out of memory | Batch size too large | Reduce `batch_size` in pose_cfg.yaml |
| Tracking drifts | Model undertrained | Increase `maxiters` |
| Low likelihood everywhere | Wrong video resolution in config | Check `x1,x2,y1,y2` crop settings |

---

## Next Step

Once DLC tracking CSVs are ready, proceed to:
**`RAT_TMAZE_PROJECT.md` → Stage 3: Heatmap Generation**
