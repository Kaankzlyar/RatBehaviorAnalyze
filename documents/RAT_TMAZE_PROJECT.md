# Rat T-Maze Behavioral Analysis System
## Project Documentation & Roadmap

---

## Project Overview

This project builds an end-to-end pipeline for rat T-maze behavioral analysis using computer vision, pose estimation, and machine learning. The goal is to extract meaningful psychological and behavioral indicators from raw video footage through automated analysis.

---

## Pipeline Architecture

```
Raw Video
    |
    v
DeepLabCut (Pose Estimation & Tracking)
    |
    v
Heatmap Generation + Path Line Extraction
    |
    v
Feature Engineering (Behavioral Metrics)
    |
    v
ML Model Training (Psychological State Classification)
    |
    v
Psychological Analysis & Reporting
```

---

## Stage 1 — Video Collection & Preprocessing

### Input
- T-maze experiment videos (top-down camera view recommended)
- Multiple sessions per rat, multiple rats per experimental group

### Tasks
- [ ] Standardize video resolution, frame rate, and lighting conditions
- [ ] Define experimental groups (e.g., control, stressed, drug-treated)
- [ ] Label videos with metadata: rat ID, session number, date, condition
- [ ] Trim videos to trial start/end markers

### Deliverables
- Organized video dataset with metadata CSV
- Preprocessing script for batch normalization

---

## Stage 2 — DeepLabCut Analysis

### Purpose
Track body part positions of each rat across all video frames.

### Setup
- Train DeepLabCut model on labeled rat T-maze frames
- Body parts to label: nose, head, neck, body-center, tail-base, left/right limbs

### Tasks
- [ ] Label 200-400 frames per condition for training
- [ ] Train DLC model, evaluate test error (target: < 5px RMSE)
- [ ] Run inference on all experiment videos
- [ ] Export tracking data as CSV/H5 files (x, y coordinates per body part per frame)

### Output Format
```
frame | nose_x | nose_y | head_x | head_y | body_x | body_y | tail_x | tail_y | likelihood
```

---

## Stage 3 — Heatmap Generation

### Purpose
Visualize spatial occupancy and movement density across the T-maze.

### Heatmap Types
| Heatmap Type | Description | Psychological Relevance |
|---|---|---|
| Occupancy Heatmap | Time spent per maze zone | Preference / avoidance |
| Velocity Heatmap | Speed at each location | Anxiety / exploration drive |
| Entry Frequency Map | How often each zone is entered | Decision-making bias |
| Nose-Point Heatmap | Where rat investigates | Attention / curiosity |
| Path Density Map | Overlaid trajectory lines | Route learning / perseveration |

### Tasks
- [ ] Generate per-session heatmaps for each rat
- [ ] Generate group-average heatmaps per experimental condition
- [ ] Normalize heatmaps for cross-session comparison
- [ ] Export heatmaps as images and numerical arrays (numpy)

---

## Stage 4 — Path Line Analysis

### Purpose
Extract geometric and kinematic features from movement trajectories.

### Reference Lines
Define anatomical reference lines in the T-maze:
- **Stem line**: Entry point to decision point
- **Left arm line**: Decision point to left goal zone
- **Right arm line**: Decision point to right goal zone
- **Center reference**: Maze midline axis

### Path Metrics to Extract
| Metric | Definition | Behavioral Indicator |
|---|---|---|
| Turn bias (L/R ratio) | % of left vs right arm choices | Lateralization, perseveration |
| Path efficiency | Actual path length / optimal path length | Cognitive load, confusion |
| Decision latency | Time spent at choice point | Anxiety, deliberation |
| Velocity profile | Speed over time per zone | Exploration drive, fear |
| Heading angle | Body orientation relative to maze axis | Spatial attention |
| Backtrack rate | Frequency of direction reversals | Uncertainty, memory deficit |
| Zone dwell time | Time in stem / arms / goal zones | Preference, avoidance |
| Trajectory smoothness | Curvature variance of path | Motor control, anxiety |

### Tasks
- [ ] Implement zone detection from maze coordinate system
- [ ] Calculate all metrics per trial, per session, per rat
- [ ] Aggregate metrics into a structured feature dataset
- [ ] Visualize path lines colored by velocity or time

---

## Stage 5 — Feature Engineering

### Feature Dataset Structure
Each row = one trial. Columns = behavioral metrics + labels.

```
rat_id | session | condition | trial | turn_choice | latency | path_efficiency |
velocity_stem | velocity_arm | dwell_stem | dwell_arm | backtrack_rate |
heading_variance | occupancy_left | occupancy_right | ... | label
```

### Derived Features
- Rolling averages across trials (learning curves)
- Within-session variability metrics
- Inter-session difference scores (baseline vs treatment)
- Symmetry indices (left/right zone balance)

### Labels / Targets
Define psychological states as classification or regression targets:
| Label | Description |
|---|---|
| Anxiety level | High / Low based on open-arm avoidance, freezing |
| Cognitive flexibility | Reversal learning speed |
| Spatial memory | Correct arm choice rate |
| Stress response | Behavioral change from baseline |
| Exploratory drive | Novel zone investigation rate |

---

## Stage 6 — Model Training

### Approach
Train supervised ML / deep learning models to classify psychological states from behavioral features.

### Models to Evaluate
| Model | Use Case |
|---|---|
| Random Forest | Baseline classifier, feature importance |
| Gradient Boosting (XGBoost) | High-accuracy tabular classification |
| SVM | Small dataset, binary classification |
| LSTM / GRU | Sequential trial-level temporal patterns |
| Transformer (behavioral) | Long-session behavioral sequences |
| CNN on heatmaps | Direct image-based spatial classification |

### Training Strategy
- [ ] Split data: train/validation/test by rat (not by trial) to prevent leakage
- [ ] Cross-validate across experimental groups
- [ ] Handle class imbalance with SMOTE or weighted loss
- [ ] Hyperparameter tuning with Optuna or GridSearch

### Evaluation Metrics
- Accuracy, F1-score, AUC-ROC
- Confusion matrix per psychological state
- Feature importance plots (SHAP values)
- Behavioral pattern visualization per predicted class

---

## Stage 7 — Psychological Analysis & Reporting

### Analysis Outputs
- Per-rat behavioral profiles across sessions
- Group comparison statistics (ANOVA, t-test, Mann-Whitney U)
- Correlation matrices: behavioral features vs psychological labels
- Learning curve plots per condition
- Decision bias maps (L/R preference over time)

### Report Structure
```
1. Experiment Summary
2. Group Demographics & Conditions
3. Heatmap Gallery (per group, per session)
4. Path Analysis Results
5. Feature Distributions & Statistics
6. Model Performance Report
7. Psychological State Classifications
8. Conclusions & Behavioral Interpretation
```

---

## Technology Stack

| Component | Tool |
|---|---|
| Pose Estimation | DeepLabCut |
| Video Processing | OpenCV, FFmpeg |
| Data Processing | Python, Pandas, NumPy |
| Heatmaps & Visualization | Matplotlib, Seaborn, Plotly |
| Path Analysis | SciPy, Shapely |
| ML Models | Scikit-learn, XGBoost, PyTorch |
| Experiment Tracking | MLflow or Weights & Biases |
| Notebooks | Jupyter Lab |

---

## Project Roadmap

```
Phase 1 — Data Foundation         [Weeks 1-3]
  - Video collection & organization
  - DeepLabCut model training
  - Tracking data export

Phase 2 — Analysis Pipeline       [Weeks 4-6]
  - Heatmap generation module
  - Path line extraction module
  - Feature dataset construction

Phase 3 — Model Development       [Weeks 7-10]
  - Baseline model (Random Forest)
  - Deep learning models (LSTM, CNN)
  - Evaluation & comparison

Phase 4 — Psychological Mapping   [Weeks 11-13]
  - Label refinement with domain experts
  - Model calibration
  - Behavioral profile generation

Phase 5 — Reporting & Iteration   [Weeks 14-16]
  - Analysis reports
  - Visualization dashboard
  - Paper/thesis writing support
```

---

## Directory Structure

```
ThesisWork/
├── data/
│   ├── raw_videos/          # Original experiment videos
│   ├── dlc_output/          # DeepLabCut tracking CSVs/H5
│   ├── heatmaps/            # Generated heatmap images & arrays
│   └── features/            # Extracted behavioral feature datasets
├── models/
│   ├── dlc_model/           # Trained DeepLabCut model
│   └── classifier/          # Trained psychological state models
├── notebooks/
│   ├── 01_dlc_analysis.ipynb
│   ├── 02_heatmap_generation.ipynb
│   ├── 03_path_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_training.ipynb
├── src/
│   ├── heatmap.py
│   ├── path_analysis.py
│   ├── features.py
│   └── model.py
├── reports/
│   └── figures/
└── RAT_TMAZE_PROJECT.md     # This file
```

---

## Key Research Questions

1. Can spatial heatmap patterns reliably distinguish between anxiety levels in rats?
2. Do path efficiency metrics correlate with cognitive impairment?
3. Can a model trained on one experimental group generalize to another condition?
4. What behavioral features are most predictive of psychological state?
5. How do learning curves in path analysis reflect memory formation?

---

## Notes

- All analysis should be blinded to condition labels during feature extraction to avoid bias
- Validate behavioral metrics against established manual scoring (e.g., ethogram comparison)
- Consider ethically approved stress protocols and document all experimental conditions
- Model predictions are behavioral proxies — final psychological interpretation requires expert validation
