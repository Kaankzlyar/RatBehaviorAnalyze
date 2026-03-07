# Stage 05 — Feature Engineering
## Plan & Roadmap

---

## Objective

Merge OFT (Stage 03A) and T-maze (Stage 04) metrics into a unified feature dataset. Engineer derived features, aggregate by session and rat, and produce the final table ready for ML model training.

---

## Status

- [ ] Not started — requires Stage 03A and Stage 04 outputs

---

## Input Sources

| File | Source Stage | Description |
|------|-------------|-------------|
| `data/features/oft_metrics.csv` | Stage 03A | Per-session OFT metrics |
| `data/features/tmaze_metrics.csv` | Stage 04 | Per-trial T-maze metrics |
| `data/metadata.csv` | Stage 01 | Rat IDs, conditions, sessions |

---

## Tasks

### 5.1 — Session-Level Aggregation of T-Maze Trials

T-maze metrics are per-trial; aggregate to per-session for merging with OFT.

- [ ] Mean and SD of each per-trial metric across all trials in a session
- [ ] Correct-arm choice rate (if correct arm is defined per session)
- [ ] Learning slope (turn_bias change across trials in session)

```python
import pandas as pd

tmaze = pd.read_csv("data/features/tmaze_metrics.csv")

agg = tmaze.groupby(["rat_id", "session"]).agg(
    turn_bias=("turn_choice", lambda x: (x == "left").mean()),
    path_efficiency_mean=("path_efficiency", "mean"),
    path_efficiency_std=("path_efficiency", "std"),
    decision_latency_mean=("decision_latency_s", "mean"),
    velocity_stem_mean=("velocity_stem", "mean"),
    backtrack_rate=("backtrack_rate", "first"),
    heading_var_mean=("heading_angle_var", "mean"),
).reset_index()
```

### 5.2 — Merge OFT + T-Maze Features

- [ ] Left-join on `rat_id` + `session`
- [ ] Verify no sessions are lost (all 4 rats × 3 sessions = 12 rows)
- [ ] Add `condition` column from metadata

```python
oft   = pd.read_csv("data/features/oft_metrics.csv")
meta  = pd.read_csv("data/metadata.csv")[["rat_id", "session", "condition"]].drop_duplicates()

features = (
    oft
    .merge(agg, on=["rat_id", "session"], how="outer")
    .merge(meta, on=["rat_id", "session"], how="left")
)
```

### 5.3 — Cross-Arena Derived Features

Engineer features that combine OFT and T-maze signals.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `anxiety_composite` | `peripheral_time_ratio * decision_latency_mean` | Combined anxiety index |
| `locomotion_delta` | `oft_total_distance - tmaze_path_length_mean` | Activity shift across arenas |
| `exploration_vs_efficiency` | `oft_exploration_rate / path_efficiency_mean` | Exploration strategy |
| `velocity_consistency` | `oft_mean_velocity / tmaze_velocity_stem_mean` | Cross-arena motor consistency |

- [ ] Implement all cross-arena features in `src/features.py`

### 5.4 — Rolling / Within-Session Features

- [ ] Rolling 3-session average for each metric (learning curves)
- [ ] Session-over-session delta (session 2 − session 1, session 3 − session 2)
- [ ] Within-session variability (SD across trials for T-maze metrics)

### 5.5 — Missing Value Handling

- [ ] Report % missing per column
- [ ] If T-maze session has no valid trials: set aggregated metrics to NaN
- [ ] Impute remaining NaN with column median (document any imputations)
- [ ] Drop columns with > 30% missing values

### 5.6 — Encoding & Normalization

- [ ] Encode `rat_id` as integer factor
- [ ] One-hot encode `condition` (control, stressed, etc.)
- [ ] StandardScaler on all continuous features for ML model input
- [ ] Save scaler to `models/classifier/scaler.pkl` for inference

### 5.7 — Label Assignment

Define classification targets in consultation with domain expert:

| Label | Type | Definition |
|-------|------|-----------|
| `anxiety_level` | binary | High vs Low (OFT thigmotaxis + T-maze latency) |
| `spatial_memory` | ordinal | Correct arm choice rate in T-maze |
| `cognitive_flexibility` | binary | Fast vs slow reversal learning |
| `stress_response` | binary | Behavioral deviation from baseline |

- [ ] Add label columns to feature dataset

### 5.8 — Export

- [ ] Save raw merged features to `data/features/features_raw.csv`
- [ ] Save normalized features to `data/features/features_normalized.csv`
- [ ] Save label columns to `data/features/labels.csv`

---

## Final Feature Dataset Structure

```
rat_id | session | condition |
# OFT
oft_center_time | oft_peripheral_time | oft_total_distance |
oft_mean_velocity | oft_exploration_rate | oft_immobility_bouts |
# T-Maze (aggregated)
tmaze_turn_bias | tmaze_path_efficiency_mean | tmaze_decision_latency_mean |
tmaze_velocity_stem | tmaze_backtrack_rate | tmaze_heading_var |
# Cross-arena
anxiety_composite | locomotion_delta | exploration_vs_efficiency |
# Session deltas
oft_distance_delta | tmaze_latency_delta |
# Labels
anxiety_level | spatial_memory | cognitive_flexibility
```

---

## Acceptance Criteria

- Unified feature dataset with 12 rows (4 rats × 3 sessions), no dropped sessions
- All features documented with units and derivation
- Correlation matrix generated showing OFT ↔ T-maze relationships
- Label columns assigned (at minimum `anxiety_level`)

---

## Output Files

| Path | Description |
|------|-------------|
| `data/features/features_raw.csv` | Merged, un-normalized features |
| `data/features/features_normalized.csv` | StandardScaled features |
| `data/features/labels.csv` | Target labels |
| `models/classifier/scaler.pkl` | Saved StandardScaler |
| `src/features.py` | Feature engineering script |

---

## Next Step

→ **Stage 06:** `STAGE_06_MODEL_TRAINING.md`
