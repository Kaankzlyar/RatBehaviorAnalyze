# Stage 04 — Path Line Analysis (T-Maze)
## Plan & Roadmap

---

## Objective

Extract geometric, kinematic, and decision-level metrics from rat movement trajectories in the T-maze. These structured features quantify navigation strategy, cognitive load, and decision-making behavior.

---

## Status

- [ ] Not started — requires Stage 02 clean CSVs and Stage 03B zone config

---

## Reference Lines & Coordinate System

All metrics are computed relative to the T-maze coordinate system defined in Stage 03B (`arena_config_tmaze.json`).

```
         ┌──────┐  ┌──────┐
         │  LG  │  │  RG  │    LG = left_goal, RG = right_goal
         └──┬───┘  └───┬──┘
            │    CP    │        CP = choice_point zone
            └────┬─────┘
                 │
              [STEM]
                 │
            [ENTRY]
```

**Maze axis:** vertical line from entry midpoint to choice point — used for heading angle calculations.

---

## Tasks

### 4.1 — Trial Segmentation

- [ ] Segment each session into individual trials (entry → goal)
- [ ] Define trial start: rat enters `stem` zone from `entry`
- [ ] Define trial end: rat reaches `left_goal` or `right_goal`
- [ ] Discard incomplete or anomalous trials (no goal reached within timeout)
- [ ] Export trial index with start/end frame numbers

```python
def segment_trials(zone_sequence, fps, timeout_s=60):
    """
    zone_sequence: list of zone labels per frame
    Returns: list of (start_frame, end_frame, choice) tuples
    """
    trials = []
    in_trial = False
    for i, zone in enumerate(zone_sequence):
        if zone == "stem" and not in_trial:
            start = i
            in_trial = True
        if in_trial and zone in ("left_goal", "right_goal"):
            trials.append((start, i, zone))
            in_trial = False
        if in_trial and (i - start) / fps > timeout_s:
            in_trial = False   # timeout — discard trial
    return trials
```

### 4.2 — Turn Bias

- [ ] Count left vs right goal choices per session
- [ ] Compute `turn_bias = left_choices / (left_choices + right_choices)`
- [ ] Flag perseveration: > 3 consecutive same-side choices

### 4.3 — Path Efficiency

- [ ] Compute actual path length (Euclidean sum along trajectory)
- [ ] Compute optimal (straight-line) path length from entry to goal
- [ ] `path_efficiency = optimal_length / actual_length` (1.0 = perfect straight line)

```python
def path_efficiency(x, y, start_frame, end_frame, arena_config):
    traj_x = x[start_frame:end_frame]
    traj_y = y[start_frame:end_frame]
    dx = np.diff(traj_x); dy = np.diff(traj_y)
    actual = np.nansum(np.sqrt(dx**2 + dy**2))
    optimal = np.sqrt((traj_x[-1] - traj_x[0])**2 + (traj_y[-1] - traj_y[0])**2)
    return optimal / actual if actual > 0 else np.nan
```

### 4.4 — Decision Latency

- [ ] Time spent in `choice` zone per trial (frames × 1/fps)
- [ ] Also compute total latency from trial start to goal reach
- [ ] Flag high-latency trials (> 2 SD above rat mean)

### 4.5 — Velocity Profile

- [ ] Compute instantaneous velocity (cm/s) from body_center
- [ ] Compute mean velocity per zone per trial:
  - `velocity_stem`, `velocity_choice`, `velocity_arm`
- [ ] Detect and count freezing events (velocity < 2 cm/s for > 1s)

### 4.6 — Heading Angle

- [ ] Compute heading angle = angle between nose-to-tail vector and maze axis
- [ ] Mean heading angle in `choice` zone: indicates directional commitment
- [ ] Heading variance: high variance = undecided / anxious

```python
def heading_angle(nose_x, nose_y, tail_x, tail_y, maze_axis_angle=90):
    dx = nose_x - tail_x
    dy = nose_y - tail_y
    angle = np.degrees(np.arctan2(dy, dx))
    return angle - maze_axis_angle   # relative to maze axis
```

### 4.7 — Backtrack Rate

- [ ] Detect zone reversals: `stem → choice → stem` within same trial
- [ ] Compute backtrack rate = reversals / total trials per session

### 4.8 — Trajectory Smoothness

- [ ] Compute curvature at each point: `κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)`
- [ ] Mean curvature per trial → low = smooth navigation, high = erratic

### 4.9 — Visualization

- [ ] Plot trajectory per trial colored by velocity (colormap: cool→hot)
- [ ] Overlay all trials of a session (path density)
- [ ] Decision point dwell time map

---

## Metric Summary Table

| Metric | Per | Script |
|--------|-----|--------|
| `turn_choice` | trial | `path_analysis.py` |
| `turn_bias` | session | `path_analysis.py` |
| `path_efficiency` | trial | `path_analysis.py` |
| `decision_latency_s` | trial | `path_analysis.py` |
| `total_latency_s` | trial | `path_analysis.py` |
| `velocity_stem` | trial | `path_analysis.py` |
| `velocity_choice` | trial | `path_analysis.py` |
| `velocity_arm` | trial | `path_analysis.py` |
| `heading_angle_mean` | trial | `path_analysis.py` |
| `heading_angle_var` | trial | `path_analysis.py` |
| `backtrack_rate` | session | `path_analysis.py` |
| `trajectory_smoothness` | trial | `path_analysis.py` |
| `zone_dwell_stem_s` | trial | `path_analysis.py` |
| `zone_dwell_arm_s` | trial | `path_analysis.py` |

---

## Acceptance Criteria

- All 12 T-maze sessions have per-trial metrics exported
- `tmaze_metrics.csv` populated with no missing sessions
- Visualization plots saved for at least 2 example sessions

---

## Output Files

| Path | Description |
|------|-------------|
| `data/features/tmaze_metrics.csv` | Per-trial T-maze metrics |
| `data/heatmaps/tmaze/paths/` | Trajectory visualization plots |
| `src/path_analysis.py` | Analysis script |

---

## Next Step

→ **Stage 05:** `STAGE_05_FEATURE_ENGINEERING.md`
