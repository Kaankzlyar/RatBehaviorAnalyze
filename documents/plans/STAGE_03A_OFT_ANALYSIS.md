# Stage 03A — Open Field Test (Rectangle) Analysis
## Plan & Roadmap

---

## Objective

Extract anxiety, locomotion, and exploration metrics from the rectangle arena (Part0) tracking data. These behavioral measures serve as a baseline psychological profile for each rat, cross-correlated with T-maze performance in Stage 5.

---

## Status

- [ ] Not started — requires Stage 02 clean CSVs

---

## Arena Definition

The rectangle arena is an open field test (OFT) setup. The key zones are:

```
┌─────────────────────────────┐
│  ·  ·  ·  ·  ·  ·  ·  ·  · │  ← Peripheral zone (wall-hugging region)
│  ·  ┌───────────────┐  ·  · │
│  ·  │               │  ·  · │
│  ·  │  Center zone  │  ·  · │
│  ·  │               │  ·  · │
│  ·  └───────────────┘  ·  · │
│  ·  ·  ·  ·  ·  ·  ·  ·  · │
└─────────────────────────────┘
```

- **Center zone:** inner ~25% of the arena area
- **Peripheral zone:** outer border (~10% width from walls)

---

## Tasks

### 3A.1 — Coordinate System Calibration

- [ ] Define arena boundary corners from a reference frame (pixels)
- [ ] Compute pixel-to-cm scale factor (measure known dimension)
- [ ] Define center zone polygon (inner 25% by area)
- [ ] Define peripheral zone polygon (outer 10% border)
- [ ] Save zone definitions to `data/arena_config_rectangle.json`

```python
# Example arena config structure
arena_config = {
    "arena_corners_px": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "px_per_cm": 4.2,
    "center_zone": [[cx1,cy1], [cx2,cy2], [cx3,cy3], [cx4,cy4]],
    "peripheral_width_px": 40
}
```

### 3A.2 — OFT Metric Computation

Implement `src/oft_analysis.py` to compute per-session per-rat metrics.

- [ ] **Center time ratio** — fraction of session in center zone
- [ ] **Peripheral time ratio** — fraction of session near walls (thigmotaxis index)
- [ ] **Total distance (cm)** — cumulative Euclidean displacement of body_center
- [ ] **Mean velocity (cm/s)** — total distance / session duration
- [ ] **Velocity histogram** — distribution of instantaneous speeds
- [ ] **Exploration rate** — number of unique spatial bins visited per minute
- [ ] **Immobility bouts** — periods where velocity < threshold (e.g., 2 cm/s) for > 2s
- [ ] **First center entry latency** — time until first entry into center zone

```python
# Metric implementation sketch
def compute_oft_metrics(tracking, arena_config, fps):
    """
    tracking: dict from dlc_inference.load_tracking()
    arena_config: dict with zone polygons
    fps: video frame rate
    Returns: dict of per-session metrics
    """
    body = tracking["body_center"]
    x, y = body["x"], body["y"]

    # Distance
    dx = np.diff(x); dy = np.diff(y)
    total_distance_px = np.nansum(np.sqrt(dx**2 + dy**2))
    total_distance_cm = total_distance_px / arena_config["px_per_cm"]

    # Velocity
    velocity = np.sqrt(dx**2 + dy**2) * fps / arena_config["px_per_cm"]  # cm/s

    # Zone membership (use Shapely for polygon checks)
    ...
```

### 3A.3 — Heatmap Generation (OFT)

- [ ] Occupancy heatmap — time spent per spatial bin (grid size: 10×10)
- [ ] Velocity heatmap — mean speed at each location
- [ ] Trajectory overlay — path line colored by velocity
- [ ] Save heatmaps to `data/heatmaps/rectangle/` as PNG + numpy array

```python
def occupancy_heatmap(x, y, arena_bounds, grid_size=50):
    """2D histogram of body center positions."""
    heatmap, xedges, yedges = np.histogram2d(
        x[~np.isnan(x)], y[~np.isnan(y)],
        bins=grid_size,
        range=[[arena_bounds[0], arena_bounds[2]],
               [arena_bounds[1], arena_bounds[3]]]
    )
    return heatmap / heatmap.sum()   # normalize to probability
```

### 3A.4 — Group Averages

- [ ] Compute mean ± SD of each OFT metric per condition group
- [ ] Generate group-average heatmaps (mean normalized occupancy)
- [ ] Run statistical tests: Mann-Whitney U or t-test per metric

### 3A.5 — Export

- [ ] Save per-session OFT metrics to `data/features/oft_metrics.csv`

| Column | Type | Description |
|--------|------|-------------|
| `rat_id` | str | MA1/MA3/MA5/MA7 |
| `session` | int | 1/2/3 |
| `condition` | str | control/stressed/... |
| `center_time_ratio` | float | 0–1 |
| `peripheral_time_ratio` | float | 0–1 |
| `total_distance_cm` | float | |
| `mean_velocity_cms` | float | |
| `exploration_rate` | float | bins/min |
| `immobility_bouts` | int | |
| `first_center_latency_s` | float | |

---

## Acceptance Criteria

- OFT metrics computed for all 12 Part0 sessions
- `oft_metrics.csv` populated with no missing values
- At least one heatmap generated per session
- Statistical comparison table between condition groups

---

## Output Files

| Path | Description |
|------|-------------|
| `data/arena_config_rectangle.json` | Zone definitions |
| `data/features/oft_metrics.csv` | Per-session OFT metrics |
| `data/heatmaps/rectangle/` | Occupancy and velocity heatmaps |
| `src/oft_analysis.py` | Analysis script |

---

## Next Step

→ **Stage 05:** `STAGE_05_FEATURE_ENGINEERING.md` (merge OFT + T-maze features)
