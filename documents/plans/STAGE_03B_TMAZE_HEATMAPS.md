# Stage 03B — T-Maze Heatmap Generation
## Plan & Roadmap

---

## Objective

Generate spatial heatmaps from T-maze tracking data (Part1 + Part2) to visualize occupancy, movement density, and zone preference per rat and per session.

---

## Status

- [ ] Not started — requires Stage 02 clean CSVs

---

## T-Maze Zone Layout

```
         ┌──────┐  ┌──────┐
         │ Left │  │Right │     ← Goal zones (reward arms)
         │ Arm  │  │ Arm  │
         └──┬───┘  └──┬───┘
            │  Choice │
            │  Point  │
            └────┬────┘
                 │ Stem
                 │
            ┌────┴────┐
            │  Entry  │         ← Start box / entry zone
            └─────────┘
```

### Named Zones

| Zone ID | Name | Description |
|---------|------|-------------|
| `entry` | Entry zone | Start box / bottom of stem |
| `stem` | Stem | Corridor from entry to choice point |
| `choice` | Choice point | Junction between stem and arms |
| `left_arm` | Left arm | Left corridor |
| `right_arm` | Right arm | Right corridor |
| `left_goal` | Left goal | Tip of left arm |
| `right_goal` | Right goal | Tip of right arm |

---

## Tasks

### 3B.1 — Coordinate System Calibration

- [ ] Extract a reference frame from a T-maze video
- [ ] Manually define zone polygon corners in pixel coordinates
- [ ] Compute px-to-cm scale factor from known maze dimensions
- [ ] Save zone definitions to `data/arena_config_tmaze.json`

```python
arena_config = {
    "px_per_cm": 4.8,
    "zones": {
        "entry":      [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "stem":       [[...], ...],
        "choice":     [[...], ...],
        "left_arm":   [[...], ...],
        "right_arm":  [[...], ...],
        "left_goal":  [[...], ...],
        "right_goal": [[...], ...],
    }
}
```

### 3B.2 — Heatmap Types to Generate

Implement `src/heatmap.py`.

| Heatmap | Implementation | Output |
|---------|---------------|--------|
| Occupancy | 2D histogram of body_center (x, y) | `occupancy_{rat}_{session}.png` |
| Velocity | Mean speed per spatial bin | `velocity_{rat}_{session}.png` |
| Nose point | 2D histogram of nose (x, y) — investigation map | `nose_{rat}_{session}.png` |
| Entry frequency | Zone entry counts overlaid on maze | `entry_freq_{rat}_{session}.png` |
| Path density | All trajectory lines overlaid | `paths_{rat}_{session}.png` |

- [ ] Implement per-session heatmap generation
- [ ] Implement group-average heatmap (mean normalized across rats of same condition)
- [ ] Normalize all heatmaps to [0, 1] for cross-session comparison

```python
def generate_occupancy_heatmap(x, y, arena_bounds, grid_size=50, save_path=None):
    heatmap, xedges, yedges = np.histogram2d(
        x[~np.isnan(x)], y[~np.isnan(y)],
        bins=grid_size,
        range=[[arena_bounds["x_min"], arena_bounds["x_max"]],
               [arena_bounds["y_min"], arena_bounds["y_max"]]]
    )
    heatmap = heatmap / heatmap.max()   # normalize

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto")
    plt.colorbar(im, ax=ax, label="Normalized occupancy")
    ax.set_title("Occupancy Heatmap")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return heatmap
```

### 3B.3 — Zone Occupancy Time

- [ ] Per frame, assign rat to a zone using Shapely `Point.within(Polygon)`
- [ ] Compute time spent in each zone (frames × (1/fps))
- [ ] Compute zone entry counts (transitions between zones)

```python
from shapely.geometry import Point, Polygon

def assign_zone(x_frame, y_frame, zones):
    pt = Point(x_frame, y_frame)
    for zone_name, coords in zones.items():
        if Polygon(coords).contains(pt):
            return zone_name
    return "outside"
```

### 3B.4 — Per-Session Heatmap Export

- [ ] Generate and save all heatmap types for each of the 12 T-maze sessions
- [ ] Save numpy arrays alongside PNGs for downstream CNN input

```
data/heatmaps/tmaze/
├── MA1_session1_occupancy.png
├── MA1_session1_occupancy.npy
├── MA1_session1_velocity.png
...
```

### 3B.5 — Group-Average Heatmaps

- [ ] Stack normalized heatmaps per condition and compute mean
- [ ] Visualize condition differences (e.g., control vs stressed)

---

## Acceptance Criteria

- All 12 T-maze sessions have occupancy, velocity, and nose-point heatmaps
- Heatmaps are normalized and saved as both PNG and `.npy`
- Zone occupancy times computed and ready for Stage 04

---

## Output Files

| Path | Description |
|------|-------------|
| `data/arena_config_tmaze.json` | Zone polygon definitions |
| `data/heatmaps/tmaze/` | Per-session heatmap images and arrays |
| `src/heatmap.py` | Heatmap generation script |

---

## Next Step

→ **Stage 04:** `STAGE_04_PATH_ANALYSIS.md`
