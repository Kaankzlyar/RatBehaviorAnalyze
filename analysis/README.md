# Open-Field Behavior Analysis Tools

Analysis pipeline for DeepLabCut-tracked open-field recordings. Creates trajectory visualizations and spatial density heatmaps with thigmotaxis metrics.

## Tools Overview

| Tool | Purpose | Output |
|------|---------|--------|
| **show_frame_coords.py** | Interactive arena boundary selector | Terminal: `--arena` and `--inner-zone` coordinates |
| **orbit_plot.py** | Per-bodypart trajectory visualization | `*_orbit_grid.png`, `*_thigmotaxis.png` |
| **activity_heatmap.py** | Whole-animal spatial density (body_center) | `*_heatmap_histogram.png`, `*_heatmap_kde.png` |
| **bodypart_heatmaps.py** | Per-bodypart activity density grid | `*_bodypart_heatmaps.png` |

## Workflow

### Step 1: Define Arena Boundaries
```bash
python show_frame_coords.py --video ../data/DLCfiltered/OpenFieldMA1_2.mp4
```
Click 4 arena wall corners, then 4 inner-zone corners. Terminal will output:
```
--arena 396 776 153 530 --inner-zone 422 747 177 502
```

### Step 2: Orbit Trajectory & Thigmotaxis
```bash
python orbit_plot.py --arena 396 776 153 530 --inner-zone 422 747 177 502
```
Outputs:
- `OpenFieldMA1_2_orbit_grid.png` — Per-bodypart trajectories
- `OpenFieldMA1_2_thigmotaxis.png` — Body_center with thigmotaxis zones (4.7%)

### Step 3: Activity Density (Whole Animal)
```bash
python activity_heatmap.py --arena 396 776 153 530 --inner-zone 422 747 177 502
```
Outputs:
- `*_heatmap_histogram.png` — Discrete bin density
- `*_heatmap_kde.png` — Smooth kernel density (hotspots)

### Step 4: Per-Bodypart Activity Density
```bash
python bodypart_heatmaps.py --arena 396 776 153 530 --inner-zone 422 747 177 502
```
Outputs:
- `*_bodypart_heatmaps.png` — Grid of per-bodypart heatmaps

## Parameters

All scripts support:
- `--likelihood FLOAT` — Confidence threshold (default: 0.6)
- `--jump-thresh FLOAT` — Max consecutive frame distance in px (default: 60)
- `--smooth INT` — Rolling median window size (default: 5)

Heatmap scripts additionally support:
- `--bins INT` — Histogram bin count (default: 30-40)

## Data Processing Pipeline

Each script applies the same 4-step filter chain:
1. **Likelihood filter** — Remove low-confidence DLC predictions
2. **Arena bounds filter** — Set out-of-bounds points to NaN (prevents visual spillover)
3. **Jump threshold** — Detect and remove tracking jumps (temporal consistency)
4. **Rolling median smoothing** — Smooth remaining noise while preserving gaps

## Outputs Directory
All outputs saved to `../data/DLCfiltered/`

## Notes
- **Thigmotaxis metric** uses `body_center` only (single stable reference point)
- **Coordinate system** matches video frame pixel positions (Y inverted for display)
- **Frame retention** typically ~85-99% after filtering (see per-bodypart breakdown)
