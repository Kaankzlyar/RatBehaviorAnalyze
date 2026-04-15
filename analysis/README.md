# Open-Field Behavior Analysis Tools

Analysis pipeline for DeepLabCut-tracked open-field recordings. Creates trajectory visualizations and spatial density heatmaps with thigmotaxis metrics.

## ⚡ Quick Start (RECOMMENDED)

```bash
# Step 1: Define arena boundaries interactively
python show_frame_coords.py --video ../data/DLCfiltered/OpenFieldMA1_2.mp4

# Step 2: Run complete analysis (single command!)
# Copy --arena coordinates from Step 1 output, then:
python run_analysis.py --arena 396 776 153 530
```

**That's it!** All visualizations are generated automatically:
- Orbit trajectories grid
- Thigmotaxis detail (body_center)
- KDE heatmap (thesis primary)
- Per-bodypart heatmap grid

## 🔧 Master Automation Script

**File**: `run_analysis.py`

Single command generates all 4 visualization outputs.

### Options:

```bash
# Auto-calculate inner-zone (20% margin - RECOMMENDED)
python run_analysis.py --arena 396 776 153 530

# Use manually-defined inner-zone
python run_analysis.py --arena 396 776 153 530 --inner-zone 422 747 177 502

# Custom margin (25% instead of 20%)
python run_analysis.py --arena 396 776 153 530 --margin 0.25

# Stricter filtering
python run_analysis.py --arena 396 776 153 530 --likelihood 0.8 --jump-thresh 80

# Different KDE colormap
python run_analysis.py --arena 396 776 153 530 --cmap magma

# Skip specific visualizations
python run_analysis.py --arena 396 776 153 530 --skip-bodypart --skip-orbit
```

---

## 📚 Individual Tools (Manual Workflow)

If you want to run tools separately:

| Tool | Purpose | Output |
|------|---------|--------|
| **show_frame_coords.py** | Interactive arena boundary selector | Terminal: `--arena` and `--inner-zone` coordinates |
| **orbit_plot.py** | Per-bodypart trajectory visualization | `*_orbit_grid.png`, `*_thigmotaxis.png` |
| **activity_heatmap.py** | Whole-animal spatial density (body_center) | `*_heatmap_histogram.png`, `*_heatmap_kde.png` |
| **bodypart_heatmaps.py** | Per-bodypart activity density grid | `*_bodypart_heatmaps.png` |

### Manual Step-by-Step:

```bash
# Step 1: Define boundaries
python show_frame_coords.py --video ../data/DLCfiltered/OpenFieldMA1_2.mp4

# Step 2: Orbit trajectories
python orbit_plot.py --arena 396 776 153 530 --inner-zone 422 747 177 502

# Step 3: Activity density heatmaps
python activity_heatmap.py --arena 396 776 153 530 --inner-zone 422 747 177 502

# Step 4: Per-bodypart heatmaps
python bodypart_heatmaps.py --arena 396 776 153 530 --inner-zone 422 747 177 502
```

## Parameters

All scripts support:
- `--likelihood FLOAT` — Confidence threshold (default: 0.6)
- `--jump-thresh FLOAT` — Max consecutive frame distance in px (default: 60)
- `--smooth INT` — Rolling median window size (default: 5)

Heatmap scripts additionally support:
- `--bins INT` — Histogram bin count (default: 30-40)
- `--cmap STR` — KDE colormap: inferno|magma|hot|twilight (default: inferno)

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
