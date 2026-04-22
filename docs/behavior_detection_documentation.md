# Behavior Detection Algorithm Documentation

**Script:** `src/behavior_detection.py`  
**Task:** Rule-based detection of *rearing* and *grooming* from DeepLabCut (DLC) filtered tracking data.  
**Dataset:** `data/DLCfiltered/OpenFieldMA1_2.csv` — 5068 frames, 30 fps, Open Field arena.

---

## 1. Input Format

### DLC Filtered CSV

The CSV uses a 3-row header (scorer / bodypart / coordinate), so it is loaded with:

```python
df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
```

Column names are collapsed to `<bodypart>_<coord>` (e.g., `nose_x`, `nose_y`, `nose_likelihood`).

### Tracked Keypoints

| Keypoint | Role |
|---|---|
| `nose` | Snout position — grooming cue, wall proximity |
| `head` | Head centre — used in htdist and htd_y |
| `left_forepaw`, `right_forepaw` | Forepaw cluster — rearing posture, grooming |
| `left_hindpaw`, `right_hindpaw` | Hindpaw reference — vertical elevation |
| `tail_base` | Tail base — used in htdist and htd_y |

---

## 2. Arena Geometry

```
x_left  ≈ 397 px      x_right ≈ 775 px
y_top   ≈ 158 px      y_bottom ≈ 532 px
```

**Coordinate convention:** image coordinates — y increases **downward**.  
- Small y → top of image / top wall of arena  
- Large y → bottom of image / bottom wall of arena

---

## 3. Preprocessing

### Likelihood Masking

Each keypoint has a DLC confidence score (`_likelihood`). Keypoints below threshold are set to `NaN`:

```python
LIKELIHOOD_THRESH = 0.6
```

**Two DataFrames are maintained separately:**
- `raw_df` — original coordinates, no masking
- `masked_df` — NaN where likelihood < 0.6

The split is critical: `htdist` is computed from `raw_df` because `tail_base` is frequently occluded during wall rearing (tail pressed against wall, partially out of frame). Using masked coordinates would produce NaN for the entire htdist feature during the exact frames we want to detect.

---

## 4. Features

Six per-frame features are computed in `compute_features(raw_df, masked_df)`:

### 4.1 `htdist` — Head-to-Tail Distance

```
htdist = sqrt( (head_x - tail_base_x)² + (head_y - tail_base_y)² )
```

**Source:** `raw_df` (unmasked).  
**Rationale:** When the rat stands upright against a wall, the 2-D overhead projection compresses the head toward the tail base — the body folds in the vertical axis not visible in the top-down camera.

| Behavior | Typical range |
|---|---|
| Normal locomotion | 120–200 px |
| Compact rearing (any wall) | < 55 px |
| Bottom-wall compact rearing | 55–105 px |
| Side-wall rearing | 65–110 px |

### 4.2 `fp_hp_vert` — Forepaw–Hindpaw Vertical Separation

```
fp_y  = mean(left_forepaw_y, right_forepaw_y)   # NaN-safe mean
hp_y  = mean(left_hindpaw_y, right_hindpaw_y)   # NaN-safe mean
fp_hp_vert = hp_y - fp_y
```

**Source:** `masked_df`.  
**Sign convention:**
- **Positive** → hindpaw_y > forepaw_y → forepaws are **higher** in the image (= elevated, as in top-wall rearing)
- **Negative** → forepaw_y > hindpaw_y → forepaws are **lower** in the image (= depressed, as in bottom-wall rearing where rat hangs head down)

| Behavior | Typical range |
|---|---|
| Normal locomotion | −20 to +20 px |
| Top-wall rearing (extended) | > +45 px |
| Bottom-wall rearing (strong) | < −80 px |
| Bottom-wall rearing (compact) | −45 to −80 px |
| Grooming | −10 to +10 px |

NaN-safe mean (`df[cols].mean(axis=1)`) is used: if one paw is occluded, the other paw's value is used; only both-paw occlusion produces NaN.

### 4.3 `nose_y` — Nose Vertical Position

```
nose_y = masked_df["nose_y"]
```

Used to confirm proximity to top wall (`< 165`) or bottom wall (`> 500`).

### 4.4 `nose2fp` — Nose-to-Forepaw Distance

```
nose2fp = sqrt( (nose_x - fp_x)² + (nose_y - fp_y)² )
```

**Source:** `masked_df`.  
Primary grooming cue: the rat holds its snout against its forepaws during face-washing.

| Behavior | Typical range |
|---|---|
| Grooming | < 22 px |
| Normal locomotion | 30–100 px |

### 4.5 `htd_y` — Head-to-Tail Y-Separation (Absolute)

```
htd_y = |head_y - tail_base_y|
```

**Source:** `raw_df`.  
Measures how much the head-tail axis is tilted vertically. Used to distinguish side-wall rearing (body partially upright) from pure horizontal thigmotaxis (body flat along wall).

| Behavior | Typical range |
|---|---|
| Horizontal thigmotaxis (walking along wall) | 1–22 px |
| Side-wall rearing (body partially upright) | 40–60 px |
| Normal oblique locomotion | 60–150 px |

### 4.6 `nose_x` — Nose Horizontal Position

```
nose_x = masked_df["nose_x"]
```

Used with wall boundary constants to confirm the rat's nose is pressed against the left or right arena wall. Values can exceed arena boundaries because the nose keypoint tracks the tip of the snout, which may touch or slightly pass the wall edge.

| Wall | Typical nose_x range during rearing |
|---|---|
| Right wall (arena right ≈ 775 px) | 760–802 px |
| Left wall (arena left ≈ 397 px) | 350–430 px |

---

## 5. Classification Rules

Rules are applied in `classify_frames(feat)`. All five rearing conditions are OR-combined; the grooming condition additionally excludes any frame already labelled as rearing.

### R1 — Compact Rearing (any wall)

```
htdist < 55
```

Catches rearing against any wall when the body is strongly compressed. This is the most reliable single-feature indicator.

**Threshold calibration:** Frames 685–729 (22.8–24.3 s, confirmed rearing) show htdist = 30–52. Normal locomotion shows htdist > 120. Threshold set to 55 to leave clear margin.

**Confirmed windows:** 22.8–24.3 s, 41.2–43.0 s, 67.5–68.7 s

---

### R2 — Top-Wall Extended Rearing

```
fp_hp_vert > 45  AND  nose_y < 165
```

Catches rearing against the **top wall** when the body is not fully compressed (rat extends arms up the wall). The dual condition (forepaws elevated + nose near top boundary) avoids false positives from other upright-body postures.

**Threshold calibration:** Frames 323–371 (10.8–12.4 s) show fp_hp_vert = 48–95 and nose_y = 140–162. `REAR_NOSE_Y_MAX` was reduced from 200 to 165 to eliminate a false positive at 57–60 s (rat walking toward wall: fp_hp_vert satisfied but nose_y = 170–195).

**Confirmed windows:** 10.8–12.4 s, 59.7–63.1 s

---

### R3 — Bottom-Wall Strong Rearing

```
fp_hp_vert < -80  AND  nose_y > 500
```

Catches rearing against the **bottom wall** with a strong postural signal: forepaws are clearly below hindpaws and the nose is near the bottom boundary.

**Threshold calibration:** Frames 21–57 (0.7–1.9 s) show fp_hp_vert = −80 to −120 and nose_y = 505–530. `REAR_BOTTOM_STRONG_FPHP = -80` gives sufficient separation from locomotion (fp_hp_vert > −30).

**Confirmed windows:** 0.7–1.9 s, 33.9–34.6 s (partial)

---

### R4 — Bottom-Wall Compact Rearing

```
fp_hp_vert < -45  AND  nose_y > 500  AND  htdist < 105
```

Catches bottom-wall rearing where the postural signal is moderate but the body is simultaneously compressed. The htdist upper bound eliminates a false positive at 31.8–32.8 s (rat walking fast near bottom wall: fp_hp_vert = −48 but htdist = 105–134, velocity high).

**Threshold calibration:** True bottom-wall compact rearing (33.9–34.6 s) shows htdist = 58–100. The false positive at 31.8–32.8 s had htdist = 105–134, so the threshold is set to 105.

**Confirmed windows:** 0.7–1.9 s (overlap with R3), 33.9–34.6 s, ~29.2–31.5 s (partial)

---

### R5 — Side-Wall Rearing

```
htd_y > 40  AND  htd_y < 60
AND  ( nose_x > 760  OR  nose_x < 430 )
```

Catches rearing against the **left or right walls** of the arena. The rat's body is partially vertical (tilted axis detectable in htd_y) and the nose is pressed against the lateral boundary.

**Feature motivation:** Pure horizontal thigmotaxis (walking along the wall) produces htd_y = 1–22 — the head and tail are at nearly the same image height as the body runs horizontal. During side-wall rearing the body is partially vertical, producing htd_y = 40–60. Normal oblique locomotion produces htd_y > 60 but the nose is not at the wall boundary.

**Threshold calibration:**
- 26.1–26.7 s (right wall rearing): nose_x = 775–802, htd_y = 45–56 ✓
- 29.2–31.5 s (left wall rearing): nose_x = 350–375, htd_y = 35–51 ✓
- False positive at 13.9–15.3 s eliminated: htd_y = 1–22 (pure horizontal thigmotaxis) → filtered by `htd_y > 40`
- False positive at 6.1–7.3 s: remaining frames after htd_y filter < `MIN_BOUT_FRAMES`, discarded

**Confirmed windows:** 26.1–26.7 s, 29.2–31.5 s

---

### G1 — Grooming

```
nose2fp < 22  AND  fp_hp_vert < 10  AND  NOT rearing
```

The rat holds its snout against its forepaws during face-washing. The `fp_hp_vert < 10` guard prevents elevated-forepaw rearing postures (where the nose can also be close to the forepaws during early posture transitions) from being mislabelled as grooming.

**Threshold calibration:** Frames 2181–2196 (72.7–73.2 s, brief grooming) show nose2fp = 8–20 and fp_hp_vert = −5 to +5. False grooming at 39.7–40.8 s had nose2fp = 18–21 but fp_hp_vert = 12–22; eliminated by the fp_hp_vert < 10 guard.

**Confirmed windows:** 72.7–73.2 s (brief), 160.2–167.7 s (extended)

---

## 6. Priority and Label Assignment

```
labels["other"]    = default
labels[grooming]   = "grooming"
labels[rearing]    = "rearing"   ← overwrites grooming if both are true
```

Rearing takes priority over grooming in the rare case both conditions are simultaneously met.

---

## 7. Bout Grouping

Isolated detected frames are merged into bouts in `frames_to_bouts(frames, gap, min_dur)`:

| Parameter | Value | Meaning |
|---|---|---|
| `INTER_BOUT_GAP` | 15 frames (0.50 s) | Gaps ≤ 15 frames between positives are bridged into one bout |
| `MIN_BOUT_FRAMES` | 10 frames (0.33 s) | Bouts shorter than 10 frames are discarded as noise |

`MIN_BOUT_FRAMES` was lowered from 20 to 10 to capture the brief 0.7–1.9 s rearing bout (only ~12 frames meet R3/R4 thresholds within that window).

---

## 8. Threshold Constants Summary

| Constant | Value | Rule | Description |
|---|---|---|---|
| `LIKELIHOOD_THRESH` | 0.6 | all | DLC confidence cutoff |
| `REAR_COMPACT_HTDIST` | 55 px | R1 | Max htdist for compact rearing |
| `REAR_EXTEND_FPHP` | 45 px | R2 | Min fp_hp_vert for top-wall rearing |
| `REAR_NOSE_Y_MAX` | 165 px | R2 | Max nose_y for top-wall rearing |
| `REAR_BOTTOM_STRONG_FPHP` | −80 px | R3 | Max fp_hp_vert for strong bottom rearing |
| `REAR_BOTTOM_COMPACT_FPHP` | −45 px | R4 | Max fp_hp_vert for compact bottom rearing |
| `REAR_BOTTOM_HTDIST` | 105 px | R4 | Max htdist for compact bottom rearing |
| `REAR_BOTTOM_NOSE_Y` | 500 px | R3, R4 | Min nose_y for bottom-wall rearing |
| `REAR_SIDE_NOSE_X_RIGHT` | 760 px | R5 | Min nose_x for right-wall rearing |
| `REAR_SIDE_NOSE_X_LEFT` | 430 px | R5 | Max nose_x for left-wall rearing |
| `REAR_SIDE_HTD_Y_MIN` | 40 px | R5 | Min htd_y (excludes horizontal thigmotaxis) |
| `REAR_SIDE_HTD_Y_MAX` | 60 px | R5 | Max htd_y (excludes oblique locomotion) |
| `GROOM_NOSE2FP` | 22 px | G1 | Max nose-to-forepaw distance for grooming |
| `GROOM_MAX_FPHP` | 10 px | G1 | Max fp_hp_vert for grooming (excludes rearing) |
| `INTER_BOUT_GAP` | 15 frames | grouping | Bridge gap between bouts |
| `MIN_BOUT_FRAMES` | 10 frames | grouping | Minimum bout length |

---

## 9. Ground Truth and Performance

All windows confirmed by manual video review at 30 fps.

### Rearing Bouts (9 confirmed)

| # | Time window | Frames | Rule(s) | Wall |
|---|---|---|---|---|
| 1 | 0.7–1.9 s | ~21–57 | R3, R4 | Bottom |
| 2 | 10.8–12.4 s | ~323–371 | R2 | Top |
| 3 | 22.8–24.3 s | ~685–729 | R1 | Any (compact) |
| 4 | 26.1–26.7 s | ~783–801 | R5 | Right |
| 5 | 29.2–31.5 s | ~876–945 | R5 | Left |
| 6 | 33.9–34.6 s | ~990–1040 | R3, R4 | Bottom |
| 7 | 41.2–43.0 s | ~1236–1290 | R1 | Any (compact) |
| 8 | 59.7–63.1 s | ~1800–1893 | R2 | Top |
| 9 | 67.5–68.7 s | ~2025–2061 | R1 | Any (compact) |

### Grooming Bouts (2 confirmed)

| # | Time window | Frames | Note |
|---|---|---|---|
| 1 | 72.7–73.2 s | ~2181–2196 | Brief face-washing |
| 2 | 160.2–167.7 s | ~4806–5031 | Extended grooming |

### Known False Positives Eliminated

| Time | Description | Fix applied |
|---|---|---|
| 31.8–32.8 s | Fast walk near bottom wall (htdist=105–134) | Split R3/R4; R4 requires htdist < 105 |
| 57–60 s | Walking toward top wall (nose_y=170–195) | Reduced REAR_NOSE_Y_MAX from 200 to 165 |
| 39.7–40.8 s | Nose briefly near paws during locomotion | Added fp_hp_vert < 10 to grooming rule |
| 13.9–15.3 s | Horizontal thigmotaxis (htd_y=1–22) | Added htd_y > 40 lower bound to R5 |
| 6.1–7.3 s | Brief side-wall proximity (< 10 frames) | Remaining frames < MIN_BOUT_FRAMES; discarded |

---

## 10. Output Files

| File | Description |
|---|---|
| `<name>_behavior_timeline.png` | 3-row colour-coded timeline: ground truth / detected bouts / per-frame labels |
| `<name>_behavior_bouts.csv` | Per-bout table: behaviour, bout index, start/end frame, start/end seconds, duration |
| `<name>_behavior_frames.csv` | Per-frame table: frame index, time_s, behaviour label |

---

## 11. Limitations

1. **Rule-based, not learned:** Thresholds were calibrated on one video (`OpenFieldMA1_2`). A different camera angle, arena size, or rat size may require re-calibration.
2. **2-D projection artefacts:** The algorithm relies on overhead view only. Occluded keypoints reduce feature quality.
3. **Side-wall rearing range is narrow:** The `htd_y` window (40–60) was tuned to one arena. If the camera zoom or rat size differs, this range may need adjustment.
4. **No velocity features:** High-speed locomotion near walls (especially bottom wall) can produce brief false positives that only the htdist guard prevents.
5. **Bout bridging heuristic:** `INTER_BOUT_GAP = 15 frames` is fixed. Adjacent behaviors (e.g., rearing directly followed by grooming) may merge if the inter-bout gap is smaller than the bridge window.
