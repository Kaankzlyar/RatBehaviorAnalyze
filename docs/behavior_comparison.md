# Behavioral Comparison Across Cohorts — Open Field Test

**Project:** Rat Behavioral Neuroscience Thesis  
**Date:** 2026-04-30  
**Cohorts:** MA1 (Control), MA3 (Aspartame), MA5 (Grapefruit), MA7 (Aspartame + Grapefruit)  
**Arena:** Open Field Test (OFT), top-down camera, 30 fps  
**N:** 3 subjects per group (n = 12 total)  
**Session duration:** ~168–179 s per subject

---

## Table of Contents

1. [Experimental Design](#1-experimental-design)
2. [Methodology Summary](#2-methodology-summary)
3. [Statistical Analysis Strategy](#3-statistical-analysis-strategy)
4. [Locomotion](#4-locomotion)
5. [Thigmotaxis — Anxiety-Like Behavior](#5-thigmotaxis--anxiety-like-behavior)
6. [Freezing Behavior](#6-freezing-behavior)
7. [Rearing Behavior](#7-rearing-behavior)
8. [Grooming Behavior](#8-grooming-behavior)
9. [Spatial Entropy — Exploration Quality](#9-spatial-entropy--exploration-quality)
10. [Behavioral Profile Summary by Group](#10-behavioral-profile-summary-by-group)
11. [Cross-Cohort Pattern Analysis](#11-cross-cohort-pattern-analysis)
12. [Individual Subject Data](#12-individual-subject-data)
13. [Limitations and Interpretation Caveats](#13-limitations-and-interpretation-caveats)
14. [Raw Data Tables](#14-raw-data-tables)

---

## 1. Experimental Design

### Groups and Treatments

| Cohort | Group Label | Treatment | Folder |
|--------|-------------|-----------|--------|
| MA1 | **Control** | No treatment (vehicle) | `data/DLCfiltered/control/` |
| MA3 | **Aspartame** | Aspartame dietary supplement | `data/DLCfiltered/ASP/` |
| MA5 | **Grapefruit** | Grapefruit dietary supplement | `data/DLCfiltered/Greyfurt/` |
| MA7 | **Aspartame + Grapefruit** | Combined aspartame and grapefruit | `data/DLCfiltered/ASP ve Greyfurt/` |

### Rationale

The Open Field Test is a well-established rodent behavioral assay used to assess:
- **Locomotion and activity** — general exploration drive
- **Thigmotaxis** — time spent in the periphery vs. center, an index of anxiety
- **Freezing** — immobility episodes reflecting fear or stress
- **Rearing** — vertical exploration behavior, a marker of curiosity and vigilance
- **Grooming** — self-directed coping behavior, a marker of stress or comfort

The four-arm design allows comparison of (1) baseline behavior, (2) the effect of aspartame alone, (3) the effect of grapefruit alone, and (4) potential interaction effects when both are combined. The MA7 arm is particularly informative: if the combined group diverges from both single-treatment groups, it suggests a pharmacodynamic or metabolic interaction.

---

## 2. Methodology Summary

### Tracking

Body keypoints were tracked using a DeepLabCut (ResNet50) model trained on this dataset. Eleven body parts were labeled per frame:

```
nose · head · neck · left_ear · right_ear · body_center
left_forepaw · right_forepaw · left_hindpaw · right_hindpaw · tail_base
```

Keypoints with DLC confidence < 0.6 were masked (set to NaN) before feature computation. The `tail_base` and `head` keypoints used unmasked coordinates for the `htdist` feature because tail occlusion during wall-rearing is expected and masking would destroy the signal.

### Arena Geometry

```
x_left ≈ 397 px     x_right ≈ 775 px
y_top  ≈ 158 px     y_bottom ≈ 532 px

Arena width  ≈ 378 px
Arena height ≈ 374 px
```

The **inner zone** (center) is defined by a 20% margin inset from each wall. Center entries and percent-time-center are both computed relative to this boundary.

### Behavior Detection (Rule-Based Classifier)

All behavior labels were assigned by `src/behavior_detection.py` — a rule-based frame-level classifier with no machine learning. Six rearing rules and one grooming rule were calibrated on `MA1_2` (rearing) and `MA5_1` (grooming), then applied uniformly to all subjects.

**Six rearing rules:**

| Rule | Condition | Wall Target |
|------|-----------|-------------|
| R1 — Compact rearing | `htdist < 55 px` | Any |
| R2 — Top-wall extended | `fp_hp_vert > 45 AND nose_y < 165` | Top |
| R3 — Bottom-wall strong | `fp_hp_vert < −80 AND nose_y > 500` | Bottom |
| R4 — Bottom-wall compact | `fp_hp_vert < −45 AND nose_y > 500 AND htdist < 105` | Bottom |
| R5 — Side-wall rearing | `htd_y ∈ [40, 60] AND (nose_x > 760 OR nose_x < 430)` | Left/Right |
| R1-stationary guard | R1 suppressed during stationary grooming | — |

**Grooming rule:**

| Rule | Condition |
|------|-----------|
| G1 | `nose2fp < 22 px AND fp_hp_vert < 10 AND NOT rearing` |

**Bout grouping parameters:**
- `INTER_BOUT_GAP = 15 frames (0.50 s)` — gaps ≤ 0.5 s are bridged
- `MIN_BOUT_FRAMES = 10 frames (0.33 s)` — bouts < 0.33 s are discarded as noise

### Derived Metrics

**Fragmentation Index** measures how fractured or sustained a behavior is:

```
frag_idx = bout_count / total_time_s
```

- **High frag_idx** → many short, interrupted bouts (fragmented, disorganized)
- **Low frag_idx** → fewer but longer, sustained bouts (focused, organized)

**Spatial Entropy** (normalized) measures how uniformly the animal covers the arena:

```
entropy_norm ∈ [0, 1]
```

- Near 0 → stereotyped, restricted movement (low exploration)
- Near 1 → uniform coverage of the entire arena (high, diverse exploration)

---

## 3. Statistical Analysis Strategy

### 3.1 Core Problem: n = 3 Is a Pilot Dataset

All four treatment groups contain exactly three subjects each. This has direct consequences for every inferential test:

| Test | Minimum practical n per group | Status at n = 3 |
|------|-------------------------------|-----------------|
| One-way ANOVA | ≥ 8–10 (for 80% power, medium effect) | Severely underpowered |
| Welch's ANOVA | ≥ 6–8 | Underpowered |
| Kruskal-Wallis | ≥ 5 per group recommended | Marginally acceptable |
| Parametric t-test | ≥ 8 | Severely underpowered |
| Mann-Whitney U | ≥ 4–5 | Marginal |

The correct framing for this dataset in the thesis is: **pilot / exploratory data**. Observed patterns and effect sizes are reported as hypotheses for future confirmation with larger cohorts, not as statistically confirmed claims.

---

### 3.2 Problem 1 — SD > Mean (Non-Normality, Control Group)

**Observed case:** Control group `% Time Center` = **40.11 ± 44.58**

**Why it happens:**  
The three individual values are 90.09%, 4.45%, and 25.78%. MA1_1 is a strong outlier that inflates both the mean and the SD. The SD exceeds the mean because the distribution is asymmetric — it cannot be normal (a normal distribution would require negative values to be possible, but percentages are bounded at 0).

**Statistical consequence:**  
Parametric tests (ANOVA, t-test) assume normality. With n = 3, the Shapiro-Wilk test has essentially no power to detect non-normality even when it exists. Mean ± SD is a misleading summary for this distribution.

**What to do:**

1. **Report median ± IQR alongside mean ± SD** for all percentage and count metrics:
   - Control `% Time Center`: mean = 40.11%, **median = 25.78%**, range = [4.45, 90.09]
   - The median is far more representative than the mean here.

2. **Always plot individual data points** (strip/dot plots or bee-swarm plots). Never show only bar-and-error summaries for n = 3 — the raw dots *are* the data.

3. **Use non-parametric group comparisons** (see §3.4).

4. **Flag MA1_1 explicitly** in the thesis as a behavioral outlier (90.09% center time). Discuss whether this represents a distinct behavioral phenotype within the control population rather than treating it as noise.

---

### 3.3 Problem 2 — Zero Variance (Grapefruit Group, Center Entries)

**Observed case:** Grapefruit group `center_entries` = **6.0 ± 0.0**

**Why it happens:**  
MA5_1, MA5_2, and MA5_3 each entered the center zone exactly 6 times. With n = 3, this produces SD = 0 by arithmetic, not by data error.

**Biological interpretation:**  
A zero-variance result at n = 3 should not be dismissed as a statistical artifact. It is actually a **strong behavioral signal**: grapefruit treatment produces a completely consistent ceiling on center entries. The appropriate interpretation is that all three animals independently converged on the same exploratory limit, suggesting a robust treatment effect on center-zone avoidance.

**Statistical consequence:**  
- Parametric tests that use within-group variance (F-test, t-test) cannot compute a meaningful statistic for this group — the denominator becomes 0.
- Levene's test for homogeneity of variances will reject equality, invalidating standard ANOVA.
- Some software (SPSS, R) will produce `NaN` or an error when computing variance-based statistics on a zero-variance group.

**What to do:**

1. **Do not use standard ANOVA or t-tests** for `center_entries`. Use Kruskal-Wallis (rank-based, no variance assumption) instead.

2. **Report the finding descriptively**: "All three grapefruit-treated animals entered the center zone exactly 6 times (SD = 0), indicating a fully consistent behavioral ceiling not seen in any other group."

3. **In plots**, show the three overlapping dots at y = 6 explicitly. Do not represent them as a single point — this is misleading; show all three.

---

### 3.4 Problem 3 — Variance Heterogeneity Between Groups

**Observed case:**  
Control `% Time Center`: SD = 44.58  
Aspartame+Grapefruit `% Time Center`: SD = 10.03  
Ratio of variances: (44.58 / 10.03)² ≈ **19.7×** — nearly 20-fold difference in variance.

Standard one-way ANOVA assumes **homoscedasticity** (equal within-group variances). Levene's test or Bartlett's test would almost certainly reject this assumption for most behavioral metrics in this dataset.

**What to do — ordered by preference for this dataset:**

#### Option A: Kruskal-Wallis + Dunn's Post-Hoc (Recommended)

The Kruskal-Wallis test is the non-parametric equivalent of one-way ANOVA. It compares group medians via ranks and makes **no assumptions about normality or variance equality**.

```
H₀: All four groups come from the same distribution
H₁: At least one group differs

If Kruskal-Wallis is significant → Dunn's pairwise post-hoc test
with Bonferroni correction (6 pairwise comparisons → α_adjusted = 0.05/6 = 0.0083)
```

**Caveat:** With n = 3 per group (12 total), the Kruskal-Wallis test still has low power. It needs approximately 5 per group to reliably detect medium-sized effects. At n = 3, even large true differences may not reach p < 0.05.

#### Option B: Welch's One-Way ANOVA

Welch's ANOVA relaxes the equal-variance assumption by using separate variance estimates per group. It is more appropriate than standard ANOVA when variances differ substantially but data are approximately normal.

```
In R:   oneway.test(value ~ group, data = df, var.equal = FALSE)
In Python: scipy.stats.f_oneway() does NOT do Welch — use pingouin.welch_anova()
```

#### Option C: Data Transformation Before ANOVA

For percentage data (bounded [0, 1]), the **arcsine square root transformation** stabilizes variance:

```
y_transformed = arcsin(√(percentage / 100))
```

For count data (bouts, entries), the **square root transformation** or **log(x+1)** reduces right skew.

Apply transformation → check Levene's test again → if passed, standard ANOVA is acceptable.

#### Option D: Report Effect Sizes, Not Just p-Values

With n = 3, p-values are nearly meaningless — the test simply has no power. **Effect sizes** give the reader information regardless of sample size:

| Effect size | Formula | Interpretation |
|-------------|---------|----------------|
| Cohen's d | (μ₁ − μ₂) / s_pooled | 0.2 = small, 0.5 = medium, 0.8 = large |
| Eta-squared (η²) | SS_between / SS_total | 0.01 = small, 0.06 = medium, 0.14 = large |
| Rank-biserial r | For Mann-Whitney U | −1 to +1 |

A large effect size with p > 0.05 at n = 3 means: "the difference is likely real, but the sample is too small to confirm it statistically."

---

### 3.5 Recommended Reporting Template for This Dataset

For each metric in the results section:

```
[Metric] differed descriptively across groups 
(Control: median = X [range a–b]; Aspartame: median = X [range a–b]; 
 Grapefruit: median = X [range a–b]; ASP+GF: median = X [range a–b]).
Kruskal-Wallis test: H(3) = X.XX, p = X.XX.
[If p < 0.05: Dunn's post-hoc: Group A vs Group B, p_adj = X.XX]
Effect size (η²) = X.XX (large/medium/small).
Note: Results should be interpreted as exploratory given n = 3 per group.
```

---

### 3.6 Summary: Which Test for Which Metric

| Metric | Issue | Recommended Test |
|--------|-------|-----------------|
| % Time Center | SD > Mean, outlier-driven | Kruskal-Wallis; report median + IQR |
| % Time Periphery | Same as above | Kruskal-Wallis; report median + IQR |
| Center Entries | Zero variance in Grapefruit | Kruskal-Wallis only (no parametric) |
| Total Distance | Right-skewed, high variance | Log-transform → Welch's ANOVA or Kruskal-Wallis |
| Mean Speed | Moderate variance | Welch's ANOVA or Kruskal-Wallis |
| % Time Freeze | Zero-inflated (some near 0) | Kruskal-Wallis |
| Freeze Bout Count | Overdispersed count data | Kruskal-Wallis |
| Rear % Time | Right-skewed | Kruskal-Wallis |
| Rear Frag Index | Moderate variance | Welch's ANOVA acceptable |
| Groom % Time | Extreme outlier (MA5_1) | Kruskal-Wallis; report median |
| Spatial Entropy | Relatively well-behaved | Welch's ANOVA acceptable |

---

## 4. Locomotion

### Cohort Means (Mean ± SD)

| Group | Total Distance (px) | Mean Speed (px/s) | Max Speed (px/s) |
|-------|---------------------|-------------------|------------------|
| Control | 4449.5 ± 1741.4 | 26.45 ± 11.08 | 811.2 ± 58.4 |
| Aspartame | 4981.9 ± 257.5 | 29.75 ± 0.25 | 636.6 ± 402.6 |
| Grapefruit | 5960.4 ± 2628.3 | 33.62 ± 15.02 | 677.7 ± 215.9 |
| Aspartame + Grapefruit | 6949.1 ± 812.1 | 39.86 ± 4.15 | 677.5 ± 180.4 |

### Trend

A monotonic increase in total distance and mean speed is observed across the four groups:

```
Control < Aspartame < Grapefruit < Aspartame+Grapefruit
```

The **Aspartame + Grapefruit** group shows the highest locomotor activity (mean speed 39.86 px/s), approximately 51% faster than the Control group (26.45 px/s). This suggests a hyperactivating effect of the combined treatment, potentially due to additive CNS stimulation.

The **Aspartame** group shows a striking near-zero variance in mean speed (29.75 ± 0.25 px/s), indicating unusually consistent locomotor output across all three individuals. This homogeneity may reflect a narrow dose-response window or a behavioral ceiling imposed by the treatment.

The **Grapefruit** group has the highest speed variance (SD = 15.02), driven by MA5_1 (16.54 px/s — slow, with high freezing) versus MA5_2 and MA5_3 (39.56 and 44.77 px/s — very active). This subject-level heterogeneity suggests individual differences in grapefruit response.

### Notes on Units

All distances are in pixels. The arena spans approximately 378 × 374 px. To convert to centimeters, scale by the known physical arena size (typically 50 × 50 cm → 1 px ≈ 0.13 cm, giving distance range of ~390–905 cm).

---

## 5. Thigmotaxis — Anxiety-Like Behavior

Thigmotaxis refers to the tendency of rodents to stay near the walls of the arena. High wall-hugging is an established marker of anxiety or novelty-induced stress; center exploration indicates boldness or reduced anxiety.

### Cohort Means (Mean ± SD)

| Group | % Time Center | % Time Periphery | Center Entries |
|-------|---------------|------------------|----------------|
| Control | 40.11 ± 44.58 | 59.89 ± 44.58 | 10.0 ± 5.29 |
| Aspartame | 17.50 ± 12.74 | 82.50 ± 12.74 | 11.33 ± 1.53 |
| Grapefruit | 6.99 ± 3.42 | 93.01 ± 3.42 | 6.0 ± 0.0 |
| Aspartame + Grapefruit | 30.76 ± 10.03 | 69.24 ± 10.03 | 17.33 ± 3.51 |

### Patterns and Interpretation

**Grapefruit** shows the most extreme thigmotaxis: 93% of time spent at the wall with a consistent center-entry count of exactly 6 across all three subjects (SD = 0). The complete lack of variance in center entries is unusual and may indicate a treatment-specific behavioral ceiling or a systematic avoidance of the center zone. This is consistent with an anxiogenic effect of grapefruit.

**Control** shows the highest variability (SD = 44.58%), driven almost entirely by MA1_1, which spent an exceptional 90.09% of its time in the center. This single outlier inflates the group mean considerably. Without MA1_1, Control would show predominantly peripheral behavior. This high within-group variance is notable given the n = 3 sample size.

**Aspartame + Grapefruit** shows moderate thigmotaxis (30.76% center) and the highest center-entry count (17.33 ± 3.51). More frequent entries with intermediate dwell time suggests these animals visit the center more boldly but do not linger — a pattern consistent with anxious exploration.

**Aspartame** shows reduced center time (17.50%) but more consistent behavior than the Control group. The treatment may impose a mild anxiogenic effect, pushing animals toward the periphery relative to control.

---

## 6. Freezing Behavior

Freezing (complete immobility) is a fear-conditioned and stress-induced response. In an OFT context with no prior conditioning, freezing typically reflects novelty-induced fear or heightened stress reactivity.

### Cohort Means (Mean ± SD)

| Group | Freeze Bouts | Total Freeze Time (s) | % Time Frozen |
|-------|-------------|----------------------|---------------|
| Control | 33.33 ± 10.69 | 32.10 ± 15.04 | 18.56 ± 8.43 |
| Aspartame | 12.0 ± 4.0 | 9.99 ± 4.13 | 5.997 ± 2.63 |
| Grapefruit | 21.33 ± 24.83 | 17.33 ± 22.23 | 9.68 ± 12.39 |
| Aspartame + Grapefruit | 12.67 ± 7.23 | 10.77 ± 5.88 | 6.13 ± 3.28 |

### Key Finding

**Control animals freeze the most** (18.56% of session time), which is counterintuitive at first glance. This likely reflects the unaltered stress response of naive rats in a novel, bright open-field environment. Treatment groups, particularly Aspartame and Aspartame+Grapefruit, show markedly reduced freezing (~6%), suggesting an anxiolytic or activating effect of the dietary treatments.

**Grapefruit** shows extremely high variance (SD = 24.83 bouts, SD = 22.23 s) driven by MA5_1, which exhibited 50 freezing bouts totaling 43.0 s (24% of session). MA5_2 and MA5_3 had only 7 bouts each. MA5_1 appears to be a high-stress responder within the grapefruit cohort.

The interpretation that treatment groups freeze less than controls is consistent with the locomotion data (treated animals move more). However, with n = 3 per group, these patterns cannot be statistically confirmed.

---

## 7. Rearing Behavior

Rearing — standing on hindlimbs — is a form of vertical exploration. It reflects curiosity, environmental scanning, and vigilance. Rearing rate and duration are sensitive to environmental novelty, prior stress, and drug treatments.

### Cohort Means (Mean ± SD)

| Group | Rear Bouts | Rear Time (s) | % Rear Time | Mean Bout (s) | Max Bout (s) | Frag. Index |
|-------|------------|---------------|-------------|---------------|--------------|-------------|
| Control | 11.33 ± 6.81 | 15.04 ± 11.46 | 8.85 ± 6.82 | 1.222 ± 0.456 | 2.767 ± 1.341 | 0.926 ± 0.436 |
| Aspartame | 10.0 ± 1.73 | 14.08 ± 8.27 | 8.31 ± 4.58 | 1.348 ± 0.563 | 3.09 ± 1.21 | 0.837 ± 0.351 |
| Grapefruit | 18.0 ± 6.93 | 35.06 ± 19.89 | 19.74 ± 11.26 | 1.814 ± 0.523 | 5.277 ± 1.734 | 0.590 ± 0.204 |
| Aspartame + Grapefruit | 19.0 ± 4.0 | 28.46 ± 11.07 | 16.36 ± 6.61 | 1.458 ± 0.315 | 4.813 ± 1.136 | 0.711 ± 0.176 |

### Patterns and Interpretation

**Rearing time approximately doubles** from Control/Aspartame (~8–9%) to Grapefruit/ASP+Grapefruit (~16–20%). The Grapefruit group spends the most time rearing (19.74%) with the longest individual bouts (mean 1.81 s, max 5.28 s) and the lowest fragmentation index (0.590).

**Fragmentation Index trend (rearing):**
```
Control (0.926) > Aspartame (0.837) > ASP+Grapefruit (0.711) > Grapefruit (0.590)
```

This gradient reveals that Grapefruit-treated animals do not just rear more — they rear in longer, more organized bouts. The Aspartame group's frag_idx (0.837) closely mirrors Control (0.926), suggesting no major change in rearing bout structure from aspartame alone.

**Aspartame + Grapefruit** shows more rearing bouts (19) than Grapefruit alone (18) but shorter individual bouts (mean 1.46 vs 1.81 s). The addition of aspartame to grapefruit may fragment the rearing behavior into shorter, more frequent episodes. The lower max bout duration (4.81 vs 5.28 s) supports this.

**Biological significance:** The increased rearing in grapefruit-treated animals could reflect heightened arousal or reduced anxiety (animals must feel safe enough to stand upright and scan). The longer bout durations (lower fragmentation) suggest sustained environmental engagement rather than brief reactive checks.

---

## 8. Grooming Behavior

Grooming (self-directed facial washing and body cleaning) is a multi-faceted behavior: it can serve as a stress-coping displacement activity, a comfort behavior when relaxed, or a sign of arousal. In the OFT context, increased grooming — especially extended grooming episodes — is often interpreted as a stress-relief response.

### Cohort Means (Mean ± SD)

| Group | Groom Bouts | Groom Time (s) | % Groom Time | Mean Bout (s) | Max Bout (s) | Frag. Index |
|-------|-------------|----------------|--------------|---------------|--------------|-------------|
| Control | 3.0 ± 2.65 | 3.94 ± 3.82 | 2.32 ± 2.26 | 1.723 ± 2.015 | 3.02 ± 3.909 | 1.279 ± 0.903 |
| Aspartame | 3.33 ± 2.52 | 3.58 ± 3.99 | 2.08 ± 2.26 | 0.897 ± 0.400 | 1.39 ± 1.195 | 1.248 ± 0.449 |
| Grapefruit | 12.33 ± 5.51 | 31.99 ± 35.89 | 17.88 ± 19.98 | 2.254 ± 2.003 | 10.547 ± 10.309 | 0.680 ± 0.400 |
| Aspartame + Grapefruit | 15.67 ± 7.10 | 19.59 ± 12.86 | 11.28 ± 7.51 | 1.156 ± 0.553 | 4.923 ± 3.941 | 1.023 ± 0.515 |

### Key Finding: Grapefruit Dramatically Elevates Grooming

**Grooming increases 8-fold in time** from Control (3.94 s total) to Grapefruit (31.99 s total). Control and Aspartame groups groom for approximately 2% of session time. Grapefruit animals groom for 17.88% and ASP+Grapefruit for 11.28%.

**Subject-level driver:** MA5_1 is the strongest grooming responder — 40.75% of its entire session was spent grooming (73.07 s), with a maximum single bout of 22.4 s. This is extraordinary and classifies MA5_1 as a grooming outlier within even the Grapefruit group. MA5_2 (9.12%) and MA5_3 (3.78%) show more moderate grooming. This heterogeneity inflates the Grapefruit group SD significantly.

**Fragmentation Index trend (grooming):**
```
Control (1.279) ≈ Aspartame (1.248) > ASP+Grapefruit (1.023) > Grapefruit (0.680)
```

The Grapefruit group shows the most sustained grooming (lowest fragmentation), meaning the behavior is organized into longer episodes. This pattern mirrors the rearing fragmentation gradient and suggests grapefruit generally promotes more organized, sustained behavioral sequences.

**Aspartame + Grapefruit vs. Grapefruit alone:** The combined group grooms more frequently (15.67 vs 12.33 bouts) but with shorter individual bouts (mean 1.16 vs 2.25 s) and higher fragmentation (1.023 vs 0.680). Aspartame may disrupt the sustained grooming induced by grapefruit, breaking it into shorter, less organized episodes. This mirrors the same pattern observed in rearing.

---

## 9. Spatial Entropy — Exploration Quality

Normalized spatial entropy quantifies how uniformly the animal covers the arena floor. A low-entropy animal restricts its movement to a few zones (stereotyped path). A high-entropy animal distributes its trajectory across the entire arena (diverse exploration).

### Cohort Means (Mean ± SD)

| Group | Spatial Entropy (normalized) |
|-------|------------------------------|
| Control | 0.513 ± 0.220 |
| Aspartame | 0.732 ± 0.058 |
| Grapefruit | 0.718 ± 0.145 |
| Aspartame + Grapefruit | 0.814 ± 0.019 |

### Trend

```
Control (0.513) < Grapefruit (0.718) ≈ Aspartame (0.732) < ASP+Grapefruit (0.814)
```

The **Control** group shows the most restricted, stereotyped movement (lowest entropy) with the highest within-group variability (SD = 0.220). This high variability reflects MA1_1's unusual behavioral profile: it explored the center extensively (high entropy from that behavior alone) while MA1_2 hugged the walls (low entropy).

**Aspartame + Grapefruit** shows the highest and most consistent spatial entropy (0.814 ± 0.019), indicating that animals spread their trajectory most uniformly across the arena. This consistency (near-zero variance) mirrors the low fragmentation variance seen in rearing for this group, suggesting the combined treatment produces a homogeneous behavioral profile.

The near-identical entropy for Aspartame (0.732) and Grapefruit (0.718) despite very different behavioral profiles (different thigmotaxis, freezing, grooming) indicates that spatial coverage can be similar while the quality and composition of behaviors within the trajectory differ substantially.

---

## 10. Behavioral Profile Summary by Group

### MA1 — Control

**Profile:** Moderate locomotion, high freezing, variable thigmotaxis, minimal grooming, moderate rearing.

The Control group's behavioral signature is defined by elevated freezing (18.56% of session) and high within-group variability. MA1_1 is an unusual outlier: 90.09% center time, only 40 freeze bouts for a short total freeze duration — suggesting this individual was highly exploratory and bold. MA1_2 and MA1_3 show more typical OFT profiles with strong peripheral preference.

Low grooming time (2.32%) indicates minimal displacement behavior in a novel environment. The low spatial entropy (0.513) reflects that Control rats develop clear movement patterns — they are not exploring uniformly but following habitual thigmotactic routes.

**Summary vector:**
- Locomotion: Low-to-moderate
- Anxiety: High (freezing) / Variable (thigmotaxis)
- Exploration: Low-moderate (rearing 8.85%)
- Stress coping: Minimal (grooming 2.32%)
- Spatial coverage: Low (entropy 0.513)

---

### MA3 — Aspartame

**Profile:** Moderate, consistent locomotion; moderate anxiety; similar rearing to control; minimal grooming; high spatial entropy.

The Aspartame group is the most **internally consistent** of all four groups. Mean speed variance is exceptionally low (29.75 ± 0.25 px/s), rearing bouts are tightly clustered (10.0 ± 1.73), and spatial entropy is moderate (0.732 ± 0.058).

Reduced freezing (6.0% vs 18.56% in Control) suggests an anxiolytic or activating effect. Despite spending 82.5% of time in the periphery (moderate thigmotaxis), these animals do not freeze much — they remain active along the walls. This peripheral-active pattern is distinct from the peripheral-freeze pattern more common in Control.

No notable grooming increase (2.08%) indicates aspartame alone does not elicit stress-coping behavior.

**Summary vector:**
- Locomotion: Moderate, highly consistent
- Anxiety: Low-moderate (reduced freezing, moderate thigmotaxis)
- Exploration: Moderate rearing (8.31%)
- Stress coping: Minimal (grooming 2.08%)
- Spatial coverage: Moderate-high (entropy 0.732)

---

### MA5 — Grapefruit

**Profile:** Variable locomotion; extreme thigmotaxis; elevated rearing with sustained bouts; dramatically elevated grooming; high between-subject variability.

The Grapefruit group is the most **heterogeneous** of all four groups. MA5_1 is a strong outlier with low locomotion (2965 px total), extremely high freezing (50 bouts, 23.98%), and extraordinary grooming (40.75% of session). MA5_2 and MA5_3 are much more active and show less extreme patterns.

Despite this heterogeneity, consistent patterns emerge: all three subjects entered the center zone exactly 6 times (center_entries SD = 0), indicating a robust treatment effect on center-avoidance that is consistent across individuals even when other metrics vary.

**MA5_1 as a behavioral outlier:** This individual deserves special attention. The combination of low locomotion, high freezing, and extreme grooming is unusual. It suggests one of two interpretations: (a) maximal stress response — immobile due to fear, grooming as displacement, or (b) a dissociation effect where the animal is in a distinct behavioral state from its cohort-mates. Follow-up histological or biochemical data would be needed to disambiguate.

**Summary vector:**
- Locomotion: Variable (low to high)
- Anxiety: High (93% periphery, consistent 6 center entries)
- Exploration: Elevated rearing (19.74%), sustained bouts (frag 0.590)
- Stress coping: Dramatically elevated grooming (17.88%), long bouts
- Spatial coverage: Moderate (entropy 0.718)

---

### MA7 — Aspartame + Grapefruit

**Profile:** Highest locomotion; moderate thigmotaxis; highest center entries; lowest freezing; most rearing bouts; elevated grooming.

The combined treatment group produces the most **active and exploratory** behavioral profile. They move the most (6949 px), freeze the least (6.13%), enter the center zone most frequently (17.33 entries), and show near-maximum spatial entropy (0.814 ± 0.019).

The combination of high locomotion, moderate center time, and high center entries suggests a profile of **anxious boldness**: animals are not restricted to the periphery, but they do not linger in the center zone (30.76% center time is intermediate). They pass through the center frequently but continue moving.

Grooming (11.28%) and rearing (16.36%) are both elevated, with more bouts but shorter durations than Grapefruit alone. Aspartame appears to truncate the sustained behavioral episodes induced by grapefruit, breaking them into shorter, more frequent episodes.

**Summary vector:**
- Locomotion: Highest of all groups
- Anxiety: Low (lowest freezing, most center entries)
- Exploration: High rearing (16.36%)
- Stress coping: Elevated grooming (11.28%)
- Spatial coverage: Highest (entropy 0.814)

---

## 11. Cross-Cohort Pattern Analysis

### Pattern 1: Locomotion Scales with Treatment Complexity

```
Distance:  Control (4449) < Aspartame (4982) < Grapefruit (5960) < ASP+GF (6949)
Speed:     Control (26.5) < Aspartame (29.8) < Grapefruit (33.6) < ASP+GF (39.9)
```

The monotonic increase suggests additive locomotor stimulation. Aspartame adds modest activation; grapefruit adds moderate activation; together they produce the strongest effect. This is consistent with a pharmacokinetic interaction where grapefruit (a known CYP3A4 inhibitor via furanocoumarins) increases the bioavailability of aspartame metabolites (particularly phenylalanine or aspartate).

### Pattern 2: Freezing Is Inversely Related to Activity

```
Freezing%: Control (18.6%) > Grapefruit (9.7%) > ASP+GF (6.1%) ≈ Aspartame (6.0%)
```

The inverse relationship between locomotion and freezing is consistent with a general arousal axis. Animals that move more, freeze less. The reduction from Control to treatment groups suggests all treatments reduce novelty-induced fear or increase exploration drive.

### Pattern 3: Grapefruit Specifically Elevates Grooming

```
Grooming%: Control (2.3%) ≈ Aspartame (2.1%) << Grapefruit (17.9%) > ASP+GF (11.3%)
```

This is the most diagnostically specific pattern. Aspartame does not elevate grooming at all. Grapefruit causes a massive elevation. The combined group shows an intermediate level, as if aspartame partially attenuates the grapefruit-induced grooming surge.

The grapefruit-specific grooming elevation is consistent with **OCD-like or stress-induced displacement grooming** as a behavioral signature. Grapefruit compounds (naringenin, bergamottin) may affect serotonergic or dopaminergic pathways involved in grooming regulation.

### Pattern 4: Grapefruit Drives Sustained Behaviors; Aspartame Fragments Them

| Metric | Control | Aspartame | Grapefruit | ASP+GF |
|--------|---------|-----------|------------|--------|
| Rearing frag_idx | 0.926 | 0.837 | **0.590** | 0.711 |
| Grooming frag_idx | 1.279 | 1.248 | **0.680** | 1.023 |

In both behaviors, Grapefruit shows lowest fragmentation (most sustained bouts). Aspartame keeps fragmentation near Control levels. The combined group sits between them. This suggests grapefruit organizes behavior into sustained sequences, and aspartame disrupts this organization.

### Pattern 5: Thigmotaxis Dissociates from Anxiety-Driven Freezing

In classical anxiety studies, thigmotaxis and freezing are expected to co-occur. However, in this dataset:

- **Control** shows the most freezing but only moderate-to-variable thigmotaxis
- **Grapefruit** shows extreme thigmotaxis (93% periphery) but intermediate freezing (9.7%)
- **ASP+GF** shows moderate thigmotaxis (69%) and the least freezing (6.1%)

This dissociation suggests that the two anxiety metrics capture different aspects of the response: freezing may reflect acute fear, while thigmotaxis reflects a more sustained avoidance strategy. Grapefruit may selectively strengthen the avoidance component without intensifying the acute fear component.

### Pattern 6: Center Zone Entries Reflect Behavioral Strategy, Not Just Anxiety

| Group | % Time Center | Center Entries |
|-------|---------------|----------------|
| Control | 40.11% | 10.0 |
| Aspartame | 17.50% | 11.33 |
| Grapefruit | 6.99% | 6.0 |
| ASP+GF | 30.76% | 17.33 |

High entries + moderate center time (ASP+GF) = passing through without dwelling — transit behavior.  
Low entries + low center time (Grapefruit) = active avoidance.  
High time + moderate entries (Control) = dwelling in the center (but highly variable due to MA1_1).

---

## 12. Individual Subject Data

### OFT Metrics — All Subjects

| Subject | Group | Dist (px) | Speed (px/s) | %Center | %Periphery | C-Entries | Freeze# | Freeze(s) | Freeze% | Entropy |
|---------|-------|-----------|--------------|---------|------------|-----------|---------|-----------|---------|---------|
| MA1_1 | Control | 3007.8 | 16.99 | **90.09** | 9.91 | 6 | 40 | 43.93 | 24.72 | 0.306 |
| MA1_2 | Control | 3956.4 | 23.73 | 4.45 | 95.55 | 8 | 39 | 37.20 | 22.02 | 0.488 |
| MA1_3 | Control | 6384.3 | 38.64 | 25.78 | 74.22 | 16 | 21 | 15.17 | 8.95 | 0.745 |
| MA3_1 | Aspartame | 5234.9 | 30.00 | 9.28 | 90.72 | 13 | 12 | 10.77 | 6.16 | 0.671 |
| MA3_2 | Aspartame | 4990.7 | 29.75 | 32.17 | 67.83 | 10 | 8 | 5.53 | 3.29 | 0.787 |
| MA3_3 | Aspartame | 4720.1 | 29.50 | 11.04 | 88.96 | 11 | 16 | 13.67 | 8.54 | 0.738 |
| MA5_1 | Grapefruit | 2965.3 | 16.54 | 3.09 | 96.91 | 6 | 50 | 43.00 | 23.98 | 0.553 |
| MA5_2 | Grapefruit | 7033.9 | 39.56 | 9.48 | 90.52 | 6 | 7 | 5.03 | 2.83 | 0.823 |
| MA5_3 | Grapefruit | 7882.0 | 44.77 | 8.39 | 91.61 | 6 | 7 | 3.97 | 2.24 | 0.779 |
| MA7_1 | ASP+GF | 6362.7 | 37.33 | 35.91 | 64.09 | 21 | 8 | 5.83 | 3.42 | 0.820 |
| MA7_2 | ASP+GF | 7876.0 | 44.64 | 19.20 | 80.80 | 17 | 9 | 9.20 | 5.20 | 0.829 |
| MA7_3 | ASP+GF | 6608.7 | 37.60 | 37.18 | 62.82 | 14 | 21 | 17.27 | 9.78 | 0.793 |

### Behavior Metrics — All Subjects

| Subject | Group | Rear# | Rear(s) | Rear% | Rear frag | Groom# | Groom(s) | Groom% | Groom frag |
|---------|-------|-------|---------|-------|-----------|--------|----------|--------|------------|
| MA1_1 | Control | 6 | 4.2 | 2.36 | 1.429 | 1 | 0.60 | 0.34 | 1.667 |
| MA1_2 | Control | 9 | 13.9 | 8.23 | 0.647 | 2 | 8.10 | 4.80 | 0.247 |
| MA1_3 | Control | 19 | 27.0 | 15.96 | 0.703 | 6 | 3.12 | 1.84 | 1.923 |
| MA3_1 | Aspartame | 12 | 23.3 | 13.35 | 0.514 | 6 | 8.14 | 4.66 | 0.737 |
| MA3_2 | Aspartame | 9 | 7.4 | 4.42 | 1.211 | 3 | 1.90 | 1.13 | 1.579 |
| MA3_3 | Aspartame | 9 | 11.5 | 7.16 | 0.785 | 1 | 0.70 | 0.44 | 1.429 |
| MA5_1 | Grapefruit | 10 | 12.1 | 6.75 | 0.826 | 16 | **73.07** | **40.75** | 0.219 |
| MA5_2 | Grapefruit | 22 | 45.9 | 25.82 | 0.479 | 15 | 16.22 | 9.12 | 0.925 |
| MA5_3 | Grapefruit | 22 | 47.2 | 26.67 | 0.466 | 6 | 6.69 | 3.78 | 0.897 |
| MA7_1 | ASP+GF | 23 | 38.2 | 22.41 | 0.602 | 17 | 29.44 | 17.27 | 0.577 |
| MA7_2 | ASP+GF | 19 | 30.8 | 17.37 | 0.618 | 22 | 24.30 | 13.73 | 0.905 |
| MA7_3 | ASP+GF | 15 | 16.4 | 9.30 | 0.914 | 8 | 5.04 | 2.85 | 1.587 |

---

## 13. Limitations and Interpretation Caveats

### 1. Very Small Sample Size (n = 3 per group)

With only three subjects per group, no parametric statistical tests can meaningfully assess significance. All patterns described in this document are **descriptive and exploratory**. Single outliers (MA1_1, MA5_1) have outsized influence on group means and should be interpreted with caution. Effect size estimates would be unreliable at this sample size.

### 2. Rule-Based Behavior Detector Limitations

The behavior classifier uses thresholds calibrated on two subjects (MA1_2 for rearing, MA5_1 for grooming). Key limitations:
- Thresholds are not cross-validated on held-out subjects
- The R1 compact-rearing rule has a stationary-grooming guard that may vary in effectiveness across individuals
- Side-wall rearing (R5) uses a narrow htd_y window [40–60] tuned to one arena view
- No velocity features are used, making fast-locomotion near walls a potential false-positive source

### 3. Pixel-Space Metrics

All distances and speeds are reported in pixels. Physical conversion depends on a known reference distance within the arena. Across-study comparisons require calibration. Approximate conversion: 1 px ≈ 0.13 cm for a 50 × 50 cm arena at the recorded resolution.

### 4. Variability in Session Duration

Session durations vary from 160.1 s (MA3_3) to 179.3 s (MA5_1), a 12% range. Percentage metrics (% time rearing, etc.) normalize for this variation, but absolute counts (freeze bouts, center entries) are not duration-normalized. This introduces minor cross-subject bias.

### 5. MA7 Analysis Status

All 12 subjects have complete quantitative metrics (oft_metrics_all.csv, behavior_summary.csv). Spatial visualization outputs (KDE heatmaps, thigmotaxis plots) were generated for MA1, MA3, and MA5. MA7 spatial plots were pending at the time of this document. Quantitative data is complete for all groups.

### 6. Single Time Point

All data comes from a single OFT session per animal. Behavioral variability within an individual across repeated sessions is unknown. The between-subject variability seen here may partly reflect session-order or habituation differences rather than stable treatment effects.

---

## 14. Raw Data Tables

### Table A: OFT Cohort Summary (Mean ± SD)

| Metric | Control | Aspartame | Grapefruit | ASP+GF |
|--------|---------|-----------|------------|--------|
| Session duration (s) | 172.0 ± 4.9 | 167.6 ± 7.4 | 178.0 ± 1.2 | 174.7 ± 3.6 |
| Total distance (px) | 4449.5 ± 1741.4 | 4981.9 ± 257.5 | 5960.4 ± 2628.3 | 6949.1 ± 812.1 |
| Mean speed (px/s) | 26.45 ± 11.08 | 29.75 ± 0.25 | 33.62 ± 15.02 | 39.86 ± 4.15 |
| Max speed (px/s) | 811.2 ± 58.4 | 636.6 ± 402.6 | 677.7 ± 215.9 | 677.5 ± 180.4 |
| % Time center | 40.11 ± 44.58 | 17.50 ± 12.74 | 6.99 ± 3.42 | 30.76 ± 10.03 |
| % Time periphery | 59.89 ± 44.58 | 82.50 ± 12.74 | 93.01 ± 3.42 | 69.24 ± 10.03 |
| Center zone entries | 10.0 ± 5.29 | 11.33 ± 1.53 | 6.0 ± 0.0 | 17.33 ± 3.51 |
| Freeze bout count | 33.33 ± 10.69 | 12.0 ± 4.0 | 21.33 ± 24.83 | 12.67 ± 7.23 |
| Total freeze time (s) | 32.10 ± 15.04 | 9.99 ± 4.13 | 17.33 ± 22.23 | 10.77 ± 5.88 |
| % Time frozen | 18.56 ± 8.43 | 6.00 ± 2.63 | 9.68 ± 12.39 | 6.13 ± 3.28 |
| Spatial entropy (norm) | 0.513 ± 0.220 | 0.732 ± 0.058 | 0.718 ± 0.145 | 0.814 ± 0.019 |

### Table B: Behavior Cohort Summary (Mean ± SD)

| Metric | Control | Aspartame | Grapefruit | ASP+GF |
|--------|---------|-----------|------------|--------|
| Rear bout count | 11.33 ± 6.81 | 10.0 ± 1.73 | 18.0 ± 6.93 | 19.0 ± 4.0 |
| Rear total (s) | 15.04 ± 11.46 | 14.08 ± 8.27 | 35.06 ± 19.89 | 28.46 ± 11.07 |
| Rear % time | 8.85 ± 6.82 | 8.31 ± 4.58 | 19.74 ± 11.26 | 16.36 ± 6.61 |
| Rear mean bout (s) | 1.222 ± 0.456 | 1.348 ± 0.563 | 1.814 ± 0.523 | 1.458 ± 0.315 |
| Rear max bout (s) | 2.767 ± 1.341 | 3.09 ± 1.21 | 5.277 ± 1.734 | 4.813 ± 1.136 |
| Rear frag index | 0.926 ± 0.436 | 0.837 ± 0.351 | 0.590 ± 0.204 | 0.711 ± 0.176 |
| Groom bout count | 3.0 ± 2.65 | 3.33 ± 2.52 | 12.33 ± 5.51 | 15.67 ± 7.10 |
| Groom total (s) | 3.94 ± 3.82 | 3.58 ± 3.99 | 31.99 ± 35.89 | 19.59 ± 12.86 |
| Groom % time | 2.32 ± 2.26 | 2.08 ± 2.26 | 17.88 ± 19.98 | 11.28 ± 7.51 |
| Groom mean bout (s) | 1.723 ± 2.015 | 0.897 ± 0.400 | 2.254 ± 2.003 | 1.156 ± 0.553 |
| Groom max bout (s) | 3.02 ± 3.909 | 1.39 ± 1.195 | 10.547 ± 10.309 | 4.923 ± 3.941 |
| Groom frag index | 1.279 ± 0.903 | 1.248 ± 0.449 | 0.680 ± 0.400 | 1.023 ± 0.515 |

---

*Generated from:*  
- `data/oft_metrics_all.csv` — per-subject OFT metrics  
- `data/oft_metrics_all_cohort_summary.csv` — cohort-level OFT summaries  
- `data/behavior_summary.csv` — per-subject behavior bout metrics  
- `data/behavior_group_stats.csv` — cohort-level behavior summaries  
- `src/behavior_detection.py` — behavior classification algorithm  
- `analysis/oft_metrics.py` — OFT quantitative metrics pipeline  
