# Stage 07 — Psychological Analysis & Reporting
## Plan & Roadmap

---

## Objective

Synthesize all pipeline outputs into a structured analysis report: behavioral profiles per rat, group comparisons, model performance, and psychological state interpretations ready for thesis writing.

---

## Status

- [ ] Not started — requires Stages 03–06 outputs

---

## Tasks

### 7.1 — Per-Rat Behavioral Profiles

For each rat (MA1, MA3, MA5, MA7), produce a one-page summary:

- [ ] OFT occupancy heatmap (3 sessions side-by-side)
- [ ] OFT metric trends across sessions (line plots)
- [ ] T-maze turn bias over sessions
- [ ] T-maze path efficiency over sessions
- [ ] Predicted psychological state labels from Stage 06

```
reports/profiles/MA1_profile.pdf
reports/profiles/MA3_profile.pdf
...
```

### 7.2 — Group Comparison Statistics

- [ ] Mann-Whitney U test for each feature between condition groups
- [ ] Bonferroni correction for multiple comparisons
- [ ] Report effect sizes (Cohen's d or rank-biserial r)
- [ ] Produce table: feature × p-value × effect size × direction

```python
from scipy.stats import mannwhitneyu

for col in feature_cols:
    group_a = features[features["condition"] == "control"][col]
    group_b = features[features["condition"] == "stressed"][col]
    stat, p = mannwhitneyu(group_a, group_b, alternative="two-sided")
    ...
```

### 7.3 — Cross-Arena Correlation Matrix

- [ ] Compute Pearson / Spearman correlations between all OFT and T-maze features
- [ ] Generate annotated heatmap (seaborn `clustermap`)
- [ ] Highlight significant correlations (p < 0.05 after correction)
- [ ] Key hypothesis check: `oft_peripheral_time ↔ tmaze_decision_latency`

```python
import seaborn as sns
corr = features[feature_cols].corr(method="spearman")
sns.clustermap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
```

### 7.4 — Heatmap Gallery

- [ ] Per-session occupancy heatmaps for all rats (OFT + T-maze)
- [ ] Group-average heatmaps per condition
- [ ] Side-by-side comparison: session 1 vs session 3 (learning effect)
- [ ] Export gallery to `reports/figures/heatmaps/`

### 7.5 — Learning Curve Analysis

- [ ] Plot each behavioral metric across sessions 1→2→3 per rat
- [ ] Overlay individual rats + group mean ± SD
- [ ] Mark significant within-subject changes (Wilcoxon signed-rank)

### 7.6 — Decision Bias Maps

- [ ] For each rat: bar chart of L/R choice per session
- [ ] Group-level: proportion of left/right choosers per condition
- [ ] Test for lateralization bias (binomial test: p(left) ≠ 0.5)

### 7.7 — Model Performance Summary

- [ ] Table: model × label × accuracy × F1 × AUC
- [ ] SHAP summary plot for best model
- [ ] Confusion matrices per label (TP/FP/FN/TN with rat IDs)

### 7.8 — Thesis Report Structure

Generate supporting material organized for thesis integration:

```
reports/
├── figures/
│   ├── heatmaps/
│   │   ├── oft/              # Open field heatmaps
│   │   └── tmaze/            # T-maze heatmaps
│   ├── paths/                # Trajectory plots
│   ├── statistics/           # Group comparison figures
│   ├── learning_curves/      # Session-over-session trends
│   └── model/                # SHAP, confusion matrices, ROC curves
├── tables/
│   ├── oft_metrics_summary.csv
│   ├── tmaze_metrics_summary.csv
│   ├── group_comparisons.csv
│   └── model_comparison.csv
└── profiles/
    ├── MA1_profile.pdf
    ├── MA3_profile.pdf
    ├── MA5_profile.pdf
    └── MA7_profile.pdf
```

**Report sections:**

1. **Experiment Summary** — dataset, conditions, arena types
2. **Group Demographics** — rat IDs, conditions, session counts
3. **OFT Heatmap Gallery** — occupancy, velocity per group/session
4. **T-Maze Heatmap & Path Analysis** — occupancy, path lines, zone dwell
5. **Cross-Arena Correlations** — OFT ↔ T-maze feature relationships
6. **Feature Distributions & Statistics** — box plots, group comparisons
7. **Model Performance Report** — accuracy, F1, SHAP
8. **Psychological State Classifications** — predicted labels per rat
9. **Conclusions** — behavioral interpretation, limitations, future work

### 7.9 — Notebook Finalization

- [ ] `01_dlc_analysis.ipynb` — DLC QC and tracking visualization
- [ ] `02_heatmap_generation.ipynb` — all heatmap outputs with commentary
- [ ] `03_path_analysis.ipynb` — T-maze metrics walkthrough
- [ ] `04_feature_engineering.ipynb` — feature derivation and correlation analysis
- [ ] `05_model_training.ipynb` — model training, evaluation, SHAP interpretation

---

## Acceptance Criteria

- Per-rat behavioral profiles generated for all 4 rats
- Group comparison table with p-values and effect sizes
- Cross-arena correlation matrix with at least 3 significant cross-arena features
- Model performance tables for at least 2 models × 2 labels
- All figures exported at ≥ 300 DPI for print quality

---

## Output Files

| Path | Description |
|------|-------------|
| `reports/figures/` | All publication-quality figures |
| `reports/tables/` | Summary statistics tables |
| `reports/profiles/` | Per-rat behavioral profile PDFs |
| `notebooks/*.ipynb` | Final analysis notebooks |

---

## Notes

- All analysis should be run blinded to condition labels during feature extraction
- Model psychological state labels are behavioral proxies — expert validation required before clinical interpretation
- Correlations with n=12 have low statistical power — treat as hypothesis-generating, not confirmatory
