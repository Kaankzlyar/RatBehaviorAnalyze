# Final Report — Open-Field Behavioral Analysis & Inference Roadmap

**Project:** Rat Behavioral Neuroscience Thesis
**Compiled:** 2026-05-03
**Scope:** Open-Field Test (OFT) only — T-maze arm postponed to an unscheduled future phase.
**Cohorts analysed:** MA1 (Control), MA3 (Aspartame), MA5 (Grapefruit), MA7 (Aspartame + Grapefruit).
**N:** 3 subjects per group, 12 sessions total.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [What is finished vs. postponed](#2-what-is-finished-vs-postponed)
3. [Cross-cohort findings (synthesis)](#3-cross-cohort-findings-synthesis)
4. [Validation against the professor's 2017 MATLAB pipeline](#4-validation-against-the-professors-2017-matlab-pipeline)
5. [Limitations of the current dataset](#5-limitations-of-the-current-dataset)
6. [Inference goal — what we want the model to do](#6-inference-goal--what-we-want-the-model-to-do)
7. [Defining the targets (anxiety, rearing, grooming)](#7-defining-the-targets-anxiety-rearing-grooming)
8. [Why an end-to-end heatmap CNN is *not* the right starting point](#8-why-an-end-to-end-heatmap-cnn-is-not-the-right-starting-point)
9. [Recommended modelling stack](#9-recommended-modelling-stack)
10. [Validation strategy (LOSO-CV)](#10-validation-strategy-loso-cv)
11. [Data augmentation — turning 12 sessions into hundreds of samples](#11-data-augmentation--turning-12-sessions-into-hundreds-of-samples)
12. [Concrete model menu and when to pick which](#12-concrete-model-menu-and-when-to-pick-which)
13. [Interpretability — SHAP, saliency, "why anxious?"](#13-interpretability--shap-saliency-why-anxious)
14. [Honest expectations and stop-rules](#14-honest-expectations-and-stop-rules)
15. [Next steps checklist](#15-next-steps-checklist)

---

## 1. Executive summary

After running the full DLC → behavior-detection → spatial-analysis pipeline across all four cohorts, the dataset now carries:

- **12 raw DLC pose CSVs** (one per subject, 9 body parts × 30 fps × ~170 s).
- **20+ engineered metrics per subject** in `data/oft_metrics_all.csv` and `data/behavior_summary.csv` (locomotion, thigmotaxis, freezing, rearing, grooming, spatial entropy, fragmentation indices).
- **Heatmap and trajectory images per subject** under `data/DLCfiltered/<subject>/` (KDE, histogram, per-bodypart panels, orbit grid, thigmotaxis trajectory, behavior timeline).
- **Cross-validation** of mean speed against the professor's 2017 HSV-blob MATLAB pipeline (`data/kutu_validation_summary.csv`) — DLC reads ~10–20 % faster, but the bias is uniform across cohorts (see §4).

Three behavioural patterns survive the n = 3 caveat:

1. **Locomotion scales monotonically** with treatment complexity: Control < Aspartame < Grapefruit < Aspartame+Grapefruit.
2. **Grapefruit specifically and dramatically elevates grooming** (≈ 8× Control), and this is the most diagnostically clean signal in the dataset.
3. **Center avoidance is strongest in Grapefruit** (93 % periphery, all three rats made *exactly* 6 center entries — a zero-variance result that is biologically informative, not a data error).

The next phase is **inference** — building models that, given a new rat's session, output behavioural state estimates (anxiety level, rearing profile, grooming profile) with calibrated confidence and an explanation of *which* features drove the call. Sections 6–14 lay out how to do this defensibly at n = 12.

---

## 2. What is finished vs. postponed

| Stage | Status |
|-------|--------|
| 01 — Video preprocessing / QC | done |
| 02 — DeepLabCut pose estimation | done (all 12 subjects filtered) |
| 02B — Rule-based behaviour detection | done; calibrated on MA1_2 (rearing) and MA5_1 (grooming), applied uniformly |
| 03A — OFT spatial analysis | **done for all 4 cohorts** (MA1 / MA3 / MA5 / MA7) |
| 03B — T-maze heatmaps | **postponed indefinitely** — no current target date |
| 04 — Path analysis (T-maze-driven) | postponed with 03B |
| 05 — Feature engineering for ML | next up — see §11 |
| 06 — ML inference (anxiety + rearing + grooming) | planned — see §6–§14 |
| 07 — Cross-cohort thesis report | this document is the open-field half |

> **T-maze status:** the 12 T-maze recordings are not abandoned — they remain in the raw video set — but DLC labelling, behaviour detection, and zone analysis for the T-maze arm are deferred to a later thesis phase. The thesis chapter for the open-field arm should be writeable as a self-contained piece without waiting for T-maze data.

---

## 3. Cross-cohort findings (synthesis)

The detailed group-by-group analysis lives in [`behavior_comparison.md`](behavior_comparison.md). This section pulls out only the *patterns* — the things the inference model should ultimately recover.

### Pattern A — Locomotion gradient (additive activation)

```
Total distance (px):  Control 4 449  <  Aspartame 4 982  <  Grapefruit 5 960  <  ASP+GF 6 949
Mean speed (px/s):    Control 26.5   <  Aspartame 29.8   <  Grapefruit 33.6   <  ASP+GF 39.9
```

The combined treatment is roughly 50 % faster than control. Suggestive of additive CNS stimulation — possibly a CYP3A4-mediated grapefruit + aspartame interaction.

### Pattern B — Inverse freezing/locomotion axis

```
% Time frozen:        Control 18.6 %  >  Grapefruit 9.7 %  >  ASP+GF 6.1 %  ≈  Aspartame 6.0 %
```

Animals that move more freeze less. Treatment groups show roughly 1/3 the control freezing time, consistent with all three treatments reducing novelty-induced fear.

### Pattern C — Grapefruit-specific grooming surge

```
% Time grooming:      Control 2.3 %  ≈  Aspartame 2.1 %  <<  Grapefruit 17.9 %  >  ASP+GF 11.3 %
```

This is the single sharpest treatment-versus-control signal in the dataset. Grapefruit roughly multiplies grooming by 8×; aspartame appears to *attenuate* the surge when added on top.

### Pattern D — Sustained vs. fragmented behaviour

| Metric | Control | Aspartame | Grapefruit | ASP+GF |
|--------|---------|-----------|------------|--------|
| Rearing fragmentation index | 0.926 | 0.837 | **0.590** | 0.711 |
| Grooming fragmentation index | 1.279 | 1.248 | **0.680** | 1.023 |

Lower fragmentation = longer, more sustained bouts. Grapefruit organises rearing and grooming into long episodes; adding aspartame breaks them up again.

### Pattern E — Thigmotaxis ≠ freezing

Classical anxiety theory predicts wall-hugging and freezing co-occur. They do not, here:

- Control freezes most but is variable in thigmotaxis (driven by MA1_1, an exploration outlier).
- Grapefruit shows the most extreme thigmotaxis (93 % periphery, 6 ± 0 center entries) but only intermediate freezing.
- ASP+GF shows the lowest freezing (6.1 %) and the most center entries (17.3) but only moderate dwell time in the center — a "transit through center, don't linger" profile.

This dissociation matters for §7: an anxiety target should weight multiple metrics, not collapse to one.

---

## 4. Validation against the professor's 2017 MATLAB pipeline

`data/kutu_validation_summary.csv` compares per-frame instantaneous speed from our DLC pose to the professor's HSV-blob centroid (`Kare/<cohort>/*_res.mat`):

- **Pearson r ≥ 0.78** on 9 of 12 subjects (median 0.85).
- **Outlier r:** MA3_1 (0.45), MA3_3 (0.54) — worth eyeballing the trajectories before publishing.
- **Speed ratio kutu/dlc = 0.79–0.90** uniformly across all cohorts: DLC consistently reads 10–20 % faster than the blob centroid.

That bias is **systematic and uniform**, not random — almost certainly because DLC tracks individual body parts (and a smoothed centroid through them) while the HSV blob loses fine motion under occlusion. The bias does *not* invalidate cross-group comparisons, since every cohort is biased the same way, but it should be flagged in the thesis methods section.

---

## 5. Limitations of the current dataset

These constrain every inferential and modelling decision in §§ 6–14.

1. **n = 3 per group** — exploratory data only. No parametric ANOVA is meaningfully powered. Single subjects (MA1_1, MA5_1) drive group-level statistics. (Full statistical-strategy discussion: `behavior_comparison.md` §3.)
2. **Treatment label is not a behavioural label.** A "Control" rat (MA1_1) can be more exploratory than a "treated" one. Models trained on the *treatment* label answer "did this rat get a pill?" not "is this rat anxious?". §7 derives behavioural targets independent of treatment.
3. **Behaviour detector calibrated on 2 subjects.** Rearing thresholds tuned on MA1_2, grooming on MA5_1. No held-out cross-subject validation yet. False-positive risk in subjects with unusual posture.
4. **Pixel-space metrics, not centimetres.** Cross-arena generalisation needs calibration.
5. **Single session per animal.** Within-individual reliability across repeated sessions is unknown.
6. **MA1_1 and MA5_1 are statistical leverage points.** Any model trained on raw subject-level summaries will be unduly influenced by these two. Robust losses (Huber) or window-based augmentation (§11) reduce their pull.

---

## 6. Inference goal — what we want the model to do

The goal stated by the user, restated formally:

> Given a new open-field session for a single rat (DLC-tracked, with all the heatmaps and engineered metrics our pipeline produces), output:
> - **Anxiety state** — low / moderate / high (and a continuous score)
> - **Rearing profile** — low / moderate / high *and* fragmented / sustained
> - **Grooming profile** — low / moderate / high *and* fragmented / sustained
>
> with an explanation of which features drove each call.

Three sub-models. They share the same input (one session) but predict independent outputs. Each output also has a continuous version (regression) and a categorical version (classification) — both are useful: the score is what compares animals, the bin is what reads in the thesis.

---

## 7. Defining the targets (anxiety, rearing, grooming)

The dataset has *measurements*, not labels. The first job is to **construct** behavioural labels from the measurements, *then* train a model to predict those labels from a different subset of the inputs (for example: predict anxiety from heatmaps, given that the label was constructed from time-series metrics).

### 7.1 — Anxiety score (composite z-score)

Open-field anxiety is canonically a multi-metric construct. We define:

```
anxiety_score = + z(pct_time_periphery)
                + z(pct_time_freeze)
                − z(center_zone_entries)
                − z(pct_time_center)
```

Each term is z-scored across the 12 subjects, then summed. Higher = more anxious. Bin thresholds for the categorical label:

- `low`      : score < −0.5
- `moderate` : −0.5 ≤ score ≤ +0.5
- `high`     : score > +0.5

Rationale: at n = 12, three roughly-balanced bins give 3–5 subjects per class, which is the floor for any classifier.

### 7.2 — Rearing profile (two axes)

- **Volume axis:** `rear_pct_time` → low (< 7 %), moderate (7–15 %), high (> 15 %).
- **Structure axis:** `rear_frag_idx` → sustained (< 0.7), mixed (0.7–1.0), fragmented (> 1.0).

Either can be the supervised target; both can be modelled jointly as a 2-D label.

### 7.3 — Grooming profile (two axes)

Same shape as rearing:

- **Volume axis:** `groom_pct_time` → low (< 4 %), moderate (4–10 %), high (> 10 %).
- **Structure axis:** `groom_frag_idx` → sustained (< 0.8), mixed (0.8–1.2), fragmented (> 1.2).

### 7.4 — Why this two-stage approach

The labels are built from the same measurements they will be predicted from. That is fine **as long as** the prediction inputs and the label-generation inputs are kept separate at evaluation time. Concretely:

| Inputs the model sees | Held back to construct labels |
|-----------------------|-------------------------------|
| Heatmap images, trajectory images, raw pose features, per-frame velocity | The aggregated metrics in `oft_metrics_all.csv` used to compute z-scores |

If you train a model on `pct_time_periphery` and then evaluate it against an anxiety label built from `pct_time_periphery`, you have label leakage. §11 enforces the separation by feature group.

---

## 8. Why an end-to-end heatmap CNN is *not* the right starting point

The instinct is "give the model the heatmap, let it learn anxiety." This will not work at n = 12. Reasons:

1. **Sample count.** A from-scratch CNN needs 10³–10⁴ labelled images. We have 12 heatmaps per output type. Even a small ResNet has ~10⁶ parameters.
2. **Class balance.** 4–5 samples per anxiety class is not a class — it is one good or bad day for one rat.
3. **Distribution shift.** Heatmaps differ in arena tilt, lighting, camera framing. A CNN with no inductive bias will latch onto these confounds (cohort-specific arena pixels can become the strongest "feature").
4. **Interpretability.** A thesis defended on "the CNN said so" with no SHAP/saliency story is a hard sell to a behavioural-neuroscience committee.

CNNs return as a useful tool **after** (a) heavy augmentation (§11), (b) using a pretrained backbone, and (c) fusing image features with tabular features (§12). They are not the place to start.

---

## 9. Recommended modelling stack

Build in this order. Each layer is shippable on its own; later layers strictly improve on earlier ones.

```
Layer 1 — Tabular ML on engineered features  ← START HERE
            ├── Logistic Regression with L1
            ├── Random Forest
            └── XGBoost / Gradient Boosting

Layer 2 — Window-augmented tabular ML        ← +data through windowing
            └── Same models, evaluated per window

Layer 3 — Image embeddings (frozen pretrained CNN)
            └── ResNet18/50 ImageNet → 512-d vector → concat with tabular → small MLP

Layer 4 — Multi-modal fusion (optional, if Layer 3 helps)
            └── Tabular + image embedding + sequence (LSTM on velocity trace)

Layer 5 — End-to-end CNN fine-tuning (only when n grows by 5×+)
```

The shippable thesis result lives at **Layer 1 + Layer 2**. Layer 3 is the upgrade path. Layers 4–5 are deferred until cohort size grows.

---

## 10. Validation strategy (LOSO-CV)

**The single most important rule for this dataset:**

> **Hold out by *subject*, never by frame, window, or session.** A frame from MA5_1 in train and another frame from MA5_1 in test is a guaranteed leak — the model will learn the rat's gait, not the treatment.

### 10.1 — Leave-One-Subject-Out (LOSO) cross-validation

12 subjects → 12 folds. Each fold trains on 11 subjects, tests on 1. Report per-fold prediction + the mean ± SD across folds. This gives 12 honest test predictions, one per rat.

```python
from sklearn.model_selection import LeaveOneGroupOut

X = features_array            # (n_samples, n_features)
y = anxiety_labels            # (n_samples,)
groups = subject_id_array     # (n_samples,) — same id for all rows of one rat

logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    ...
```

### 10.2 — When windowing is used (§11)

If each session is split into N windows, **the group key remains the subject ID**, not the window ID. A 30-window dataset for MA5_1 must all be in either train *or* test, never split.

### 10.3 — Reporting metrics that mean something at this n

Per-fold accuracy is noisy at n = 1 test point. Report:

- **Mean accuracy and macro-F1 across 12 folds** with their SD.
- **Confusion matrix aggregated across folds** (12 predictions × 1 true label each).
- **Per-subject prediction probabilities** as a table — readers should see exactly which rats the model got right/wrong.
- **Effect-size-style claims:** "the model predicts the high-anxiety class correctly on N of K such rats" rather than "AUC = 0.87 ± nothing."

---

## 11. Data augmentation — turning 12 sessions into hundreds of samples

The single biggest leverage point. Two complementary axes:

### 11.1 — Temporal windowing

Each ~170 s session at 30 fps is ~5 100 frames. Slice into non-overlapping windows; recompute features per window:

| Window length | Windows per session | Total samples (12 subjects) |
|--------------:|--------------------:|----------------------------:|
| 60 s          | 2–3                 | 24–36 |
| 30 s          | 5–6                 | 60–72 |
| 15 s          | 11                  | 132 |
| 10 s          | 17                  | 204 |

30 s is a defensible default — long enough to estimate thigmotaxis stably, short enough to multiply the dataset 6×. The label per window can be either:

- the global session label (cheap, but smooths over within-session state changes), or
- a per-window label re-derived from that window's own metrics (richer, but requires the same z-score machinery as §7).

**LOSO is still by subject, not window** (§10.2).

### 11.2 — Image augmentation (for Layer 3 onward)

For heatmaps and trajectory plots, the open-field arena has approximate four-fold symmetry — augmentation that preserves the inner/outer zone structure is allowed:

- horizontal/vertical flips (arena is roughly square)
- 90° / 180° / 270° rotations
- small random translations (≤ 5 % of arena width)
- small brightness / contrast jitter on the heatmap colour scale

Avoid: large rotations off the cardinal angles (would interpolate the inner-zone geometry away), and any augmentation that crops out the periphery (deletes the thigmotaxis signal).

### 11.3 — Synthetic minority oversampling (SMOTE)

If, after windowing, one anxiety class is underrepresented (e.g., 80 windows "moderate" vs 20 "high"), SMOTE on the *tabular* feature vectors is acceptable — but **only inside the training fold**, never on the held-out test subject's windows. Apply with `imblearn.pipeline.Pipeline` so the held-out fold is never touched.

---

## 12. Concrete model menu and when to pick which

For each behavioural target (anxiety, rearing-volume, rearing-structure, grooming-volume, grooming-structure), evaluate the following:

| # | Model | Library | Inputs | Why pick it |
|---|-------|---------|--------|-------------|
| 1 | **Logistic Regression (L1)** | scikit-learn | tabular metrics | Interpretable baseline. L1 selects features → tells you which 3–5 metrics matter. |
| 2 | **Random Forest** | scikit-learn | tabular metrics | Handles non-linearity, gives feature importance, no scaling needed. |
| 3 | **XGBoost / Gradient Boosting** | xgboost | tabular metrics | Best small-tabular-data performer. Strong with mixed feature types. SHAP-friendly. |
| 4 | **SVM (RBF)** | scikit-learn | tabular metrics | Strong on small n with kernels. Less interpretable. |
| 5 | **kNN** | scikit-learn | tabular metrics | Naive but useful as sanity check; failing kNN means features are not separable. |
| 6 | **Pretrained CNN feature extractor + classifier** | torchvision (ResNet18) → sklearn | heatmap image | Layer 3. Frozen backbone, train only the head. Robust at small n. |
| 7 | **Multi-modal MLP** | PyTorch | tabular + CNN embedding | Layer 4. Concat features, 2 hidden layers, dropout. Final upgrade. |
| 8 | **LSTM on per-frame velocity trace** | PyTorch | (frames, velocity, body-axis angle) | Captures temporal structure freezes/bouts compress out. |

**Default winner at this n: XGBoost on tabular features (model #3).** Logistic regression L1 (model #1) is the obligatory sanity check and the most readable result — even if XGBoost wins on accuracy, the L1 model is what you put in the thesis methods section to explain *what the classifier is using*.

For the rearing- and grooming-structure targets specifically, fragmentation index is so dominant that even a one-feature decision stump should perform near-ceiling — that is fine to report and important to acknowledge (it tells you the structure target is essentially predetermined by one engineered metric).

---

## 13. Interpretability — SHAP, saliency, "why anxious?"

A thesis-grade prediction must explain itself. Two layers of explanation:

### 13.1 — Tabular models (Layers 1–2)

- **SHAP values** (`shap.TreeExplainer` for RF/XGB, `shap.LinearExplainer` for logistic) — global feature-importance ranking and per-prediction breakdowns.
- **Top-5 feature plot per target** in the thesis figure.
- **Per-rat narrative**: for each predicted "high anxiety" call, list the three SHAP-largest contributing features. Example: *"MA5_1 was classed high-anxiety because pct_time_freeze (+1.6), groom_pct (+0.9), pct_time_periphery (+0.7)."* This is what a behavioural-neuroscience reviewer wants to see.

### 13.2 — Image models (Layer 3+)

- **Grad-CAM / saliency maps** over heatmap input — show *which pixels* drove the prediction. Expected: high-anxiety predictions should highlight the periphery; low-anxiety should highlight the inner zone.
- If saliency maps fail this sanity check (e.g., model focuses on arena corners that are camera artefacts), the image model is learning a confound and should be downweighted.

### 13.3 — Sanity checks the thesis must pass

- A model that predicts anxiety from `pct_time_freeze` alone is **not interesting** — it is repeating its own label-generation input. Show that the model still works when freeze and periphery features are *removed* from the input. If it does, that is genuine inference; if it doesn't, the model is a thin tautology.
- Predict treatment group from features → if the cross-cohort signal is real, this should be possible at well above chance (1/4). If a 4-class classifier hits ~75 %+ macro-F1 with LOSO, the cohort-level patterns in §3 are confirmed model-side.

---

## 14. Honest expectations and stop-rules

What a defensible thesis result actually looks like at n = 12, after windowing to ~60–200 samples:

| Target | Realistic LOSO macro-F1 | Floor ("publish anyway") | Ceiling ("almost certainly leakage") |
|--------|------------------------:|-------------------------:|-------------------------------------:|
| Anxiety (3-class) | 0.55–0.70 | 0.40 | > 0.90 |
| Rearing volume (3-class) | 0.50–0.65 | 0.35 | > 0.90 |
| Grooming volume (3-class) | 0.60–0.80 (Grapefruit signal is strong) | 0.45 | > 0.95 |
| Treatment (4-class) | 0.50–0.70 | 0.30 | > 0.95 |

If a model exceeds the **ceiling** column, treat it as a leakage hypothesis until proven otherwise. Re-check:

- Are train and test sharing a subject's windows?
- Did label-generation features sneak into the input set?
- Is augmentation crossing the fold boundary?

If a model falls below the **floor** column on every architecture: the labels are likely not separable from this feature set at this n, and the honest write-up is "the n = 12 OFT data does not support reliable supervised classification of [target]; with larger cohorts the patterns observed in §3 should be revisited."

That null result is also a thesis result, and it is far better than overclaiming.

---

## 15. Next steps checklist

1. **Lock the targets.** Compute the anxiety / rearing / grooming labels per §7 and commit them to `data/labels.csv`.
2. **Feature table.** Consolidate `oft_metrics_all.csv` + `behavior_summary.csv` into a single `data/features/features.csv` indexed by `subject_id`. Keep label-generation features separate from prediction features.
3. **Windowing module.** Add a script that takes a DLC-filtered CSV + window length and emits one feature row per window, preserving subject_id.
4. **Layer 1 baseline.** Train logistic-L1, RF, XGBoost on session-level features, LOSO-CV, dump SHAP plots. This is the minimum-viable thesis result.
5. **Layer 2 with windowing.** Re-train on windowed features, same LOSO. Compare macro-F1 to Layer 1.
6. **Layer 3 image branch.** Run a frozen ResNet18 over each subject's KDE heatmap and bodypart-heatmap PNG; concat the 512-d embedding with the tabular features and re-train.
7. **Sanity-check experiments** from §13.3 (drop-the-leakage-feature, predict-the-cohort).
8. **Write-up.** Per-target table of LOSO accuracy, confusion matrix, SHAP top-5 plot, per-subject probabilities, plus the dropped-feature sanity check.

Files to produce, in order:

```
data/labels.csv                      # §7
data/features/features.csv           # §15 step 2
src/feature_windowing.py             # §15 step 3
src/train_baseline.py                # Layer 1
src/train_windowed.py                # Layer 2
src/extract_image_embeddings.py      # Layer 3
src/train_multimodal.py              # Layer 3 fusion
reports/loso_results.csv             # all models × all targets
reports/figures/shap_*.png           # one per target
reports/figures/confusion_*.png      # one per target
docs/inference_results.md            # final thesis-ready writeup
```

---

## Sources used to build this document

- `data/oft_metrics_all.csv` — per-subject OFT metrics
- `data/oft_metrics_all_cohort_summary.csv` — cohort means / SDs
- `data/behavior_summary.csv` — per-subject behavior bouts
- `data/behavior_group_stats.csv` — cohort behavior summaries
- `data/kutu_validation_summary.csv` — DLC vs MATLAB-blob speed cross-validation
- `data/DLCfiltered/<subject>/*_heatmap_kde.png` — heatmaps available as model input
- `docs/behavior_comparison.md` — per-group descriptive analysis (companion document)
- `docs/plans/STAGE_06_MODEL_TRAINING.md` — the existing Turkish-language modelling plan, which this document refines and replaces for the open-field arm
