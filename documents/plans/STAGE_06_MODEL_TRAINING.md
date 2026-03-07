# Stage 06 — Model Training
## Plan & Roadmap

---

## Objective

Train and evaluate supervised ML models to classify psychological states (anxiety level, cognitive flexibility, spatial memory) from the behavioral feature dataset built in Stage 05.

---

## Status

- [ ] Not started — requires Stage 05 feature dataset

---

## Dataset Constraints

| Property | Value |
|----------|-------|
| Total samples | 12 (4 rats × 3 sessions) |
| Features | ~20–30 behavioral metrics |
| Target labels | 2–3 binary/ordinal labels |
| Primary split risk | Data leakage if same rat in train and test |

> **Important:** Split by rat, not by session. Training and testing on different sessions of the same rat constitutes data leakage in this study.

---

## Tasks

### 6.1 — Data Preparation

- [ ] Load `data/features/features_normalized.csv` and `data/features/labels.csv`
- [ ] Implement **rat-based cross-validation** (leave-one-rat-out, LORO-CV)
  - 4 folds: each fold holds out one rat's 3 sessions for testing
- [ ] Check class balance per label — flag any class with < 25% representation

```python
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd

features = pd.read_csv("data/features/features_normalized.csv")
labels   = pd.read_csv("data/features/labels.csv")

groups   = features["rat_id"].values   # leave-one-rat-out
logo     = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### 6.2 — Baseline Models

Implement in `src/model.py`. Evaluate with LORO-CV.

| Model | Library | Notes |
|-------|---------|-------|
| Random Forest | `sklearn` | Baseline + feature importance |
| Gradient Boosting (XGBoost) | `xgboost` | Best for small tabular data |
| SVM (RBF kernel) | `sklearn` | Strong for small datasets |
| Logistic Regression | `sklearn` | Interpretable baseline |

- [ ] Train each model on all 4 LORO folds
- [ ] Report mean accuracy, F1, AUC-ROC across folds
- [ ] Save best model per label to `models/classifier/`

### 6.3 — Deep Learning Models (if dataset is augmented)

Given the small dataset (n=12), deep learning is only viable with data augmentation or if per-trial data is used (n ≈ 12 sessions × ~10 trials = ~120 rows).

| Model | Input | Use Case |
|-------|-------|---------|
| LSTM / GRU | Per-trial sequence per session | Temporal pattern in trial progression |
| CNN | Occupancy heatmap (numpy array) | Spatial classification from image |

- [ ] LSTM: use sequence of per-trial T-maze metrics as input (length = n_trials)
- [ ] CNN: use stacked occupancy heatmap arrays from `data/heatmaps/`

### 6.4 — Hyperparameter Tuning

- [ ] Use Optuna or `GridSearchCV` for top-performing models
- [ ] Tune inside CV loop (no test set contamination)
- [ ] Log all trials to `reports/hyperparam_search.csv`

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth    = trial.suggest_int("max_depth", 2, 10)
    ...
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    scores = cross_val_score(model, X_train, y_train, cv=logo, groups=groups_train)
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 6.5 — Evaluation

For each model × label combination:

- [ ] Accuracy
- [ ] F1-score (macro, weighted)
- [ ] AUC-ROC (binary labels)
- [ ] Confusion matrix
- [ ] SHAP feature importance (top 10 features per model)

```python
import shap

explainer  = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### 6.6 — Model Comparison Report

- [ ] Compile comparison table: all models × all labels × all metrics
- [ ] Identify best model per label
- [ ] Report which features are most predictive (SHAP top-5)
- [ ] Visualize learning curves (train vs val score over CV folds)

### 6.7 — Save Final Models

- [ ] Save best model per label using `joblib.dump()`
- [ ] Save feature names list (must match inference input order)
- [ ] Document model version and training date in `models/classifier/model_card.md`

---

## Evaluation Targets

| Metric | Minimum | Good |
|--------|---------|------|
| Accuracy | > 0.65 | > 0.80 |
| F1-score | > 0.60 | > 0.75 |
| AUC-ROC | > 0.70 | > 0.85 |

> Note: With n=12, high variance across folds is expected. Interpret trends, not point estimates.

---

## Output Files

| Path | Description |
|------|-------------|
| `models/classifier/rf_anxiety.pkl` | Best RF model for anxiety label |
| `models/classifier/xgb_anxiety.pkl` | Best XGBoost model |
| `models/classifier/scaler.pkl` | Feature scaler |
| `models/classifier/feature_names.json` | Feature order for inference |
| `reports/model_comparison.csv` | All model × metric results |
| `reports/figures/shap_*.png` | SHAP importance plots |
| `src/model.py` | Training and evaluation script |

---

## Next Step

→ **Stage 07:** `STAGE_07_REPORTING.md`
