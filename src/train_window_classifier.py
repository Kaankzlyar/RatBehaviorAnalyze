"""
Aşama 07 — Pencere-seviyesi Davranış Sınıflandırıcı

Inputs
------
    data/windows_labeled.parquet   (src/window_label_join.py)

Çıktılar
--------
    reports/window_classifier_metrics.csv      (model, cv, class, f1, support)
    reports/window_classifier_summary.csv      (model, cv, macro_f1, accuracy)
    reports/figures/open_field/window_cm_<cv>_<model>.png (confusion matrix)
    reports/figures/open_field/window_shap_<best>.png     (SHAP top-N feature)
    reports/window_shap_<best>.csv             (SHAP rank tablosu)
    models/window_classifier/<model>.pkl       (tüm veriyle eğitilmiş)

CV stratejileri
---------------
    GroupKFold(subject_id, n_splits=5)   — denek-bağımsız genelleme
    LeaveOneGroupOut(cohort)              — cohort-bağımsız genelleme
"""
from __future__ import annotations

import pathlib
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA     = ROOT / "data"
MODELS   = ROOT / "models" / "window_classifier"
REPORTS  = ROOT / "reports"
FIGS     = REPORTS / "figures" / "open_field"
for d in (MODELS, REPORTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

WIN_PATH = DATA / "windows_labeled.parquet"

META_COLS = ("cohort", "subject_id", "window_start", "window_end", "label")
RANDOM_STATE = 42


# ── Veri ──────────────────────────────────────────────────────────────────────

def load_labeled(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"{path} bulunamadı — önce src/window_label_join.py'yi çalıştır."
    )


def make_models(class_weight_map: dict[int, float]) -> dict:
    """Tek noktada model paneli; class_weight_map sklearn-style dict."""
    cw_array = np.array(
        [class_weight_map[k] for k in sorted(class_weight_map.keys())]
    )
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            class_weight=class_weight_map,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        ),
    }
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, num_leaves=31, max_depth=-1,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
        )
    return models, cw_array


def fit_and_evaluate(model, model_name, X, y, groups, splitter, cv_name,
                     class_names, sample_weight_full):
    """Run splitter; return per-fold concatenated y_true, y_pred."""
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        # XGBoost & LightGBM benefit from sample_weight even though
        # class_weight is set; for RF class_weight handles it directly.
        if model_name == "XGBoost":
            model.fit(X[tr], y[tr], sample_weight=sample_weight_full[tr])
        else:
            model.fit(X[tr], y[tr])
        y_pred_all.extend(model.predict(X[te]).tolist())
        y_true_all.extend(y[te].tolist())
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    return y_true, y_pred


def metrics_table(y_true, y_pred, class_names, model_name, cv_name):
    """Per-class F1 + support; macro F1 + accuracy (single row each)."""
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    rows = []
    for cls in class_names:
        rows.append({
            "model":   model_name,
            "cv":      cv_name,
            "class":   cls,
            "f1":      round(rep[cls]["f1-score"], 4),
            "precision": round(rep[cls]["precision"], 4),
            "recall":  round(rep[cls]["recall"], 4),
            "support": int(rep[cls]["support"]),
        })
    summary = {
        "model":      model_name,
        "cv":         cv_name,
        "f1_macro":   round(rep["macro avg"]["f1-score"], 4),
        "f1_weighted": round(rep["weighted avg"]["f1-score"], 4),
        "accuracy":   round(rep["accuracy"], 4),
        "n_samples":  int(rep["macro avg"]["support"]),
    }
    return rows, summary


def plot_confusion(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_top(model, X, feature_names, class_names, model_name, save_path):
    """SHAP TreeExplainer; top-15 global + per-class panel."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        sv_arr = np.stack(shap_vals, axis=-1)        # (n, p, k)
    else:
        sv_arr = shap_vals
        if sv_arr.ndim == 2:
            sv_arr = sv_arr[..., np.newaxis]
    mean_abs_per_class = np.abs(sv_arr).mean(axis=0)  # (p, k)
    global_imp         = mean_abs_per_class.mean(axis=1)

    TOP_N = min(15, len(feature_names))
    top_idx = np.argsort(global_imp)[::-1][:TOP_N]

    rank_rows = []
    for rank, idx in enumerate(top_idx, 1):
        row = {
            "rank":                 rank,
            "feature":              feature_names[idx],
            "global_mean_abs_shap": round(float(global_imp[idx]), 5),
        }
        for ci, cn in enumerate(class_names):
            row[f"shap_{cn}"] = round(float(mean_abs_per_class[idx, ci]), 5)
        rank_rows.append(row)

    n_cls = len(class_names)
    fig, axes = plt.subplots(1, n_cls + 1, figsize=(4 * (n_cls + 1), 6),
                             sharey=True)
    if n_cls + 1 == 1:
        axes = [axes]

    feats_top = [feature_names[i] for i in top_idx][::-1]
    ax = axes[0]
    vals_top = global_imp[top_idx][::-1]
    ax.barh(range(len(vals_top)), vals_top, color="#333333", alpha=0.85)
    ax.set_yticks(range(len(vals_top)))
    ax.set_yticklabels(feats_top, fontsize=8)
    ax.set_title("Genel (sınıf ortalaması)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Ort. |SHAP|", fontsize=9)

    palette = ["#9C27B0", "#4CAF50", "#FF9800", "#2196F3"]
    for ci, (cn, ax_c) in enumerate(zip(class_names, axes[1:])):
        vals = mean_abs_per_class[top_idx, ci][::-1]
        ax_c.barh(range(len(vals)), vals,
                  color=palette[ci % len(palette)], alpha=0.85)
        ax_c.set_yticks(range(len(vals)))
        ax_c.set_title(f"{cn}", fontsize=10, fontweight="bold",
                       color=palette[ci % len(palette)])
        ax_c.set_xlabel("Ort. |SHAP|", fontsize=9)

    fig.suptitle(f"{model_name} — Pencere SHAP Top-{TOP_N}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rank_rows


# ── Ana ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_labeled(WIN_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"[load] {len(df)} pencere × {len(feature_cols)} feature")
    print(f"[class] {df['label'].value_counts().to_dict()}")
    print(f"[subjects] {df['subject_id'].nunique()}  "
          f"[cohorts] {df['cohort'].nunique()}")

    # X imputed (mean), y encoded
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(df[feature_cols].values).astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    subjects = df["subject_id"].values
    cohorts  = df["cohort"].values

    # class weights
    cw_array = compute_class_weight(class_weight="balanced",
                                    classes=np.arange(n_classes), y=y)
    cw_map = {i: float(w) for i, w in enumerate(cw_array)}
    sample_weight = np.array([cw_map[label] for label in y])
    print(f"[class weight] {cw_map}")

    models, _ = make_models(cw_map)

    # CV stratejileri
    cv_strategies = [
        ("GroupKFold-subject",
         GroupKFold(n_splits=5),
         subjects),
        ("LOGOCV-cohort",
         LeaveOneGroupOut(),
         cohorts),
    ]

    per_class_rows: list[dict] = []
    summary_rows:   list[dict] = []
    best = {"model": None, "cv": None, "macro_f1": -1.0,
            "y_true": None, "y_pred": None}

    for cv_name, splitter, groups in cv_strategies:
        n_splits = (splitter.get_n_splits(X, y, groups)
                    if cv_name.startswith("LOGOCV")
                    else splitter.get_n_splits())
        print(f"\n{'='*60}\n[CV] {cv_name}  n_splits={n_splits}\n{'='*60}")
        for model_name, model in models.items():
            y_true, y_pred = fit_and_evaluate(
                model, model_name, X, y, groups, splitter, cv_name,
                class_names, sample_weight,
            )
            rows, summary = metrics_table(y_true, y_pred, class_names,
                                          model_name, cv_name)
            per_class_rows.extend(rows)
            summary_rows.append(summary)
            print(f"  {model_name:<14}  macro_f1={summary['f1_macro']:.3f}  "
                  f"acc={summary['accuracy']:.3f}  per-class F1={[r['f1'] for r in rows]}")

            cm_path = FIGS / f"window_cm_{cv_name}_{model_name}.png".replace(
                " ", "_")
            plot_confusion(
                y_true, y_pred, class_names,
                title=f"{model_name} — {cv_name}\nmacro F1={summary['f1_macro']:.2f}",
                save_path=cm_path,
            )

            # En iyi modeli takip et — LOGOCV (cohort-aware) önceliği var,
            # eşitlikte GroupKFold'u tercih.
            cv_priority = 2 if cv_name.startswith("LOGOCV") else 1
            best_priority = (2 if best["cv"] == "LOGOCV-cohort"
                             else 1 if best["cv"] is not None else 0)
            replace = (cv_priority > best_priority or
                       (cv_priority == best_priority
                        and summary["f1_macro"] > best["macro_f1"]))
            if replace:
                best.update({
                    "model": model_name, "cv": cv_name,
                    "macro_f1": summary["f1_macro"],
                    "y_true": y_true, "y_pred": y_pred,
                })

    pd.DataFrame(per_class_rows).to_csv(
        REPORTS / "window_classifier_metrics.csv", index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        REPORTS / "window_classifier_summary.csv", index=False,
    )

    print(f"\n{'='*60}")
    print(f"[best] {best['model']} ({best['cv']})  macro F1 = {best['macro_f1']:.3f}")
    print(f"{'='*60}")

    # ── Final modelleri tüm veriyle eğit + kaydet ───────────────────────────
    print("\n[final] tüm veriyle yeniden eğitiliyor...")
    final_models, _ = make_models(cw_map)
    for model_name, model in final_models.items():
        if model_name == "XGBoost":
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
        with open(MODELS / f"{model_name.lower()}.pkl", "wb") as f:
            pickle.dump({
                "model":         model,
                "label_encoder": le,
                "feature_cols":  feature_cols,
                "imputer":       imputer,
            }, f)
        print(f"  models/window_classifier/{model_name.lower()}.pkl")

    # SHAP — best model (final fit'tan)
    print("\n[SHAP] en iyi model üzerinde feature önemi hesaplanıyor...")
    best_model = final_models[best["model"]]
    rank_rows = plot_shap_top(
        best_model, X, feature_cols, class_names,
        model_name=best["model"],
        save_path=FIGS / f"window_shap_{best['model'].lower()}.png",
    )
    pd.DataFrame(rank_rows).to_csv(
        REPORTS / f"window_shap_{best['model'].lower()}.csv", index=False,
    )
    print(f"  reports/figures/window_shap_{best['model'].lower()}.png")
    print(f"  reports/window_shap_{best['model'].lower()}.csv")

    print(f"\n[done] raporlar: {REPORTS}\n        modeller: {MODELS}\n"
          f"        görseller: {FIGS}")


if __name__ == "__main__":
    main()
