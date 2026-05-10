"""
Anxiety Classifier — Binary (Control vs Treated)
-------------------------------------------------
src/anxiety/profile.py tarafından üretilen anxiety_features.csv'yi
girdi olarak alır. Hedef: `is_treated = (group != "Control")`.

Model ailesi: Logistic Regression (L2), Random Forest, SVM (RBF).
Doğrulama: LeaveOneOut (n=29). class_weight="balanced" — kontrol grubu
~7, işlenmiş ~22 olduğu için.

Çıktılar
--------
    reports/anxiety_classifier_metrics.csv      — model × metric
    reports/anxiety_classifier_predictions.csv  — fold başına tahmin
    reports/anxiety_classifier_importance.csv   — RF importance + LR coef
    reports/figures/anxiety_classifier_cm.png   — 3 model confusion matrix
    reports/figures/anxiety_classifier_roc.png  — aggregate ROC eğrileri
    models/anxiety_classifier/{lr,rf,svm}.pkl   — yeniden eğitilmiş tam-veri modeli
    models/anxiety_classifier/scaler.pkl

Kullanım
--------
    python -m src.anxiety.classifier
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
FEAT_CSV = ROOT / "data" / "anxiety_features.csv"
MODELS   = ROOT / "models" / "anxiety_classifier"
REPORTS  = ROOT / "reports"
FIGS     = REPORTS / "figures"

for d in (MODELS, REPORTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

META_COLS    = {"subject_id", "cohort", "group", "pc1_score", "pc2_score"}
NA_THRESHOLD = 0.5   # >%50 NaN olan sütunları at
RNG_SEED     = 42


def make_models() -> dict:
    """Üç model — hepsi class_weight='balanced'."""
    return {
        "LogisticReg": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            class_weight="balanced", max_iter=1000, random_state=RNG_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=2,
            class_weight="balanced", random_state=RNG_SEED, n_jobs=-1,
        ),
        "SVM-RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=RNG_SEED,
        ),
    }


# ── Veri ──────────────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if not FEAT_CSV.exists():
        raise FileNotFoundError(
            f"{FEAT_CSV} yok — önce `python -m src.anxiety.profile` çalıştır."
        )
    df = pd.read_csv(FEAT_CSV)
    if "group" not in df.columns:
        raise ValueError("anxiety_features.csv 'group' sütununu içermiyor.")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    # NaN ağırlıklı sütunları at
    na_frac = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if na_frac[c] < NA_THRESHOLD]

    X = df[feature_cols].values.astype(float)
    y = (df["group"].astype(str).str.lower() != "control").astype(int).values
    print(f"[load] n={len(df)}  features={len(feature_cols)}  "
          f"control={int((y == 0).sum())}  treated={int((y == 1).sum())}")
    return df, X, y, feature_cols


# ── LOOCV ─────────────────────────────────────────────────────────────────────

def loocv_run(X: np.ndarray, y: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    """LOOCV — her fold'da 1 hayvan test, kalan eğitim. predict_proba toplu döner."""
    loo = LeaveOneOut()
    n = len(y)
    pred  = np.zeros(n, dtype=int)
    proba = np.zeros(n, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr       = y[train_idx]

        imp = SimpleImputer(strategy="median")
        X_tr_i = imp.fit_transform(X_tr)
        X_te_i = imp.transform(X_te)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_i)
        X_te_s = sc.transform(X_te_i)

        m = model.__class__(**model.get_params())
        m.fit(X_tr_s, y_tr)

        pred[test_idx]  = m.predict(X_te_s)
        proba[test_idx] = m.predict_proba(X_te_s)[:, 1]

    return pred, proba


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray) -> dict:
    return {
        "accuracy":       round(accuracy_score(y_true, y_pred), 3),
        "balanced_acc":   round(balanced_accuracy_score(y_true, y_pred), 3),
        "f1_macro":       round(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0), 3),
        "f1_treated":     round(f1_score(y_true, y_pred, pos_label=1,
                                         zero_division=0), 3),
        "f1_control":     round(f1_score(y_true, y_pred, pos_label=0,
                                         zero_division=0), 3),
        "auc":            (round(roc_auc_score(y_true, y_proba), 3)
                           if len(np.unique(y_true)) == 2 else float("nan")),
    }


# ── Görselleştirme ────────────────────────────────────────────────────────────

def plot_confusion_grid(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, color="black" if cm[i, j] < cm.max() / 2 else "white")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Control", "Treated"])
        ax.set_yticklabels(["Control", "Treated"])
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        ax.set_title(f"{name}\nacc={r['metrics']['accuracy']:.2f}  "
                     f"AUC={r['metrics']['auc']:.2f}", fontsize=10)
    fig.suptitle("LOOCV Confusion Matrix — Control vs Treated",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


def plot_roc(results: dict, y: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y, r["y_proba"])
        ax.plot(fpr, tpr, lw=2,
                label=f"{name} (AUC={r['metrics']['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Control vs Treated (LOOCV aggregated)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


# ── Tam veri ile yeniden eğitim + feature importance ──────────────────────────

def fit_full_and_save(X: np.ndarray, y: np.ndarray,
                      feature_cols: list[str]) -> pd.DataFrame:
    """Tüm veride yeniden eğit, pickle olarak kaydet, önemleri döner."""
    imp = SimpleImputer(strategy="median")
    X_i = imp.fit_transform(X)
    sc  = StandardScaler()
    X_s = sc.fit_transform(X_i)

    with open(MODELS / "scaler.pkl", "wb") as f:
        pickle.dump({"imputer": imp, "scaler": sc, "features": feature_cols}, f)
    print(f"[write] models/anxiety_classifier/scaler.pkl")

    importances = {}
    for name, model in make_models().items():
        m = model.__class__(**model.get_params())
        m.fit(X_s, y)
        slug = {"LogisticReg": "lr", "RandomForest": "rf", "SVM-RBF": "svm"}[name]
        with open(MODELS / f"{slug}.pkl", "wb") as f:
            pickle.dump(m, f)
        print(f"[write] models/anxiety_classifier/{slug}.pkl")

        if name == "RandomForest":
            importances["rf_importance"] = m.feature_importances_
        elif name == "LogisticReg":
            importances["lr_coef"] = m.coef_.ravel()

    imp_df = pd.DataFrame({"feature": feature_cols, **importances})
    if "rf_importance" in imp_df:
        imp_df = imp_df.sort_values("rf_importance", ascending=False)
    return imp_df


# ── Ana ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df, X, y, feature_cols = load_features()

    print("\n[loocv] LOOCV başlıyor...")
    results: dict[str, dict] = {}
    for name, model in make_models().items():
        pred, proba = loocv_run(X, y, model)
        metrics = evaluate(y, pred, proba)
        results[name] = {"y_true": y, "y_pred": pred,
                         "y_proba": proba, "metrics": metrics}
        print(f"  {name:<14s}  "
              f"acc={metrics['accuracy']:.2f}  "
              f"bal_acc={metrics['balanced_acc']:.2f}  "
              f"F1_macro={metrics['f1_macro']:.2f}  "
              f"AUC={metrics['auc']:.2f}")

    # metrics CSV
    metrics_rows = [{"model": n, **r["metrics"]} for n, r in results.items()]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = REPORTS / "anxiety_classifier_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[write] {metrics_path.relative_to(ROOT)}")

    # per-fold predictions CSV
    pred_rows = []
    for name, r in results.items():
        for sid, true, pred, prob in zip(df["subject_id"], r["y_true"],
                                         r["y_pred"], r["y_proba"]):
            pred_rows.append({"subject_id": sid, "model": name,
                              "true": int(true), "pred": int(pred),
                              "proba_treated": round(float(prob), 3)})
    pred_df = pd.DataFrame(pred_rows)
    pred_path = REPORTS / "anxiety_classifier_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[write] {pred_path.relative_to(ROOT)}")

    plot_confusion_grid(results, FIGS / "anxiety_classifier_cm.png")
    plot_roc(results, y, FIGS / "anxiety_classifier_roc.png")

    print("\n[refit] tam veri üzerinde yeniden eğitim + pickle...")
    imp_df = fit_full_and_save(X, y, feature_cols)
    imp_path = REPORTS / "anxiety_classifier_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"[write] {imp_path.relative_to(ROOT)}")

    print("\n[top features by RF importance]")
    print(imp_df.head(8).to_string(index=False))
    print("\n[done]")


if __name__ == "__main__":
    main()
