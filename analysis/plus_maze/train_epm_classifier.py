# -*- coding: utf-8 -*-
"""
train_epm_classifier.py
-----------------------
plus_maze_metrics_all.csv → Control vs Treated ikili sınıflandırıcı.

Open Field'daki src/anxiety/classifier.py ile aynı yapı:
  1. LOOCV  → performans değerlendirme (AUC, F1, confusion matrix)
  2. Refit  → tüm 37 subject üzerinde yeniden eğit
  3. Kaydet → models/epm_classifier/ altına .pkl dosyaları

Modeli yeniden eğitmek için:
    python analysis/plus_maze/train_epm_classifier.py

GUI / webapp bu modeli yükleyip tek bir yeni CSV için tahmin yapar.
"""

import pathlib
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).resolve().parent.parent.parent
DATA     = ROOT / "data" / "plus_maze_metrics_all.csv"
MODELS   = ROOT / "models" / "epm_classifier"
REPORTS  = ROOT / "reports"
FIGS     = REPORTS / "figures" / "plus_maze"

MODELS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

RNG = 42

# Modele dahil edilecek feature sütunları
FEATURE_COLS = [
    "pct_open_arm",
    "anxiety_index_epm",
    "pct_open_arm_entries",
    "total_entries",
    "successive_alternation_pct",
    "perseveration_rate_pct",
    "mean_speed_px_s",
    "arm_preference_index",
    "pct_time_junction",
]


# ── Veri yükleme ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(DATA)
    df["label"] = (df["cohort"] != "Control").astype(int)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")

    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values
    print(f"[load] {len(df)} subject | "
          f"Control={int((y==0).sum())}  Treated={int((y==1).sum())}")
    return X, y, df


# ── LOOCV değerlendirme ───────────────────────────────────────────────────────

def run_loocv(X: np.ndarray, y: np.ndarray) -> dict:
    loo          = LeaveOneOut()
    y_true_all, y_prob_all, y_pred_all = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr       = y[train_idx]

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        clf = LogisticRegression(class_weight="balanced",
                                 max_iter=1000, random_state=RNG)
        clf.fit(X_tr, y_tr)

        y_true_all.append(int(y[test_idx[0]]))
        y_prob_all.append(float(clf.predict_proba(X_te)[0, 1]))
        y_pred_all.append(int(clf.predict(X_te)[0]))

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = np.array(y_pred_all)

    auc            = roc_auc_score(y_true, y_prob)
    acc            = accuracy_score(y_true, y_pred)
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _    = roc_curve(y_true, y_prob)
    report         = classification_report(
        y_true, y_pred,
        target_names=["Control", "Treated"],
        output_dict=True,
    )

    print(f"\n{'='*50}")
    print(f"LOOCV  AUC={auc:.3f}  Acc={acc:.3f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  F1(Treated)={report['Treated']['f1-score']:.3f}  "
          f"F1(Control)={report['Control']['f1-score']:.3f}")
    print(f"  Sensitivity={report['Treated']['recall']:.3f}  "
          f"Specificity={report['Control']['recall']:.3f}")

    return {
        "auc": auc, "acc": acc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fpr, "tpr": tpr,
        "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred,
        "report": report,
    }


# ── Görselleştirme ────────────────────────────────────────────────────────────

def plot_roc(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    j    = res["tpr"] - res["fpr"]
    best = np.argmax(j)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Şans (AUC=0.50)")
    ax.plot(res["fpr"], res["tpr"], lw=2.5, color="#2196F3",
            label=f"EPM Sınıflandırıcı  AUC={res['auc']:.3f}")
    ax.scatter(res["fpr"][best], res["tpr"][best],
               color="#2196F3", s=130, zorder=5,
               edgecolors="white", linewidths=1.5,
               label="Youden J optimum")

    ax.set_xlabel("1 − Özgüllük (FPR)", fontsize=11)
    ax.set_ylabel("Duyarlılık (TPR)", fontsize=11)
    ax.set_title("EPM Sınıflandırıcı — ROC Eğrisi (LOOCV)\n"
                 f"Control (n=7) vs Treated (n=30)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = FIGS / "epm_classifier_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


def plot_confusion(res: dict) -> None:
    cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control", "Treated"], fontsize=11)
    ax.set_yticklabels(["Control", "Treated"], fontsize=11)
    ax.set_xlabel("Tahmin", fontsize=11)
    ax.set_ylabel("Gerçek", fontsize=11)
    ax.set_title(f"LOOCV  AUC={res['auc']:.3f}  Acc={res['acc']:.3f}",
                 fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
    fig.tight_layout()

    out = FIGS / "epm_classifier_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


# ── Tam veri üzerinde yeniden eğit + kaydet ───────────────────────────────────

def fit_full_and_save(X: np.ndarray, y: np.ndarray) -> dict:
    """Tüm 37 subject üzerinde eğit ve .pkl olarak kaydet."""
    imp = SimpleImputer(strategy="median")
    X_i = imp.fit_transform(X)

    sc = StandardScaler()
    X_s = sc.fit_transform(X_i)

    clf = LogisticRegression(class_weight="balanced",
                             max_iter=1000, random_state=RNG)
    clf.fit(X_s, y)

    bundle = {
        "imputer":  imp,
        "scaler":   sc,
        "features": FEATURE_COLS,
    }

    scaler_path = MODELS / "scaler_epm.pkl"
    model_path  = MODELS / "lr_epm.pkl"

    with open(scaler_path, "wb") as f:
        pickle.dump(bundle, f)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"[write] {scaler_path.relative_to(ROOT)}")
    print(f"[write] {model_path.relative_to(ROOT)}")

    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coef":    clf.coef_.ravel().round(4),
    }).sort_values("coef", key=abs, ascending=False)
    print("\n[top coefficients]")
    print(coef_df.to_string(index=False))

    return {"clf": clf, "bundle": bundle}


# ── Özet CSV ─────────────────────────────────────────────────────────────────

def save_loocv_csv(res: dict, df: pd.DataFrame) -> None:
    rows = []
    for i, (subj, coh, true, pred, prob) in enumerate(zip(
            df["subject_id"], df["cohort"],
            res["y_true"], res["y_pred"], res["y_prob"])):
        rows.append({
            "subject_id":    subj,
            "cohort":        coh,
            "true_label":    "Control" if true == 0 else "Treated",
            "pred_label":    "Control" if pred == 0 else "Treated",
            "prob_treated":  round(float(prob), 4),
            "correct":       bool(pred == true),
        })
    out = REPORTS / "epm_classifier_loocv.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[write] {out.relative_to(ROOT)}")


# ── Ana ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Feature seti ({len(FEATURE_COLS)} özellik):")
    for i, f in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {f}")

    X, y, df = load_data()

    print("\n[loocv] Değerlendirme başlıyor...")
    res = run_loocv(X, y)

    plot_roc(res)
    plot_confusion(res)
    save_loocv_csv(res, df)

    print("\n[refit] Tam veri üzerinde yeniden eğitim...")
    fit_full_and_save(X, y)

    print(f"\n{'='*50}")
    print("Model eğitimi tamamlandı.")
    print(f"  AUC  = {res['auc']:.3f}")
    print(f"  F1(Treated) = {res['report']['Treated']['f1-score']:.3f}")
    print(f"  pkl  -> models/epm_classifier/")
    print(f"  Gorseller -> reports/figures/plus_maze/")


if __name__ == "__main__":
    main()
