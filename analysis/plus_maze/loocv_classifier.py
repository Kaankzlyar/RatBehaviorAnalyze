# -*- coding: utf-8 -*-
"""
loocv_classifier.py
-------------------
Control vs Treated (ikili) sınıflandırma — LOOCV ile.

Feature seti:
  A) pct_open_arm          (tek metrik)
  B) anxiety_index_epm     (tek metrik, karşılaştırma)

Sınıflandırıcı: Logistic Regression (class_weight='balanced')
Değerlendirme : LOOCV — AUC-ROC, accuracy, sensitivity, specificity,
                confusion matrix, calibration curve

Çıktılar:
  reports/loocv_results.csv
  reports/figures/plus_maze/loocv_roc.png
  reports/figures/plus_maze/loocv_confusion.png
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay,
)

ROOT  = pathlib.Path(__file__).resolve().parent.parent.parent
DATA  = ROOT / "data" / "plus_maze_metrics_all.csv"
FIGS  = ROOT / "reports" / "figures" / "plus_maze"
FIGS.mkdir(parents=True, exist_ok=True)

FEATURES = {
    "pct_open_arm"     : "Açık Kol Süresi (%)",
    "anxiety_index_epm": "Anksiyete İndeksi",
}


def run_loocv(X: np.ndarray, y: np.ndarray, feature_name: str) -> dict:
    """Tek feature için LOOCV çalıştır, tüm metrikleri döndür."""
    loo        = LeaveOneOut()
    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                                  random_state=42)
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[0, 1]
        pred = clf.predict(X_test)[0]

        y_true_all.append(y[test_idx[0]])
        y_prob_all.append(prob)
        y_pred_all.append(pred)

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = np.array(y_pred_all)

    auc  = roc_auc_score(y_true, y_prob)
    acc  = accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)

    # Sensitivity (Recall for Treated=1) ve Specificity (Recall for Control=0)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"Feature : {feature_name}")
    print(f"AUC-ROC  : {auc:.3f}")
    print(f"Accuracy : {acc:.3f}  ({int(acc*len(y_true))}/{len(y_true)})")
    print(f"Sensitivity (Treated): {sens:.3f}")
    print(f"Specificity (Control): {spec:.3f}")
    print(f"Confusion Matrix:\n  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}")

    return {
        "feature"    : feature_name,
        "auc"        : round(auc, 3),
        "accuracy"   : round(acc, 3),
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fpr, "tpr": tpr,
        "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred,
    }


def plot_roc(results: list[dict]) -> None:
    """İki feature için ROC eğrisi — tek grafik."""
    colors = {"pct_open_arm": "#2196F3", "anxiety_index_epm": "#FF9800"}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Şans seviyesi (AUC=0.50)")

    for r in results:
        label = f"{FEATURES[r['feature']]}  (AUC = {r['auc']:.3f})"
        ax.plot(r["fpr"], r["tpr"],
                color=colors.get(r["feature"], "#555555"),
                lw=2.5, label=label)
        # Optimal eşik noktası (Youden J)
        j       = r["tpr"] - r["fpr"]
        best    = np.argmax(j)
        ax.scatter(r["fpr"][best], r["tpr"][best],
                   color=colors.get(r["feature"], "#555555"),
                   s=120, zorder=5, edgecolors="white", linewidths=1.5)

    ax.set_xlabel("1 − Özgüllük (FPR)", fontsize=12)
    ax.set_ylabel("Duyarlılık (TPR)", fontsize=12)
    ax.set_title(
        "LOOCV — ROC Eğrisi\nKontrol vs Tedavi (ikili sınıflandırma)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=True, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.01,
             "● = Youden J optimum eşiği  |  Sınıflandırıcı: Logistic Regression (balanced)",
             ha="center", fontsize=8, color="#666666", style="italic")

    out = FIGS / "loocv_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] {out}")


def plot_confusion(results: list[dict]) -> None:
    """Her feature için confusion matrix — yan yana."""
    n      = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.82, wspace=0.35)

    for ax, r in zip(axes, results):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Kontrol", "Tedavi"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{FEATURES[r['feature']]}\n"
            f"AUC={r['auc']:.3f}  Acc={r['accuracy']:.3f}\n"
            f"Sens={r['sensitivity']:.3f}  Spec={r['specificity']:.3f}",
            fontsize=10, fontweight="bold",
        )

    fig.suptitle(
        "LOOCV — Confusion Matrix\nKontrol (n=7) vs Tedavi (n=30)",
        fontsize=12, fontweight="bold", y=0.98,
    )
    out = FIGS / "loocv_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_scores(results: list[dict], df: pd.DataFrame, label: np.ndarray) -> None:
    """Her sıçan için tahmin olasılığı — gerçek etikete göre renklendirme."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.84, wspace=0.35)

    subject_ids = df["subject_id"].str.replace("PlusMaze", "").values
    colors_true = {0: "#4CAF50", 1: "#9C27B0"}  # yeşil=kontrol, mor=tedavi

    for ax, r in zip(axes, results):
        probs = r["y_prob"]
        for i, (subj, prob, lbl) in enumerate(zip(subject_ids, probs, label)):
            correct = (r["y_pred"][i] == lbl)
            marker  = "o" if correct else "X"
            ax.scatter(prob, i,
                       color=colors_true[lbl],
                       marker=marker, s=100,
                       edgecolors="black" if not correct else "none",
                       linewidths=1.2, zorder=4)
            ax.text(prob + 0.02, i, subj, va="center", fontsize=7, color="#444")

        ax.axvline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.7)
        ax.set_xlabel("Tedavi tahmin olasılığı", fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(-0.05, 1.15)
        ax.set_title(
            f"{FEATURES[r['feature']]}\nAUC = {r['auc']:.3f}",
            fontsize=10, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        ax.spines[["top", "right", "left"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#4CAF50", label="Kontrol (gerçek)"),
        mpatches.Patch(color="#9C27B0", label="Tedavi (gerçek)"),
        plt.scatter([], [], marker="X", color="gray", s=80,
                    edgecolors="black", linewidths=1.2, label="Yanlış sınıf"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("LOOCV — Bireysel Tahmin Olasılıkları",
                 fontsize=12, fontweight="bold", y=0.98)

    out = FIGS / "loocv_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def main() -> None:
    df = pd.read_csv(DATA)

    # Etiket: Control=0, diğerleri=1
    df["label"] = (df["cohort"] != "Control").astype(int)
    label       = df["label"].values

    print(f"Kontrol: {(label==0).sum()} sıçan")
    print(f"Tedavi : {(label==1).sum()} sıçan")
    print(f"Toplam : {len(label)} sıçan\n")

    results = []
    for feat in FEATURES:
        X = df[[feat]].values
        r = run_loocv(X, label, feat)
        results.append(r)

    # Grafikleri üret
    plot_roc(results)
    plot_confusion(results)
    plot_scores(results, df, label)

    # CSV çıktısı
    summary = pd.DataFrame([{
        "feature"    : FEATURES[r["feature"]],
        "auc"        : r["auc"],
        "accuracy"   : r["accuracy"],
        "sensitivity": r["sensitivity"],
        "specificity": r["specificity"],
        "tn": r["tn"], "fp": r["fp"],
        "fn": r["fn"], "tp": r["tp"],
    } for r in results])

    out_csv = ROOT / "reports" / "loocv_results.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {out_csv}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
