# -*- coding: utf-8 -*-
"""
EPM LOOCV Classifier — Streamlit Web Uygulaması
------------------------------------------------
CSV yükle → LOOCV çalıştır → sonuçları göster

Çalıştırma:
    cd webapp
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score,
    classification_report,
)

# ── Sayfa ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EPM LOOCV Sınıflandırıcı",
    page_icon="🐭",
    layout="wide",
)

COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
}

FEATURES = {
    "pct_open_arm"     : "Açık Kol Süresi (%)",
    "anxiety_index_epm": "Anksiyete İndeksi",
}

# ── LOOCV fonksiyonu ──────────────────────────────────────────────────────────

def run_loocv(X: np.ndarray, y: np.ndarray) -> dict:
    loo        = LeaveOneOut()
    y_true_all, y_prob_all, y_pred_all = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                                  random_state=42)
        clf.fit(X_train, y_train)

        y_true_all.append(int(y[test_idx[0]]))
        y_prob_all.append(float(clf.predict_proba(X_test)[0, 1]))
        y_pred_all.append(int(clf.predict(X_test)[0]))

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = np.array(y_pred_all)

    auc          = roc_auc_score(y_true, y_prob)
    acc          = accuracy_score(y_true, y_pred)
    cm           = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _  = roc_curve(y_true, y_prob)

    report = classification_report(
        y_true, y_pred,
        target_names=["Kontrol", "Tedavi"],
        output_dict=True,
    )

    return {
        "auc": auc, "acc": acc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fpr, "tpr": tpr,
        "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred,
        "report": report,
    }


# ── Grafik fonksiyonları ──────────────────────────────────────────────────────

def fig_roc(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Şans (AUC=0.50)")

    colors = {"pct_open_arm": "#2196F3", "anxiety_index_epm": "#FF9800"}
    for feat, r in results.items():
        j    = r["tpr"] - r["fpr"]
        best = np.argmax(j)
        ax.plot(r["fpr"], r["tpr"],
                color=colors[feat], lw=2.2,
                label=f"{FEATURES[feat]}  AUC={r['auc']:.3f}")
        ax.scatter(r["fpr"][best], r["tpr"][best],
                   color=colors[feat], s=100, zorder=5,
                   edgecolors="white", linewidths=1.5)

    ax.set_xlabel("1 − Özgüllük (FPR)", fontsize=11)
    ax.set_ylabel("Duyarlılık (TPR)", fontsize=11)
    ax.set_title("ROC Eğrisi — LOOCV", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_confusion(r: dict, feat_label: str) -> plt.Figure:
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Kontrol", "Tedavi"], fontsize=10)
    ax.set_yticklabels(["Kontrol", "Tedavi"], fontsize=10)
    ax.set_xlabel("Tahmin", fontsize=10)
    ax.set_ylabel("Gerçek", fontsize=10)
    ax.set_title(feat_label, fontsize=10, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)
    fig.tight_layout()
    return fig


def fig_scores(r: dict, df: pd.DataFrame, feat: str) -> plt.Figure:
    subjs  = df["subject_id"].str.replace("PlusMaze", "").values
    cohort = df["cohort"].values
    probs  = r["y_prob"]
    preds  = r["y_pred"]
    true   = r["y_true"]

    fig, ax = plt.subplots(figsize=(7, max(4, len(subjs) * 0.28)))
    for i, (subj, prob, lbl, pred, coh) in enumerate(
            zip(subjs, probs, true, preds, cohort)):
        color   = COHORT_COLORS.get(coh, "#888888")
        correct = (pred == lbl)
        marker  = "o" if correct else "X"
        ax.scatter(prob, i, color=color, marker=marker, s=90,
                   edgecolors="black" if not correct else "none",
                   linewidths=1.2, zorder=4)
        ax.text(prob + 0.02, i, f"{subj} ({coh})",
                va="center", fontsize=7.5, color="#333")

    ax.axvline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Tedavi tahmin olasılığı", fontsize=10)
    ax.set_xlim(-0.05, 1.25)
    ax.set_yticks([])
    ax.set_title(f"{FEATURES[feat]} — Bireysel Tahminler", fontsize=10,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)

    handles = [
        mpatches.Patch(color=c, label=g)
        for g, c in COHORT_COLORS.items()
        if g in cohort
    ]
    handles.append(
        plt.scatter([], [], marker="X", color="gray", s=60,
                    edgecolors="black", linewidths=1.2, label="Yanlış sınıf")
    )
    ax.legend(handles=handles, fontsize=7, frameon=True,
              loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


# ── Streamlit arayüzü ─────────────────────────────────────────────────────────

st.title("🐭 EPM LOOCV Sınıflandırıcı")
st.markdown(
    "**Elevated Plus Maze** verisi içeren CSV dosyası yükleyin. "
    "Sistem LOOCV ile **Kontrol / Tedavi** ikili sınıflandırması çalıştırır "
    "ve sonuçları görselleştirir."
)
st.markdown("---")

# ── Dosya yükleme ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "CSV dosyası yükle (`plus_maze_metrics_all.csv` formatı)",
    type=["csv"],
)

if uploaded is None:
    st.info(
        "📂 Lütfen bir CSV dosyası yükleyin.\n\n"
        "Gerekli sütunlar: `cohort`, `pct_open_arm`, `anxiety_index_epm`\n\n"
        "Kohort değerleri: `Control` veya diğer herhangi bir değer (= Tedavi)"
    )
    st.stop()

# ── Veriyi oku ────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"CSV okunamadı: {e}")
    st.stop()

required = {"cohort", "pct_open_arm", "anxiety_index_epm"}
missing  = required - set(df.columns)
if missing:
    st.error(f"Eksik sütunlar: {', '.join(missing)}")
    st.stop()

df["label"] = (df["cohort"] != "Control").astype(int)
n_control   = int((df["label"] == 0).sum())
n_treated   = int((df["label"] == 1).sum())

# ── Veri önizleme ─────────────────────────────────────────────────────────────
with st.expander("📊 Veri önizleme", expanded=False):
    show_cols = [c for c in
                 ["subject_id", "cohort", "pct_open_arm",
                  "anxiety_index_epm", "pct_closed_arm", "total_entries"]
                 if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Toplam sıçan", len(df))
col2.metric("Kontrol", n_control)
col3.metric("Tedavi", n_treated)

if n_control < 2 or n_treated < 2:
    st.error("Her sınıfta en az 2 örnek gereklidir.")
    st.stop()

# ── LOOCV çalıştır ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 LOOCV Sonuçları")

with st.spinner("LOOCV hesaplanıyor..."):
    results = {}
    for feat in FEATURES:
        X = df[[feat]].values
        y = df["label"].values
        results[feat] = run_loocv(X, y)

# ── Metrik tablosu ────────────────────────────────────────────────────────────
rows = []
for feat, r in results.items():
    rep = r["report"]
    rows.append({
        "Feature"         : FEATURES[feat],
        "AUC"             : f"{r['auc']:.3f}",
        "Accuracy"        : f"{r['acc']:.3f}",
        "F1 — Tedavi"     : f"{rep['Tedavi']['f1-score']:.3f}",
        "F1 — Kontrol"    : f"{rep['Kontrol']['f1-score']:.3f}",
        "F1 — Weighted"   : f"{rep['weighted avg']['f1-score']:.3f}",
        "Sensitivity"     : f"{rep['Tedavi']['recall']:.3f}",
        "Specificity"     : f"{rep['Kontrol']['recall']:.3f}",
    })

summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── F1 ≥ 0.75 uyarısı ────────────────────────────────────────────────────────
for feat, r in results.items():
    f1_tedavi = r["report"]["Tedavi"]["f1-score"]
    if f1_tedavi >= 0.75:
        st.success(
            f"✅ **{FEATURES[feat]}**: F1 (Tedavi) = {f1_tedavi:.3f} — "
            f"F1 ≥ 0.75 eşiğini karşılıyor."
        )
    else:
        st.warning(
            f"⚠️ **{FEATURES[feat]}**: F1 (Tedavi) = {f1_tedavi:.3f} — "
            f"F1 < 0.75 (sınır altı)."
        )

st.markdown("---")

# ── Grafikler: ROC + Confusion ────────────────────────────────────────────────
col_roc, col_cm1, col_cm2 = st.columns([2, 1, 1])

with col_roc:
    st.pyplot(fig_roc(results), use_container_width=True)

with col_cm1:
    feat = "pct_open_arm"
    st.pyplot(fig_confusion(results[feat], FEATURES[feat]),
              use_container_width=True)

with col_cm2:
    feat = "anxiety_index_epm"
    st.pyplot(fig_confusion(results[feat], FEATURES[feat]),
              use_container_width=True)

st.markdown("---")

# ── Bireysel tahmin skorları ──────────────────────────────────────────────────
st.subheader("🐀 Bireysel Tahmin Olasılıkları")

tab1, tab2 = st.tabs(list(FEATURES.values()))
for tab, feat in zip([tab1, tab2], FEATURES):
    with tab:
        st.pyplot(fig_scores(results[feat], df, feat),
                  use_container_width=True)

# ── CSV indir ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Sonuçları İndir")

detail_rows = []
for feat, r in results.items():
    subj_col = df["subject_id"].str.replace("PlusMaze", "") \
               if "subject_id" in df.columns else df.index.astype(str)
    for i, (subj, prob, pred, true) in enumerate(
            zip(subj_col, r["y_prob"], r["y_pred"], r["y_true"])):
        detail_rows.append({
            "subject"    : subj,
            "cohort"     : df["cohort"].iloc[i],
            "feature"    : FEATURES[feat],
            "true_label" : "Kontrol" if true == 0 else "Tedavi",
            "pred_label" : "Kontrol" if pred == 0 else "Tedavi",
            "prob_treated": round(float(prob), 4),
            "correct"    : bool(pred == true),
        })

detail_df = pd.DataFrame(detail_rows)
csv_bytes  = detail_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

st.download_button(
    label="⬇️ Tahmin detaylarını CSV olarak indir",
    data=csv_bytes,
    file_name="loocv_predictions.csv",
    mime="text/csv",
)
