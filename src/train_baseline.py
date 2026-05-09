"""
Aşama 06 — ML Model Eğitimi (LOOCV)
Çıktılar:
  reports/model_comparison.csv
  reports/figures/confusion_<hedef>.png
  reports/figures/shap_<hedef>.png
  models/classifier/<model>_<hedef>.pkl
"""

import pathlib
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

# ── Dizinler ─────────────────────────────────────────────────────────────────

ROOT    = pathlib.Path(__file__).parent.parent
FEAT    = ROOT / "data" / "features"
MODELS  = ROOT / "models" / "classifier"
REPORTS = ROOT / "reports"
FIGS    = REPORTS / "figures"

for d in [MODELS, REPORTS, FIGS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Veri ─────────────────────────────────────────────────────────────────────

feat_norm = pd.read_csv(FEAT / "features_normalized.csv")
labels    = pd.read_csv(FEAT / "labels.csv")

FEATURE_COLS = [c for c in feat_norm.columns
                if c not in ("subject_id", "cohort", "group", "session_duration_s")]

X = feat_norm[FEATURE_COLS].values

# ── NaN Handling ──────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

TARGETS = {
    "group":            labels["group"].values,
    "anxiety_level":    labels["anxiety_level"].values,
    "rearing_profile":  labels["rearing_profile"].values,
    "grooming_profile": labels["grooming_profile"].values,
}

GROUP_ORDER = {
    "group":            ["Control", "Aspartame", "Grapefruit", "Aspartame+Grapefruit"],
    "anxiety_level":    ["low", "moderate", "high"],
    "rearing_profile":  ["low", "moderate", "high"],
    "grooming_profile": ["low", "moderate", "high"],
}

SUBJECT_IDS = feat_norm["subject_id"].values
COHORTS     = feat_norm["cohort"].values

# ── Model tanımları ───────────────────────────────────────────────────────────

def make_models(n_classes):
    models = {
        "LogisticReg": LogisticRegression(
            penalty="l1", solver="saga", C=0.5,
            max_iter=2000, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=3,
            min_samples_leaf=2, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=2,
            learning_rate=0.05, subsample=0.8,
            eval_metric="mlogloss",
            random_state=42, verbosity=0,
        ),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale",
                   probability=True, random_state=42),
    }
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=3, num_leaves=7,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=2, random_state=42, verbosity=-1,
        )
    return models

# ── CV eğitim fonksiyonu (LOOCV veya LeaveOneGroupOut) ───────────────────────

def cv_evaluate(model, X, y_enc, le, subject_ids, splitter, groups=None):
    y_true, y_pred = [], []
    subjects_out = []
    n_classes = len(le.classes_)

    split_args = (X, y_enc, groups) if groups is not None else (X, y_enc)
    for train_idx, test_idx in splitter.split(*split_args):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_enc[train_idx]

        # Eğitim setinde eksik sınıf varsa dummy satır ekle (XGBoost için)
        missing = set(range(n_classes)) - set(np.unique(y_tr))
        if missing:
            for cls in missing:
                X_tr = np.vstack([X_tr, X_tr[0:1]])
                y_tr = np.append(y_tr, cls)

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        y_pred.extend(preds.tolist())
        y_true.extend(y_enc[test_idx].tolist())
        subjects_out.extend(subject_ids[test_idx].tolist())

    all_labels = list(range(len(le.classes_)))
    f1  = f1_score(y_true, y_pred, average="macro",
                   labels=all_labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=all_labels)

    pred_labels = le.inverse_transform(y_pred)
    true_labels = le.inverse_transform(y_true)

    return f1, acc, cm, true_labels, pred_labels, subjects_out


# Backward-compat shim
def loocv_evaluate(model, X, y_enc, le, subject_ids):
    return cv_evaluate(model, X, y_enc, le, subject_ids, LeaveOneOut())


# ── Karışıklık matrisi grafiği ────────────────────────────────────────────────

def plot_confusion(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(max(4, len(class_names)*1.4),
                                    max(3, len(class_names)*1.2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short = [c.replace("Aspartame+Grapefruit", "ASP+GF") for c in class_names]
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=10)
    ax.set_xlabel("Tahmin", fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── SHAP grafiği (XGBoost) ────────────────────────────────────────────────────

def plot_shap(model, X, feature_names, class_names, target_name, save_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    n_classes = len(class_names)

    if n_classes > 2:
        # çok sınıflı: her sınıf için ayrı SHAP matrisi var
        # tüm sınıfların mutlak SHAP ortalaması
        mean_abs = np.mean(
            [np.abs(shap_values[c]) for c in range(n_classes)], axis=0
        )  # (n_samples, n_features)
        global_importance = mean_abs.mean(axis=0)
        top_idx = np.argsort(global_importance)[::-1][:12]

        fig, axes = plt.subplots(1, n_classes,
                                 figsize=(4 * n_classes, 5),
                                 sharey=True)
        if n_classes == 1:
            axes = [axes]
        for c, ax in enumerate(axes):
            vals = shap_values[c][:, top_idx]
            feat_labels = [feature_names[i] for i in top_idx]
            means = np.abs(vals).mean(axis=0)
            order = np.argsort(means)[::-1]
            ax.barh(range(len(order)),
                    means[order],
                    color="#2196F3", alpha=0.8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([feat_labels[i] for i in order], fontsize=8)
            short_cls = class_names[c].replace("Aspartame+Grapefruit", "ASP+GF")
            ax.set_title(f"{short_cls}", fontsize=10)
            ax.set_xlabel("Ort. |SHAP|", fontsize=8)
            ax.invert_yaxis()
        fig.suptitle(f"SHAP Özellik Önemi — {target_name}", fontsize=12)
    else:
        # ikili sınıflandırma
        sv = shap_values if not isinstance(shap_values, list) else shap_values[1]
        global_importance = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(global_importance)[::-1][:12]
        fig, ax = plt.subplots(figsize=(6, 5))
        feat_labels = [feature_names[i] for i in top_idx]
        means = global_importance[top_idx]
        order = np.argsort(means)
        ax.barh(range(len(order)),
                means[order], color="#2196F3", alpha=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feat_labels[i] for i in order], fontsize=9)
        ax.set_title(f"SHAP Özellik Önemi — {target_name}", fontsize=11)
        ax.set_xlabel("Ort. |SHAP|", fontsize=9)
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Ana döngü ────────────────────────────────────────────────────────────────

results = []
loocv_details = []

for target_name, y_raw in TARGETS.items():
    print(f"\n{'='*55}")
    print(f"Hedef: {target_name}")
    print(f"{'='*55}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    class_names = list(le.classes_)

    models = make_models(n_classes)
    best_f1 = -1
    best_cm  = None
    xgb_model_final = None

    for model_name, model in models.items():
        # ── LOOCV ──
        f1_loo, acc_loo, cm_loo, true_lbl, pred_lbl, subs = cv_evaluate(
            model, X, y_enc, le, SUBJECT_IDS, LeaveOneOut()
        )
        print(f"  {model_name:<15}  LOOCV   F1={f1_loo:.3f}  Acc={acc_loo:.3f}")

        results.append({
            "target":     target_name,
            "model":      model_name,
            "cv":         "LOOCV",
            "f1_macro":   round(f1_loo, 4),
            "accuracy":   round(acc_loo, 4),
            "n_classes":  n_classes,
        })

        for sid, tl, pl in zip(subs, true_lbl, pred_lbl):
            loocv_details.append({
                "target":    target_name,
                "model":     model_name,
                "subject_id": sid,
                "true":      tl,
                "predicted": pl,
                "correct":   tl == pl,
            })

        # ── LOGOCV (cohort dışarıda bırak) — gerçek genelleme testi ──
        f1_g, acc_g, _, _, _, _ = cv_evaluate(
            model, X, y_enc, le, SUBJECT_IDS,
            LeaveOneGroupOut(), groups=COHORTS,
        )
        print(f"  {' '*15}  LOGOCV  F1={f1_g:.3f}  Acc={acc_g:.3f}")
        results.append({
            "target":     target_name,
            "model":      model_name,
            "cv":         "LOGOCV",
            "f1_macro":   round(f1_g, 4),
            "accuracy":   round(acc_g, 4),
            "n_classes":  n_classes,
        })

        # Görsellerde LOOCV en-iyi confusion matrix kullanılıyor
        if f1_loo > best_f1:
            best_f1 = f1_loo
            best_cm = cm_loo

        # modeli kaydet (tüm veriyle eğitilmiş)
        model.fit(X, y_enc)
        pkl_path = MODELS / f"{model_name.lower()}_{target_name}.pkl"
        with open(pkl_path, "wb") as f_pkl:
            pickle.dump({"model": model, "label_encoder": le}, f_pkl)

        # XGBoost modelini SHAP için sakla
        if model_name == "XGBoost":
            xgb_model_final = model

    # En iyi modelin karışıklık matrisi
    plot_confusion(
        best_cm, class_names,
        title=f"En İyi Model — {target_name} (F1={best_f1:.3f})",
        save_path=FIGS / f"confusion_{target_name}.png",
    )

    # XGBoost SHAP grafiği
    if xgb_model_final is not None:
        try:
            plot_shap(
                xgb_model_final, X,
                feature_names=FEATURE_COLS,
                class_names=class_names,
                target_name=target_name,
                save_path=FIGS / f"shap_{target_name}.png",
            )
            print(f"  [SHAP] shap_{target_name}.png kaydedildi")
        except Exception as e:
            print(f"  [SHAP hata] {e}")

# ── Karşılaştırma tablosu ─────────────────────────────────────────────────────

df_results = pd.DataFrame(results)

# Geriye dönük uyumluluk: model_comparison.csv yalnız LOOCV (cv sütunu olmadan)
df_loocv = df_results[df_results["cv"] == "LOOCV"].drop(columns=["cv"])
df_loocv.to_csv(REPORTS / "model_comparison.csv", index=False)

df_logocv = df_results[df_results["cv"] == "LOGOCV"].drop(columns=["cv"])
df_logocv.to_csv(REPORTS / "model_comparison_logocv.csv", index=False)

# Birleşik tablo (cv sütunlu)
df_results.to_csv(REPORTS / "model_comparison_all.csv", index=False)

df_details = pd.DataFrame(loocv_details)
df_details.to_csv(REPORTS / "loocv_predictions.csv", index=False)

# ── Özet tablo yazdır ─────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("SONUÇ TABLOSU — F1 Macro (LOOCV)")
print(f"{'='*55}")
pivot_loo = df_loocv.pivot(index="model", columns="target", values="f1_macro")
print(pivot_loo.to_string())

print(f"\n{'='*55}")
print("SONUÇ TABLOSU — F1 Macro (LOGOCV — cohort hold-out)")
print(f"{'='*55}")
pivot_log = df_logocv.pivot(index="model", columns="target", values="f1_macro")
print(pivot_log.to_string())

# ── CV stratejisi karşılaştırma grafiği ──────────────────────────────────────

target_list = list(TARGETS.keys())
n_targets = len(target_list)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

bar_colors = {"LOOCV": "#4C72B0", "LOGOCV": "#DD8452"}
bar_width = 0.38

for ax, target in zip(axes, target_list):
    df_t = df_results[df_results["target"] == target]
    pivot = df_t.pivot(index="model", columns="cv", values="f1_macro")
    # Sıralı sütunlar
    cv_cols = [c for c in ["LOOCV", "LOGOCV"] if c in pivot.columns]
    pivot = pivot[cv_cols]

    model_names = list(pivot.index)
    x = np.arange(len(model_names))
    for j, cv_name in enumerate(cv_cols):
        offset = (j - (len(cv_cols) - 1) / 2) * bar_width
        vals = pivot[cv_name].values
        bars = ax.bar(x + offset, vals, width=bar_width,
                      color=bar_colors[cv_name], label=cv_name, alpha=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    n_cls = int(df_t["n_classes"].iloc[0])
    chance = 1.0 / n_cls
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"şans (1/{n_cls})")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Macro", fontsize=10)
    ax.set_title(target, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("CV Stratejisi Karşılaştırması — LOOCV (denek hold-out) vs "
             "LOGOCV (cohort hold-out)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
fig.savefig(FIGS / "cv_strategy_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n[OK] cv_strategy_comparison.png kaydedildi")
print(f"\nRaporlar: {REPORTS}")
print(f"Modeller: {MODELS}")
print(f"Görseller: {FIGS}")

# ── One-vs-Rest SHAP — grup bazlı ayırt edici özellikler ─────────────────────

print(f"\n{'='*55}")
print("ONE-VS-REST SHAP — Her grubun özgün imzası")
print(f"{'='*55}")

groups = labels["group"].values
unique_groups = ["Control", "Aspartame", "Grapefruit", "Aspartame+Grapefruit"]
group_short   = ["Control", "Aspartame", "Grapefruit", "ASP+GF"]
colors        = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

TOP_N = 8
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

ovr_rows = []

for i, (grp, short, color) in enumerate(zip(unique_groups, group_short, colors)):
    y_bin = (groups == grp).astype(int)

    # 3 pozitif / 9 negatif → scale_pos_weight=3 ile dengele
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    spw   = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_bin = XGBClassifier(
        n_estimators=200, max_depth=2,
        learning_rate=0.05, subsample=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42, verbosity=0,
    )

    def _ovr_cv_score(splitter, groups=None):
        y_true_b, y_pred_b = [], []
        split_args = (X, y_bin, groups) if groups is not None else (X, y_bin)
        for tr_idx, te_idx in splitter.split(*split_args):
            n_pos_tr = y_bin[tr_idx].sum()
            n_neg_tr = len(y_bin[tr_idx]) - n_pos_tr
            spw_tr   = n_neg_tr / n_pos_tr if n_pos_tr > 0 else 1.0
            xgb_bin.set_params(scale_pos_weight=spw_tr)
            xgb_bin.fit(X[tr_idx], y_bin[tr_idx])
            y_pred_b.extend(xgb_bin.predict(X[te_idx]).tolist())
            y_true_b.extend(y_bin[te_idx].tolist())
        return (f1_score(y_true_b, y_pred_b, average="binary", zero_division=0),
                accuracy_score(y_true_b, y_pred_b))

    # LOOCV (denek hold-out) — eski metrik
    f1_bin, acc_bin = _ovr_cv_score(LeaveOneOut())

    # LOGOCV (cohort hold-out) — leakage-free metrik
    f1_bin_g, acc_bin_g = _ovr_cv_score(LeaveOneGroupOut(), groups=COHORTS)

    print(f"  {short:<12}  LOOCV  F1={f1_bin:.3f} Acc={acc_bin:.3f}  | "
          f"LOGOCV F1={f1_bin_g:.3f} Acc={acc_bin_g:.3f}")

    ovr_rows.append({
        "group":            short,
        "f1_binary_loocv":  round(f1_bin, 4),
        "accuracy_loocv":   round(acc_bin, 4),
        "f1_binary_logocv": round(f1_bin_g, 4),
        "accuracy_logocv":  round(acc_bin_g, 4),
    })

    # SHAP — tüm veri üzerinde fit edilmiş modelden
    xgb_bin.fit(X, y_bin)
    # XGBoost >= 2.0 serializes base_score as a vector string like '[0.5]'.
    # SHAP's XGBTreeModelLoader does float(base_score) and crashes on the
    # brackets. Patching the JSON dump doesn't help (xgb keeps the vector
    # form internally), so we briefly wrap builtins.float during the
    # TreeExplainer call to strip brackets before conversion.
    import builtins as _builtins
    _orig_float = _builtins.float
    def _bracket_safe_float(x):
        if isinstance(x, str) and x.startswith('['):
            x = x.strip('[]').split(',')[0]
        return _orig_float(x)
    _builtins.float = _bracket_safe_float
    try:
        explainer = shap.TreeExplainer(xgb_bin.get_booster())
    finally:
        _builtins.float = _orig_float
    sv = explainer.shap_values(X)
    # ikili sınıf: sv şeklinde (n, features) veya list[2]
    if isinstance(sv, list):
        sv = sv[1]

    importance = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:TOP_N]
    top_feats = [FEATURE_COLS[j] for j in top_idx]
    top_vals  = importance[top_idx]

    # sıralı çubuk grafik (en önemli en üstte)
    order = np.argsort(top_vals)
    ax = axes[i]
    bars = ax.barh(range(TOP_N), top_vals[order], color=color, alpha=0.85)
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels([top_feats[j] for j in order], fontsize=9)
    ax.set_title(f"{short}  vs  Diğerleri\n"
                 f"LOOCV F1={f1_bin:.2f} | LOGOCV F1={f1_bin_g:.2f}",
                 fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Ort. |SHAP|", fontsize=9)
    ax.invert_yaxis()

    # değerleri çubuk ucuna yaz
    for bar, val in zip(bars, top_vals[order]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7)

fig.suptitle("One-vs-Rest SHAP — Her Grubun Ayırt Edici Özellikleri",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGS / "shap_ovr_groups.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  [OvR SHAP] shap_ovr_groups.png kaydedildi")

df_ovr = pd.DataFrame(ovr_rows)
df_ovr.to_csv(REPORTS / "ovr_binary_f1.csv", index=False)

# ── OvR LOOCV vs LOGOCV karşılaştırma çubuğu ─────────────────────────────────
fig_cmp, ax_cmp = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(df_ovr))
w = 0.38
ax_cmp.bar(x_pos - w/2, df_ovr["f1_binary_loocv"],  width=w,
           color="#4C72B0", label="LOOCV (denek)",  alpha=0.9)
ax_cmp.bar(x_pos + w/2, df_ovr["f1_binary_logocv"], width=w,
           color="#DD8452", label="LOGOCV (cohort)", alpha=0.9)
for j, (lo, lg) in enumerate(zip(df_ovr["f1_binary_loocv"],
                                 df_ovr["f1_binary_logocv"])):
    ax_cmp.text(j - w/2, lo + 0.015, f"{lo:.2f}", ha="center", fontsize=8)
    ax_cmp.text(j + w/2, lg + 0.015, f"{lg:.2f}", ha="center", fontsize=8)
ax_cmp.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="şans (binary 0.5)")
ax_cmp.set_xticks(x_pos)
ax_cmp.set_xticklabels(df_ovr["group"], fontsize=10)
ax_cmp.set_ylabel("F1 (binary)", fontsize=10)
ax_cmp.set_ylim(0, 1.05)
ax_cmp.set_title("One-vs-Rest — LOOCV vs LOGOCV (cohort hold-out)",
                 fontsize=12, fontweight="bold")
ax_cmp.legend(fontsize=9, loc="upper right")
ax_cmp.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_cmp.savefig(FIGS / "ovr_loocv_vs_logocv.png", dpi=150, bbox_inches="tight")
plt.close(fig_cmp)
print(f"  [OvR CV] ovr_loocv_vs_logocv.png kaydedildi")


# ── Final ablation: LightGBM × grooming_profile ──────────────────────────────
# Cohort hold-out altında bile yüksek F1 veren tek hedef-model çifti.
# SHAP ile hangi davranış özellikleri grooming_profile sınıflandırmasını
# sürüklüyor görselleştirilir. (Not: grooming_profile groom_pct_time
# eşiklemesinden türetildiği için SHAP'ın o feature'ı baskın bulması
# beklenir — bu kendisi de raporlanması gereken bir bulgudur.)

if HAS_LIGHTGBM:
    print(f"\n{'='*55}")
    print("FINAL ABLATION — LightGBM × grooming_profile")
    print(f"{'='*55}")

    y_groom = labels["grooming_profile"].values
    le_g = LabelEncoder()
    y_groom_enc = le_g.fit_transform(y_groom)
    class_names_g = list(le_g.classes_)

    lgb_final = LGBMClassifier(
        n_estimators=200, max_depth=3, num_leaves=7,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=2, random_state=42, verbosity=-1,
    )
    lgb_final.fit(X, y_groom_enc)

    explainer_g = shap.TreeExplainer(lgb_final)
    shap_vals_g = explainer_g.shap_values(X)
    # SHAP returns either list[k] of (n,p) or ndarray (n,p,k)
    if isinstance(shap_vals_g, list):
        sv_arr = np.stack(shap_vals_g, axis=-1)
    else:
        sv_arr = shap_vals_g
    if sv_arr.ndim == 2:  # (n,p) — binary edge case
        sv_arr = sv_arr[..., np.newaxis]

    mean_abs_per_class  = np.abs(sv_arr).mean(axis=0)         # (p, k)
    global_importance_g = mean_abs_per_class.mean(axis=1)     # (p,)

    TOP_N_GROOM = min(10, len(FEATURE_COLS))
    top_idx_g = np.argsort(global_importance_g)[::-1][:TOP_N_GROOM]

    rank_rows = []
    for rank, idx in enumerate(top_idx_g, 1):
        row = {
            "rank":                 rank,
            "feature":              FEATURE_COLS[idx],
            "global_mean_abs_shap": round(float(global_importance_g[idx]), 5),
        }
        for ci, cn in enumerate(class_names_g):
            row[f"shap_{cn}"] = round(float(mean_abs_per_class[idx, ci]), 5)
        rank_rows.append(row)
    pd.DataFrame(rank_rows).to_csv(
        REPORTS / "shap_lightgbm_grooming_top.csv", index=False,
    )

    n_cls_g = len(class_names_g)
    fig_lgb, axes_lgb = plt.subplots(
        1, n_cls_g + 1, figsize=(4 * (n_cls_g + 1), 5), sharey=True,
    )
    if n_cls_g + 1 == 1:
        axes_lgb = [axes_lgb]

    # Genel (tüm sınıfların ortalaması)
    feats_g = [FEATURE_COLS[i] for i in top_idx_g][::-1]
    ax = axes_lgb[0]
    vals_g_top = global_importance_g[top_idx_g][::-1]
    ax.barh(range(len(vals_g_top)), vals_g_top, color="#333333", alpha=0.85)
    ax.set_yticks(range(len(vals_g_top)))
    ax.set_yticklabels(feats_g, fontsize=9)
    ax.set_title("Genel (sınıf ortalaması)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Ort. |SHAP|", fontsize=9)

    class_colors_g = ["#4CAF50", "#FF9800", "#9C27B0"]
    for ci, (cn, ax_c) in enumerate(zip(class_names_g, axes_lgb[1:])):
        vals = mean_abs_per_class[top_idx_g, ci][::-1]
        ax_c.barh(range(len(vals)), vals,
                  color=class_colors_g[ci % len(class_colors_g)], alpha=0.85)
        ax_c.set_yticks(range(len(vals)))
        ax_c.set_title(f"{cn}", fontsize=10, fontweight="bold",
                       color=class_colors_g[ci % len(class_colors_g)])
        ax_c.set_xlabel("Ort. |SHAP|", fontsize=9)

    # Başlığa LOGOCV F1'i ekle (dürüst metrik)
    logocv_row = df_results[(df_results["target"] == "grooming_profile")
                            & (df_results["model"] == "LightGBM")
                            & (df_results["cv"] == "LOGOCV")]
    loocv_row  = df_results[(df_results["target"] == "grooming_profile")
                            & (df_results["model"] == "LightGBM")
                            & (df_results["cv"] == "LOOCV")]
    f1_logocv = float(logocv_row["f1_macro"].iloc[0]) if len(logocv_row) else float("nan")
    f1_loocv  = float(loocv_row["f1_macro"].iloc[0])  if len(loocv_row)  else float("nan")
    fig_lgb.suptitle(
        f"LightGBM × grooming_profile — SHAP özellik önemi  "
        f"(LOOCV F1={f1_loocv:.2f} | LOGOCV F1={f1_logocv:.2f})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_lgb.savefig(FIGS / "shap_lightgbm_grooming.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig_lgb)
    print(f"  [SHAP] shap_lightgbm_grooming.png + shap_lightgbm_grooming_top.csv")
else:
    print("\n[skip] LightGBM bulunamadı — `pip install lightgbm` ile final "
          "ablation üretilebilir.")
