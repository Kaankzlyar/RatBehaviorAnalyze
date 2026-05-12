"""
Aşama 06B — OFT + Plus Maze Combined Feature ML Eğitimi
--------------------------------------------------------
Girdi:  data/features/features_combined_normalized.csv  (12 satir, 46 ozellik)
        data/features/labels.csv                        (group/anxiety/rearing/grooming)

Cikti:
  reports/combined_model_comparison.csv
  reports/combined_loocv_predictions.csv
  reports/combined_ovr_binary_f1.csv
  reports/figures/combined/combined_confusion_<hedef>.png
  reports/figures/combined/combined_shap_<hedef>.png
  reports/figures/combined/combined_shap_ovr_groups.png
  reports/figures/combined/combined_cv_comparison.png
  reports/figures/combined/combined_vs_oft_comparison.png
  models/classifier/combined_<model>_<hedef>.pkl

Kullanim:
  python src/train_combined.py
"""
from __future__ import annotations

import pathlib
import pickle
import warnings
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"]        = "Calibri"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Dizinler ──────────────────────────────────────────────────────────────────

ROOT    = pathlib.Path(__file__).parent.parent
FEAT    = ROOT / "data" / "features"
MODELS  = ROOT / "models" / "classifier"
REPORTS = ROOT / "reports"
FIGS    = REPORTS / "figures" / "combined"

for d in [MODELS, REPORTS, FIGS]:
    d.mkdir(parents=True, exist_ok=True)

PREFIX = "combined_"   # tüm çıktı dosyalarına ön ek

# ── Veri yükle ────────────────────────────────────────────────────────────────

feat_norm = pd.read_csv(FEAT / "features_combined_normalized.csv")
labels_all = pd.read_csv(FEAT / "labels.csv")

# Sadece combined'da olan 12 sıçanı al (inner join)
# labels'dan yalnizca etiket sutunlarini al, group features'da zaten var
labels_cols = ["subject_id", "anxiety_level", "rearing_profile", "grooming_profile"]
# features'da group yoksa labels'dan da al
if "group" not in feat_norm.columns:
    labels_cols.append("group")
df = feat_norm.merge(labels_all[labels_cols], on="subject_id", how="inner")
print(f"[load] {len(df)} sıçan combined feature + etiket")
print(f"       Kohortlar: {sorted(df['cohort'].unique().tolist())}")

ID_COLS      = {"subject_id", "cohort", "group", "session_duration_s",
                "anxiety_level", "rearing_profile", "grooming_profile"}

# Sadece numeric ve ID olmayan sutunlari al
FEATURE_COLS = [
    c for c in df.columns
    if c not in ID_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]

X_raw = df[FEATURE_COLS].values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X_raw)

SUBJECT_IDS = df["subject_id"].values
COHORTS     = df["cohort"].values

TARGETS = {
    "group":            df["group"].values,
    "anxiety_level":    df["anxiety_level"].values,
    "rearing_profile":  df["rearing_profile"].values,
    "grooming_profile": df["grooming_profile"].values,
}

GROUP_ORDER = {
    "group":            ["Control", "Aspartame", "Grapefruit", "Aspartame+Grapefruit"],
    "anxiety_level":    ["low", "moderate", "high"],
    "rearing_profile":  ["low", "moderate", "high"],
    "grooming_profile": ["low", "moderate", "high"],
}

# pm_* feature'lari hicbir OFT etiketini uretmek icin kullanilmadi —
# bu yuzden leakage_map degismez, sadece OFT kokenli leakage'lar ayni.
LEAKAGE_MAP = {
    "anxiety_level": {
        "pct_time_periphery", "pct_time_freeze",
        "center_zone_entries", "pct_time_center",
        "total_freeze_s", "freeze_bout_count",
    },
    "rearing_profile": {
        "rear_pct_time", "rear_total_s",
        "rear_per_min", "rear_bout_count",
    },
    "grooming_profile": {
        "groom_pct_time", "groom_total_s",
        "groom_per_min", "groom_bout_count",
    },
    "group": set(),
}

# ── Model tanımlari ───────────────────────────────────────────────────────────

def make_models(n_features: int, n_classes: int) -> dict:
    k_top = min(8, n_features)
    models: dict = {
        "LogisticReg": LogisticRegression(
            penalty="l1", solver="saga", C=0.5,
            max_iter=2000, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=3,
            min_samples_leaf=2, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=2,
            learning_rate=0.05, subsample=0.8,
            eval_metric="mlogloss",
            random_state=42, verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=42,
        ),
        "LogReg_L1_MI8": Pipeline([
            ("select", SelectKBest(
                partial(mutual_info_classif, random_state=42),
                k=k_top,
            )),
            ("clf", LogisticRegression(
                penalty="l1", solver="saga", C=0.1,
                max_iter=4000, random_state=42,
            )),
        ]),
    }
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=3, num_leaves=7,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=2, random_state=42, verbosity=-1,
        )
    return models


# ── CV değerlendirme ──────────────────────────────────────────────────────────

def cv_evaluate(model, X, y_enc, le, subject_ids, splitter, groups=None):
    y_true, y_pred, subjects_out = [], [], []
    n_classes = len(le.classes_)

    split_args = (X, y_enc, groups) if groups is not None else (X, y_enc)
    for train_idx, test_idx in splitter.split(*split_args):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_enc[train_idx]

        missing = set(range(n_classes)) - set(np.unique(y_tr))
        if missing:
            for cls in missing:
                X_tr = np.vstack([X_tr, X_tr[0:1]])
                y_tr = np.append(y_tr, cls)

        model.fit(X_tr, y_tr)
        y_pred.extend(model.predict(X_te).tolist())
        y_true.extend(y_enc[test_idx].tolist())
        subjects_out.extend(subject_ids[test_idx].tolist())

    all_labels = list(range(n_classes))
    f1  = f1_score(y_true, y_pred, average="macro",
                   labels=all_labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=all_labels)
    return f1, acc, cm, le.inverse_transform(y_true), le.inverse_transform(y_pred), subjects_out


# ── Görsel yardımcıları ───────────────────────────────────────────────────────

def plot_confusion(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 1.4),
                                    max(3, len(class_names) * 1.2)))
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
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=10)
    ax.set_xlabel("Tahmin", fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap(model, X_in, feature_names, class_names, target_name, save_path):
    # XGBoost multi-class: direkt model kullan (get_booster() base_score vektoru kiritiyor)
    explainer = shap.TreeExplainer(model)

    sv = explainer.shap_values(X_in)
    n_cls = len(class_names)

    if n_cls > 2:
        mean_abs = np.mean([np.abs(sv[c]) for c in range(n_cls)], axis=0)
        top_idx  = np.argsort(mean_abs.mean(axis=0))[::-1][:12]
        fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 5), sharey=True)
        if n_cls == 1:
            axes = [axes]
        for c, ax in enumerate(axes):
            vals  = sv[c][:, top_idx]
            feats = [feature_names[i] for i in top_idx]
            means = np.abs(vals).mean(axis=0)
            order = np.argsort(means)[::-1]
            ax.barh(range(len(order)), means[order], color="#2196F3", alpha=0.8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([feats[i] for i in order], fontsize=8)
            ax.set_title(class_names[c].replace("Aspartame+Grapefruit", "ASP+GF"),
                         fontsize=10)
            ax.set_xlabel("Ort. |SHAP|", fontsize=8)
            ax.invert_yaxis()
        fig.suptitle(f"SHAP — {target_name} [combined]", fontsize=12)
    else:
        sv_bin = sv if not isinstance(sv, list) else sv[1]
        top_idx = np.argsort(np.abs(sv_bin).mean(axis=0))[::-1][:12]
        fig, ax = plt.subplots(figsize=(6, 5))
        feats = [feature_names[i] for i in top_idx]
        means = np.abs(sv_bin).mean(axis=0)[top_idx]
        order = np.argsort(means)
        ax.barh(range(len(order)), means[order], color="#2196F3", alpha=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feats[i] for i in order], fontsize=9)
        ax.set_title(f"SHAP — {target_name} [combined]", fontsize=11)
        ax.set_xlabel("Ort. |SHAP|", fontsize=9)
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Ana eğitim döngüsü ────────────────────────────────────────────────────────

results       = []
loocv_details = []

for target_name, y_raw in TARGETS.items():
    print(f"\n{'='*55}")
    print(f"Hedef: {target_name}  [combined]")
    print(f"{'='*55}")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_cls = len(le.classes_)

    leakage   = LEAKAGE_MAP.get(target_name, set())
    clean_cols = [c for c in FEATURE_COLS if c not in leakage]
    clean_idx  = [FEATURE_COLS.index(c) for c in clean_cols]
    X_t        = X[:, clean_idx]

    # pm_* feature sayisini goster
    pm_in_clean = [c for c in clean_cols if c.startswith("pm_")]
    print(f"  Toplam feature: {len(clean_cols)}  "
          f"(OFT: {len(clean_cols)-len(pm_in_clean)}  |  pm_*: {len(pm_in_clean)})")
    if leakage:
        print(f"  Leakage çıkarıldı: {sorted(leakage & set(FEATURE_COLS))}")

    models   = make_models(len(clean_cols), n_cls)
    best_f1  = -1
    best_cm  = None
    xgb_final = None

    for model_name, model in models.items():
        # LOOCV
        f1_loo, acc_loo, cm_loo, true_l, pred_l, subs = cv_evaluate(
            model, X_t, y_enc, le, SUBJECT_IDS, LeaveOneOut()
        )
        print(f"  {model_name:<15}  LOOCV   F1={f1_loo:.3f}  Acc={acc_loo:.3f}")
        results.append({
            "target": target_name, "model": model_name, "cv": "LOOCV",
            "f1_macro": round(f1_loo, 4), "accuracy": round(acc_loo, 4),
            "n_classes": n_cls, "leakage_removed": len(leakage) > 0,
        })
        for sid, tl, pl in zip(subs, true_l, pred_l):
            loocv_details.append({
                "target": target_name, "model": model_name,
                "subject_id": sid, "true": tl, "predicted": pl, "correct": tl == pl,
            })

        # LOGOCV
        f1_g, acc_g, *_ = cv_evaluate(
            model, X_t, y_enc, le, SUBJECT_IDS,
            LeaveOneGroupOut(), groups=COHORTS,
        )
        print(f"  {' '*15}  LOGOCV  F1={f1_g:.3f}  Acc={acc_g:.3f}")
        results.append({
            "target": target_name, "model": model_name, "cv": "LOGOCV",
            "f1_macro": round(f1_g, 4), "accuracy": round(acc_g, 4),
            "n_classes": n_cls, "leakage_removed": len(leakage) > 0,
        })

        if f1_loo > best_f1:
            best_f1 = f1_loo
            best_cm = cm_loo

        # Tum veriyle kaydet
        model.fit(X_t, y_enc)
        pkl = MODELS / f"combined_{model_name.lower()}_{target_name}.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump({"model": model, "label_encoder": le,
                         "feature_cols": clean_cols}, fh)

        if model_name == "XGBoost":
            xgb_final = model

    plot_confusion(
        best_cm, list(le.classes_),
        title=f"Combined — {target_name} (F1={best_f1:.3f})",
        save_path=FIGS / f"{PREFIX}confusion_{target_name}.png",
    )

    if xgb_final is not None:
        try:
            plot_shap(
                xgb_final, X_t, clean_cols, list(le.classes_),
                target_name, FIGS / f"{PREFIX}shap_{target_name}.png",
            )
            print(f"  [SHAP] {PREFIX}shap_{target_name}.png kaydedildi")
        except Exception as e:
            print(f"  [SHAP hata] {e}")

# ── CSV çıktıları ─────────────────────────────────────────────────────────────

df_res = pd.DataFrame(results)
df_res.to_csv(REPORTS / f"{PREFIX}model_comparison.csv", index=False)
pd.DataFrame(loocv_details).to_csv(REPORTS / f"{PREFIX}loocv_predictions.csv", index=False)

# ── CV karşılaştırma grafiği ──────────────────────────────────────────────────

target_list = list(TARGETS.keys())
fig, axes   = plt.subplots(2, 2, figsize=(15, 10))
axes        = axes.flatten()
bar_colors  = {"LOOCV": "#4C72B0", "LOGOCV": "#DD8452"}
bar_width   = 0.38

for ax, target in zip(axes, target_list):
    df_t  = df_res[df_res["target"] == target]
    pivot = df_t.pivot(index="model", columns="cv", values="f1_macro")
    cv_cols = [c for c in ["LOOCV", "LOGOCV"] if c in pivot.columns]
    pivot   = pivot[cv_cols]
    x       = np.arange(len(pivot))

    for j, cv_name in enumerate(cv_cols):
        offset = (j - (len(cv_cols) - 1) / 2) * bar_width
        vals   = pivot[cv_name].values
        bars   = ax.bar(x + offset, vals, width=bar_width,
                        color=bar_colors[cv_name], label=cv_name, alpha=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    n_cls  = int(df_t["n_classes"].iloc[0])
    ax.axhline(1 / n_cls, color="gray", linestyle="--",
               linewidth=1, alpha=0.7, label=f"şans (1/{n_cls})")
    ax.set_xticks(x)
    ax.set_xticklabels(list(pivot.index), rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Macro", fontsize=10)
    ax.set_title(f"{target} [combined]", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Combined (OFT + Plus Maze) — CV Stratejisi Karşılaştırması",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGS / f"{PREFIX}cv_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] {PREFIX}cv_comparison.png")

# ── OvR SHAP — grup bazlı ayırt edici özellikler ─────────────────────────────

print(f"\n{'='*55}")
print("ONE-VS-REST SHAP [combined] — pm_* özelliklerinin katkısı")
print(f"{'='*55}")

groups_arr    = df["group"].values
unique_groups = ["Control", "Aspartame", "Grapefruit", "Aspartame+Grapefruit"]
group_short   = ["Control", "Aspartame", "Grapefruit", "ASP+GF"]
colors        = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
TOP_N         = 10   # combined'da daha fazla feature var, top 10 daha informatif

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes      = axes.flatten()
ovr_rows  = []

for i, (grp, short, color) in enumerate(zip(unique_groups, group_short, colors)):
    y_bin = (groups_arr == grp).astype(int)
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    spw   = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_bin = XGBClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, verbosity=0,
    )

    # LOOCV F1 (binary)
    y_true_bin, y_pred_bin = [], []
    for tr_idx, te_idx in LeaveOneOut().split(X):
        n_pos_tr = y_bin[tr_idx].sum()
        n_neg_tr = len(y_bin[tr_idx]) - n_pos_tr
        xgb_bin.set_params(scale_pos_weight=n_neg_tr / n_pos_tr
                           if n_pos_tr > 0 else 1.0)
        xgb_bin.fit(X[tr_idx], y_bin[tr_idx])
        y_pred_bin.append(xgb_bin.predict(X[te_idx])[0])
        y_true_bin.append(y_bin[te_idx][0])

    f1_bin  = f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    acc_bin = accuracy_score(y_true_bin, y_pred_bin)
    print(f"  {short:<12}  F1={f1_bin:.3f}  Acc={acc_bin:.3f}")
    ovr_rows.append({"group": short, "f1_binary": round(f1_bin, 4),
                     "accuracy": round(acc_bin, 4)})

    # SHAP — tüm veri fit
    xgb_bin.fit(X, y_bin)
    import builtins as _b
    _orig = _b.float
    def _safe(x):
        if isinstance(x, str) and x.startswith("["):
            x = x.strip("[]").split(",")[0]
        return _orig(x)
    _b.float = _safe
    try:
        explainer = shap.TreeExplainer(xgb_bin.get_booster())
    finally:
        _b.float = _orig
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]

    importance = np.abs(sv).mean(axis=0)
    top_idx    = np.argsort(importance)[::-1][:TOP_N]
    top_feats  = [FEATURE_COLS[j] for j in top_idx]
    top_vals   = importance[top_idx]

    order = np.argsort(top_vals)
    ax    = axes[i]
    bars  = ax.barh(range(TOP_N), top_vals[order], color=color, alpha=0.85)
    ax.set_yticks(range(TOP_N))

    # pm_* feature'larini bold yap
    feat_labels = [top_feats[j] for j in order]
    ax.set_yticklabels(feat_labels, fontsize=9,
                       fontweight="bold" if False else "normal")  # tum normal
    # pm_* satirlarini kirmizi ile isaretlemek icin tick label renklerini degistir
    for tick, lbl in zip(ax.get_yticklabels(), feat_labels):
        tick.set_color("#B71C1C" if lbl.startswith("pm_") else "black")
        tick.set_fontweight("bold" if lbl.startswith("pm_") else "normal")

    ax.set_title(f"{short}  vs  Diğerleri\n(LOOCV F1={f1_bin:.3f})",
                 fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Ort. |SHAP|", fontsize=9)
    ax.invert_yaxis()
    for bar, val in zip(bars, top_vals[order]):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

fig.suptitle(
    "One-vs-Rest SHAP [OFT + Plus Maze Combined]\n"
    "Kırmızı etiket = pm_* (Plus Maze) özelliği",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
fig.savefig(FIGS / f"{PREFIX}shap_ovr_groups.png", dpi=150, bbox_inches="tight")
plt.close(fig)
pd.DataFrame(ovr_rows).to_csv(REPORTS / f"{PREFIX}ovr_binary_f1.csv", index=False)
print(f"  [OvR SHAP] {PREFIX}shap_ovr_groups.png kaydedildi")

# ── OFT-only vs Combined karşılaştırma grafiği ───────────────────────────────

oft_path = REPORTS / "model_comparison_all.csv"
if oft_path.exists():
    print(f"\n{'='*55}")
    print("OFT-only vs Combined karşılaştırması")
    print(f"{'='*55}")

    oft_df  = pd.read_csv(oft_path)
    comb_df = df_res.copy()
    oft_df["dataset"]  = "OFT-only"
    comb_df["dataset"] = "OFT+PM"

    compare = pd.concat([oft_df, comb_df], ignore_index=True)
    compare = compare[compare["cv"] == "LOOCV"]

    # Sadece XGBoost — en sık kullanılan karşılaştırma modeli
    xgb_compare = compare[compare["model"] == "XGBoost"]

    target_list  = list(TARGETS.keys())
    fig, axes    = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    dataset_cols = {"OFT-only": "#607D8B", "OFT+PM": "#E91E63"}

    for ax, tgt in zip(axes, target_list):
        sub  = xgb_compare[xgb_compare["target"] == tgt]
        vals = {row["dataset"]: row["f1_macro"] for _, row in sub.iterrows()}
        dsets = ["OFT-only", "OFT+PM"]
        ys    = [vals.get(d, np.nan) for d in dsets]
        cols  = [dataset_cols[d] for d in dsets]
        bars  = ax.bar(dsets, ys, color=cols, alpha=0.85, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, ys):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        n_cls  = int(sub["n_classes"].iloc[0]) if len(sub) else 3
        ax.axhline(1 / n_cls, color="gray", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"şans (1/{n_cls})")
        ax.set_ylim(0, 1.05)
        ax.set_title(tgt, fontsize=11, fontweight="bold")
        ax.set_ylabel("F1 Macro (LOOCV)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        delta = vals.get("OFT+PM", np.nan) - vals.get("OFT-only", np.nan)
        if not np.isnan(delta):
            sign  = "+" if delta >= 0 else ""
            color = "#1B5E20" if delta > 0.02 else ("#B71C1C" if delta < -0.02 else "#555")
            ax.text(0.5, 0.92, f"Δ = {sign}{delta:.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, fontweight="bold", color=color)

    fig.suptitle(
        "XGBoost LOOCV F1 — OFT-only vs OFT + Plus Maze\n"
        "Δ = Combined − OFT-only  (yeşil ↑ iyileşme, kırmızı ↓ gerileme)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGS / f"{PREFIX}vs_oft_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {PREFIX}vs_oft_comparison.png kaydedildi")

    # Sayısal özet
    print("\nXGBoost LOOCV  OFT-only -> OFT+PM (F1 Macro):")
    for tgt in target_list:
        sub  = xgb_compare[xgb_compare["target"] == tgt]
        vals = {row["dataset"]: row["f1_macro"] for _, row in sub.iterrows()}
        oft_f1  = vals.get("OFT-only", float("nan"))
        comb_f1 = vals.get("OFT+PM",   float("nan"))
        delta   = comb_f1 - oft_f1 if not np.isnan(oft_f1 + comb_f1) else float("nan")
        sign    = "+" if delta >= 0 else ""
        print(f"  {tgt:<20}  {oft_f1:.3f} -> {comb_f1:.3f}  ({sign}{delta:.3f})")

# ── Özet ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("TAMAMLANDI")
print(f"{'='*55}")
print(f"  Raporlar : {REPORTS}")
print(f"  Gorseller: {FIGS}")
print(f"  Modeller : {MODELS}")
print(f"\n  Tez icin oncelikli gorseller:")
print(f"  >> {PREFIX}shap_ovr_groups.png   (pm_* katkisi - kirmizi etiketler)")
print(f"  >> {PREFIX}vs_oft_comparison.png  (OFT-only vs Combined delta)")
print(f"  >> {PREFIX}cv_comparison.png      (LOOCV/LOGOCV F1 tablosu)")
