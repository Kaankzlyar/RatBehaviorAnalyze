"""
Aşama 06 — ML Model Eğitimi (LOOCV)
Çıktılar:
  reports/model_comparison.csv
  reports/figures/confusion_<hedef>.png
  reports/figures/shap_<hedef>.png
  models/classifier/<model>_<hedef>.pkl
"""

import json
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
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

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

# ── Model tanımları ───────────────────────────────────────────────────────────

def make_models(n_classes):
    return {
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

# ── LOOCV eğitim fonksiyonu ───────────────────────────────────────────────────

def loocv_evaluate(model, X, y_enc, le, subject_ids):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    subjects_out = []
    n_classes = len(le.classes_)

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_enc[train_idx]

        # Eğitim setinde eksik sınıf varsa dummy satır ekle (XGBoost için)
        missing = set(range(n_classes)) - set(np.unique(y_tr))
        if missing:
            for cls in missing:
                X_tr = np.vstack([X_tr, X_tr[0:1]])
                y_tr = np.append(y_tr, cls)

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)[0]
        y_pred.append(pred)
        y_true.append(y_enc[test_idx][0])
        subjects_out.append(subject_ids[test_idx][0])

    all_labels = list(range(len(le.classes_)))
    f1  = f1_score(y_true, y_pred, average="macro",
                   labels=all_labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=all_labels)

    pred_labels = le.inverse_transform(y_pred)
    true_labels = le.inverse_transform(y_true)

    return f1, acc, cm, true_labels, pred_labels, subjects_out


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
        f1, acc, cm, true_lbl, pred_lbl, subs = loocv_evaluate(
            model, X, y_enc, le, SUBJECT_IDS
        )
        print(f"  {model_name:<15}  F1={f1:.3f}  Acc={acc:.3f}")

        results.append({
            "target":     target_name,
            "model":      model_name,
            "f1_macro":   round(f1, 4),
            "accuracy":   round(acc, 4),
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

        if f1 > best_f1:
            best_f1 = f1
            best_cm = cm

        # modeli kaydet
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
df_results.to_csv(REPORTS / "model_comparison.csv", index=False)

df_details = pd.DataFrame(loocv_details)
df_details.to_csv(REPORTS / "loocv_predictions.csv", index=False)

# ── Özet tablo yazdır ─────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("SONUÇ TABLOSU — F1 Macro (LOOCV)")
print(f"{'='*55}")
pivot = df_results.pivot(index="model", columns="target", values="f1_macro")
print(pivot.to_string())

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

    # LOOCV F1
    loo = LeaveOneOut()
    y_true_bin, y_pred_bin = [], []
    for tr_idx, te_idx in loo.split(X):
        n_pos_tr = y_bin[tr_idx].sum()
        n_neg_tr = len(y_bin[tr_idx]) - n_pos_tr
        spw_tr   = n_neg_tr / n_pos_tr if n_pos_tr > 0 else 1.0
        xgb_bin.set_params(scale_pos_weight=spw_tr)
        xgb_bin.fit(X[tr_idx], y_bin[tr_idx])
        y_pred_bin.append(xgb_bin.predict(X[te_idx])[0])
        y_true_bin.append(y_bin[te_idx][0])

    f1_bin = f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    acc_bin = accuracy_score(y_true_bin, y_pred_bin)
    print(f"  {short:<12}  F1={f1_bin:.3f}  Acc={acc_bin:.3f}")

    ovr_rows.append({"group": short, "f1_binary": round(f1_bin, 4),
                     "accuracy": round(acc_bin, 4)})

    # SHAP — tüm veri üzerinde fit edilmiş modelden
    xgb_bin.fit(X, y_bin)
    # XGBoost >= 2.0 stores base_score as '[0.5]' or '[a,b,c,d]' in the model
    # JSON; SHAP reads model params (not config), so patch via save_raw/load_model
    _booster = xgb_bin.get_booster()
    _raw = json.loads(_booster.save_raw('json'))
    _bs = _raw['learner']['learner_model_param']['base_score']
    if isinstance(_bs, str) and (_bs.startswith('[') or _bs.endswith(']')):
        _raw['learner']['learner_model_param']['base_score'] = _bs.strip('[]').split(',')[0]
        _booster.load_model(bytearray(json.dumps(_raw).encode()))
    explainer = shap.TreeExplainer(_booster)
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
    ax.set_title(f"{short}  vs  Diğerleri\n(LOOCV F1={f1_bin:.3f})",
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

pd.DataFrame(ovr_rows).to_csv(REPORTS / "ovr_binary_f1.csv", index=False)
