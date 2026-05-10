"""
Anxiety Regression — PC1 as continuous "anxiety axis"
------------------------------------------------------
Hedef değişken: PCA'nın birinci bileşeni (PC1). Her fold'da PCA train
seti üzerinde **yeniden** fit edilir; test deneği train PCA üzerinden
projekte edilir → academik olarak savunulabilir held-out skoru.

İşaret tutarlılığı: her fold'da PC1 yönü `pct_periphery` (thigmotaksi)
yüklemesi pozitif kalacak şekilde sabitlenir; aksi takdirde PC1 ± rastgele
döner ve fold'lar arası R² aritmetiği kirlenir.

Asıl bilimsel bulgu: **gruplar PC1 üzerinde sıralanıyor mu?**
İkincil bulgu: tahmin edilebilirlik (R², Pearson r).

Çıktılar
--------
    reports/anxiety_regression_metrics.csv         — model × R²/RMSE/MAE/r
    reports/anxiety_regression_predictions.csv     — fold başına tahmin
    reports/anxiety_pc1_by_group.csv               — grup PC1 ortalama + bootstrap CI
    reports/figures/anxiety_regression_scatter.png — y_true vs y_pred (model başına)
    reports/figures/anxiety_pc1_ranking.png        — gruplar PC1 ekseninde
    models/anxiety_regression/{ridge,rf,svr}.pkl   — tam veriyle yeniden eğitim

Kullanım
--------
    python -m src.anxiety.regression
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
FEAT_CSV = ROOT / "data" / "anxiety_features.csv"
MODELS   = ROOT / "models" / "anxiety_regression"
REPORTS  = ROOT / "reports"
FIGS     = REPORTS / "figures"

for d in (MODELS, REPORTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

META_COLS    = {"subject_id", "cohort", "group", "pc1_score", "pc2_score"}
NA_THRESHOLD = 0.5
ANCHOR_FEATURE = "pct_periphery"   # PC1 işaret çapası (thigmotaksi pozitif)
RNG_SEED     = 42

GROUP_COLORS = {
    "Control":              "#2196F3",
    "Aspartame":            "#F44336",
    "Grapefruit":           "#4CAF50",
    "Aspartame+Grapefruit": "#FF9800",
}
_FALLBACK = ["#9C27B0", "#00BCD4", "#795548", "#607D8B"]


def make_models() -> dict:
    return {
        "Ridge": Ridge(alpha=1.0, random_state=RNG_SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=2,
            random_state=RNG_SEED, n_jobs=-1,
        ),
        "SVR-RBF": SVR(kernel="rbf", C=1.0, gamma="scale"),
    }


# ── Veri ──────────────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, np.ndarray, list[str], int]:
    if not FEAT_CSV.exists():
        raise FileNotFoundError(
            f"{FEAT_CSV} yok — önce `python -m src.anxiety.profile` çalıştır."
        )
    df = pd.read_csv(FEAT_CSV)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    na_frac = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if na_frac[c] < NA_THRESHOLD]

    if ANCHOR_FEATURE not in feature_cols:
        raise ValueError(f"PC1 anchor '{ANCHOR_FEATURE}' feature listesinde yok.")
    anchor_idx = feature_cols.index(ANCHOR_FEATURE)

    X = df[feature_cols].values.astype(float)
    print(f"[load] n={len(df)}  features={len(feature_cols)}  "
          f"anchor='{ANCHOR_FEATURE}' (idx={anchor_idx})")
    return df, X, feature_cols, anchor_idx


# ── LOOCV held-out PCA + regression ──────────────────────────────────────────

def heldout_pca_loocv(X: np.ndarray, model, anchor_idx: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    loo = LeaveOneOut()
    n = len(X)
    y_true = np.zeros(n)
    y_pred = np.zeros(n)

    for tr, te in loo.split(X):
        X_tr, X_te = X[tr], X[te]

        imp = SimpleImputer(strategy="median").fit(X_tr)
        X_tr_i, X_te_i = imp.transform(X_tr), imp.transform(X_te)

        sc = StandardScaler().fit(X_tr_i)
        X_tr_s, X_te_s = sc.transform(X_tr_i), sc.transform(X_te_i)

        pca = PCA(n_components=2).fit(X_tr_s)
        sign = 1.0 if pca.components_[0, anchor_idx] >= 0 else -1.0

        y_tr_pc1 = sign * pca.transform(X_tr_s)[:, 0]
        y_te_pc1 = sign * pca.transform(X_te_s)[:, 0]

        m = model.__class__(**model.get_params())
        m.fit(X_tr_s, y_tr_pc1)
        y_true[te] = y_te_pc1
        y_pred[te] = m.predict(X_te_s)

    return y_true, y_pred


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, p = stats.pearsonr(y_true, y_pred)
    return {
        "r2":         round(r2_score(y_true, y_pred), 3),
        "rmse":       round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae":        round(mean_absolute_error(y_true, y_pred), 3),
        "pearson_r":  round(float(r), 3),
        "pearson_p":  round(float(p), 4),
    }


# ── Group-level PC1 analizi (asıl yayın bulgusu) ─────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                 ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def group_pc1_table(df: pd.DataFrame) -> pd.DataFrame:
    if "pc1_score" not in df.columns:
        raise ValueError("anxiety_features.csv 'pc1_score' içermiyor — "
                         "profile.py'i tekrar çalıştır.")
    rows = []
    for g, sub in df.groupby("group"):
        vals = sub["pc1_score"].dropna().values
        if len(vals) < 2:
            continue
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "group":  g,
            "n":      len(vals),
            "mean":   round(float(vals.mean()), 3),
            "median": round(float(np.median(vals)), 3),
            "sd":     round(float(vals.std(ddof=1)), 3),
            "ci_lo":  round(lo, 3),
            "ci_hi":  round(hi, 3),
        })
    out = pd.DataFrame(rows).sort_values("mean", ascending=False)

    # Kruskal-Wallis on PC1 across groups
    samples = [df.loc[df["group"] == g, "pc1_score"].dropna().values
               for g in out["group"]]
    if len(samples) >= 2 and all(len(s) >= 2 for s in samples):
        h, p = stats.kruskal(*samples)
        n_total = sum(len(s) for s in samples)
        k = len(samples)
        eta2 = max(0.0, (h - k + 1) / (n_total - k)) if n_total > k else 0.0
        out.attrs["kw_H"]    = round(float(h), 3)
        out.attrs["kw_p"]    = round(float(p), 4)
        out.attrs["kw_eta2"] = round(float(eta2), 3)
    return out


# ── Görselleştirme ───────────────────────────────────────────────────────────

def plot_regression_scatter(results: dict, out: Path,
                            df: pd.DataFrame) -> None:
    color_map = dict(GROUP_COLORS)
    for i, g in enumerate(df["group"].unique()):
        if g not in color_map:
            color_map[g] = _FALLBACK[i % len(_FALLBACK)]

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        for g in df["group"].unique():
            mask = (df["group"] == g).values
            ax.scatter(r["y_true"][mask], r["y_pred"][mask],
                       label=g, color=color_map[g], s=70, alpha=0.85,
                       edgecolors="white", linewidth=0.5)
        # y=x reference
        lo = min(r["y_true"].min(), r["y_pred"].min())
        hi = max(r["y_true"].max(), r["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
        ax.set_xlabel("Held-out PC1 (gerçek)")
        ax.set_ylabel("Predicted PC1")
        m = r["metrics"]
        ax.set_title(f"{name}\nR²={m['r2']:.2f}  r={m['pearson_r']:.2f} "
                     f"(p={m['pearson_p']:.3f})", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("LOOCV Regression — PC1 anxiety axis (held-out PCA)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


def plot_pc1_ranking(df: pd.DataFrame, group_df: pd.DataFrame,
                     out: Path) -> None:
    color_map = dict(GROUP_COLORS)
    for i, g in enumerate(df["group"].unique()):
        if g not in color_map:
            color_map[g] = _FALLBACK[i % len(_FALLBACK)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sol panel: per-subject PC1, PC1'e göre sıralı
    ax = axes[0]
    sub = df.dropna(subset=["pc1_score"]).sort_values("pc1_score")
    colors = [color_map.get(g, "#888") for g in sub["group"]]
    ax.barh(np.arange(len(sub)), sub["pc1_score"].values,
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["subject_id"].values, fontsize=7)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("PC1 (anxiety axis →)")
    ax.set_title("Hayvan başına PC1 — sıralı", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map.get(g, "#888"))
               for g in df["group"].unique()]
    ax.legend(handles, list(df["group"].unique()),
              fontsize=8, loc="best", framealpha=0.7)

    # Sağ panel: grup ortalama + bootstrap %95 CI
    ax = axes[1]
    yp = np.arange(len(group_df))
    means  = group_df["mean"].values
    los    = means - group_df["ci_lo"].values
    his    = group_df["ci_hi"].values - means
    bar_colors = [color_map.get(g, "#888") for g in group_df["group"]]
    ax.barh(yp, means, xerr=[los, his], color=bar_colors,
            alpha=0.85, capsize=6, edgecolor="white",
            error_kw=dict(ecolor="black", lw=1.2))
    ax.set_yticks(yp)
    ax.set_yticklabels(group_df["group"].values, fontsize=9)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("PC1 ortalama (bootstrap %95 CI)")
    title = "Grup PC1 sıralaması"
    if "kw_p" in group_df.attrs:
        title += (f"\nKruskal-Wallis H={group_df.attrs['kw_H']}  "
                  f"p={group_df.attrs['kw_p']}  η²={group_df.attrs['kw_eta2']}")
    ax.set_title(title, fontsize=10)

    fig.suptitle("Anxiety Axis (PC1) — Group Ranking",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


# ── Tam veri ile yeniden eğitim ──────────────────────────────────────────────

def fit_full_and_save(X: np.ndarray, feature_cols: list[str],
                      anchor_idx: int) -> None:
    imp = SimpleImputer(strategy="median").fit(X)
    X_i = imp.transform(X)
    sc  = StandardScaler().fit(X_i)
    X_s = sc.transform(X_i)

    pca = PCA(n_components=2).fit(X_s)
    sign = 1.0 if pca.components_[0, anchor_idx] >= 0 else -1.0
    y    = sign * pca.transform(X_s)[:, 0]

    with open(MODELS / "scaler.pkl", "wb") as f:
        pickle.dump({"imputer": imp, "scaler": sc, "pca": pca,
                     "sign": sign, "features": feature_cols}, f)
    print(f"[write] models/anxiety_regression/scaler.pkl")

    for name, model in make_models().items():
        m = model.__class__(**model.get_params())
        m.fit(X_s, y)
        slug = {"Ridge": "ridge", "RandomForest": "rf", "SVR-RBF": "svr"}[name]
        with open(MODELS / f"{slug}.pkl", "wb") as f:
            pickle.dump(m, f)
        print(f"[write] models/anxiety_regression/{slug}.pkl")


# ── Ana ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df, X, feature_cols, anchor_idx = load_features()

    print("\n[loocv] held-out PCA + regression...")
    results: dict[str, dict] = {}
    for name, model in make_models().items():
        y_true, y_pred = heldout_pca_loocv(X, model, anchor_idx)
        metrics = evaluate_regression(y_true, y_pred)
        results[name] = {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}
        print(f"  {name:<14s}  R²={metrics['r2']:.2f}  "
              f"r={metrics['pearson_r']:.2f} (p={metrics['pearson_p']:.3f})  "
              f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}")

    metrics_rows = [{"model": n, **r["metrics"]} for n, r in results.items()]
    metrics_path = REPORTS / "anxiety_regression_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"[write] {metrics_path.relative_to(ROOT)}")

    pred_rows = []
    for name, r in results.items():
        for sid, true, pred in zip(df["subject_id"], r["y_true"], r["y_pred"]):
            pred_rows.append({"subject_id": sid, "model": name,
                              "pc1_true_heldout": round(float(true), 3),
                              "pc1_pred":         round(float(pred), 3)})
    pred_path = REPORTS / "anxiety_regression_predictions.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"[write] {pred_path.relative_to(ROOT)}")

    print("\n[group-rank] PC1 üzerinde grup analizi...")
    group_df = group_pc1_table(df)
    group_path = REPORTS / "anxiety_pc1_by_group.csv"
    group_df.to_csv(group_path, index=False)
    print(group_df.to_string(index=False))
    if "kw_p" in group_df.attrs:
        print(f"  Kruskal-Wallis: H={group_df.attrs['kw_H']}  "
              f"p={group_df.attrs['kw_p']}  η²={group_df.attrs['kw_eta2']}")
    print(f"[write] {group_path.relative_to(ROOT)}")

    plot_regression_scatter(results, FIGS / "anxiety_regression_scatter.png", df)
    plot_pc1_ranking(df, group_df, FIGS / "anxiety_pc1_ranking.png")

    print("\n[refit] tam veri üzerinde yeniden eğitim + pickle...")
    fit_full_and_save(X, feature_cols, anchor_idx)

    print("\n[done]")


if __name__ == "__main__":
    main()
