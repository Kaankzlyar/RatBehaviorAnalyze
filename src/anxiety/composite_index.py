"""
Composite Anxiety Index
-----------------------
anxiety_features.csv sütunlarından türetilen birleşik anksiyete skorları.
Tek değişken (rear_count) marjinal anlamlılıkta (p=0.045) — ratio/karma
skorların grup ayrımını güçlendirip güçlendirmediğini test eder.

Skorlar
-------
  AI_v1 = pct_periphery / (rear_count + 1)
          (Thigmotaxis arttıkça ↑, rearing arttıkça ↓ — kullanıcı önerisi)
  AI_v2 = (pct_periphery + pct_freeze) / (rear_count + 1)
          (Pasif anksiyete bileşeni — donma davranışı eklenir)
  AI_v3 = pct_periphery * (1 - rear_pct/100)
          (Multiplicative; iki güçlü sinyalin etkileşimi)

İstatistik
----------
  - Kruskal-Wallis (4 grup) — H, p, eta²
  - Mann-Whitney U (Control vs Treated) — U, p, rank-biserial r
  - Cohen's d (Control vs Treated)
  - Bonferroni düzeltmesi (3 skor için)

Çıktılar
--------
  reports/composite_anxiety_index.csv  — per-subject scores
  reports/composite_anxiety_stats.csv  — istatistik tablosu
  reports/figures/composite_anxiety_boxplot.png

Kullanım
--------
  python -m src.anxiety.composite_index
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent.parent
FEAT_CSV = ROOT / "data" / "anxiety_features.csv"
REPORTS  = ROOT / "reports"
FIGS     = REPORTS / "figures"
for d in (REPORTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

GROUP_COLORS = {
    "Control":              "#2196F3",
    "Aspartame":            "#F44336",
    "Grapefruit":           "#4CAF50",
    "Aspartame+Grapefruit": "#FF9800",
}

SCORES = ["AI_v1_thigmo_per_rear", "AI_v2_passive_per_rear", "AI_v3_thigmo_x_low_rear"]


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["subject_id", "cohort", "group"]].copy()
    rear = df["rear_count"].astype(float)
    thigmo = df["pct_periphery"].astype(float)
    freeze = df["pct_freeze"].astype(float)
    rear_pct = df["rear_pct"].astype(float)

    out["AI_v1_thigmo_per_rear"]    = thigmo / (rear + 1.0)
    out["AI_v2_passive_per_rear"]   = (thigmo + freeze) / (rear + 1.0)
    out["AI_v3_thigmo_x_low_rear"]  = thigmo * (1.0 - rear_pct / 100.0)
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if s2 <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(s2))


def stats_table(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    is_treated = df["group"].astype(str).str.lower() != "control"
    n_total = len(df)
    n_groups = df["group"].nunique()

    for col in score_cols:
        vals = df[col].dropna()
        if len(vals) < 5:
            continue

        # 4-grup Kruskal-Wallis
        groups = [df.loc[df["group"] == g, col].dropna().values
                  for g in df["group"].unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            H, p_kw = stats.kruskal(*groups)
            k = len(groups)
            eta2 = max(0.0, (H - k + 1) / (n_total - k))
        else:
            H, p_kw, eta2 = np.nan, np.nan, np.nan

        # Binary Mann-Whitney U
        ctrl = df.loc[~is_treated, col].dropna().values
        trt  = df.loc[ is_treated, col].dropna().values
        if len(ctrl) >= 2 and len(trt) >= 2:
            U, p_mw = stats.mannwhitneyu(trt, ctrl, alternative="two-sided")
            r_rb = 1.0 - (2.0 * U) / (len(ctrl) * len(trt))   # rank-biserial
            d = cohens_d(trt, ctrl)
        else:
            U, p_mw, r_rb, d = np.nan, np.nan, np.nan, np.nan

        rows.append({
            "score":         col,
            "kruskal_H":     round(H, 3) if not np.isnan(H) else np.nan,
            "kruskal_p":     round(p_kw, 4) if not np.isnan(p_kw) else np.nan,
            "eta2":          round(eta2, 3) if not np.isnan(eta2) else np.nan,
            "mw_U":          round(U, 1) if not np.isnan(U) else np.nan,
            "mw_p":          round(p_mw, 4) if not np.isnan(p_mw) else np.nan,
            "rank_biserial": round(r_rb, 3) if not np.isnan(r_rb) else np.nan,
            "cohens_d":      round(d, 3) if not np.isnan(d) else np.nan,
            "ctrl_median":   round(float(np.median(ctrl)), 3) if len(ctrl) else np.nan,
            "trt_median":    round(float(np.median(trt)), 3) if len(trt) else np.nan,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["mw_p_bonferroni"]      = (out["mw_p"]      * len(out)).clip(upper=1.0).round(4)
        out["kruskal_p_bonferroni"] = (out["kruskal_p"] * len(out)).clip(upper=1.0).round(4)
    return out


def plot_boxplots(df: pd.DataFrame, stats_df: pd.DataFrame, out: Path) -> None:
    score_cols = SCORES
    fig, axes = plt.subplots(1, len(score_cols), figsize=(5.0 * len(score_cols), 4.5))
    if len(score_cols) == 1:
        axes = [axes]

    groups = ["Control", "Aspartame", "Grapefruit", "Aspartame+Grapefruit"]
    groups = [g for g in groups if g in df["group"].unique()]

    for ax, col in zip(axes, score_cols):
        data = [df.loc[df["group"] == g, col].dropna().values for g in groups]
        bp = ax.boxplot(data, labels=[g.replace("Aspartame+Grapefruit", "ASP+G") for g in groups],
                        patch_artist=True, widths=0.55)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(GROUP_COLORS.get(g, "#888"))
            patch.set_alpha(0.65)

        # tek hayvan noktaları
        for i, (g, vals) in enumerate(zip(groups, data), start=1):
            if len(vals):
                ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.07, 0.07, len(vals)),
                           vals, color="black", s=14, zorder=3, alpha=0.75)

        row = stats_df[stats_df["score"] == col]
        if len(row):
            r = row.iloc[0]
            title = (f"{col}\nKW p={r.kruskal_p:.3g}  η²={r.eta2:.2f}\n"
                     f"MW p={r.mw_p:.3g}  d={r.cohens_d:.2f}")
        else:
            title = col
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Composite Anxiety Index — Group Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


def main() -> None:
    if not FEAT_CSV.exists():
        raise FileNotFoundError(f"{FEAT_CSV} yok — `python -m src.anxiety.profile` çalıştır.")
    df = pd.read_csv(FEAT_CSV)
    print(f"[load] {len(df)} hayvan, {df['group'].nunique()} grup")

    scores = compute_scores(df)
    out_csv = REPORTS / "composite_anxiety_index.csv"
    scores.to_csv(out_csv, index=False)
    print(f"[write] {out_csv.relative_to(ROOT)}")

    # rear_count ve pct_periphery'yi de tabloya ekle (baseline kıyas için)
    full = scores.merge(df[["subject_id", "rear_count", "pct_periphery", "pct_freeze"]],
                        on="subject_id")
    score_cols = SCORES + ["rear_count", "pct_periphery"]
    stats_df = stats_table(full, score_cols)

    stats_csv = REPORTS / "composite_anxiety_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"[write] {stats_csv.relative_to(ROOT)}")

    print("\n[stats — sorted by mw_p]")
    print(stats_df.sort_values("mw_p").to_string(index=False))

    plot_boxplots(full, stats_df, FIGS / "composite_anxiety_boxplot.png")
    print("\n[done]")


if __name__ == "__main__":
    main()
