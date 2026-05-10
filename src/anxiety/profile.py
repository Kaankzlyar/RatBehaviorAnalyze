"""
Anxiety Profile Builder
-----------------------
29 hayvan × N özellik matrisi oluşturur, PCA görselleştirir ve
Kruskal-Wallis + Dunn post-hoc ile grup karşılaştırması yapar.

Kaynaklar
---------
    data/DLCfiltered/**/*_oft_metrics.csv    (locomotion, thigmotaxis, freeze)
    data/DLCfiltered/**/*_behavior_bouts.csv (rearing/grooming bouts)

Çıktılar
--------
    data/anxiety_features.csv       — 29×N feature matrix
    reports/anxiety_pca.png         — PCA biplot (grup rengi)
    reports/anxiety_stats.csv       — Kruskal-Wallis p-değerleri + eta²
    reports/anxiety_boxplots.png    — her feature için grup kutu grafiği

Kullanım
--------
    python -m src.anxiety.profile
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT     = Path(__file__).resolve().parent.parent.parent
DLC_DIR  = ROOT / "data" / "DLCfiltered"
REPORTS  = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

EARLY_CUTOFF_S = 90.0   # ilk 90 saniye = "erken faz"

GROUP_COLORS = {
    "Control":           "#2196F3",
    "Aspartame":         "#F44336",
    "Grapefruit":        "#4CAF50",
    "Aspartame+Grapefruit": "#FF9800",
}
# fallback renk paleti (bilinmeyen grup adları için)
_FALLBACK = ["#9C27B0", "#00BCD4", "#795548", "#607D8B"]


# ── Veri yükleme ─────────────────────────────────────────────────────────────

def load_all_metrics() -> pd.DataFrame:
    parts = []
    for p in sorted(DLC_DIR.glob("**/*_oft_metrics.csv")):
        try:
            parts.append(pd.read_csv(p))
        except Exception as e:
            print(f"  [warn] {p.name}: {e}")
    if not parts:
        raise FileNotFoundError("Hiç *_oft_metrics.csv bulunamadı.")
    df = pd.concat(parts, ignore_index=True)
    print(f"[load] oft_metrics: {len(df)} hayvan")
    return df


def bout_extra_features(subject_id: str, session_s: float) -> dict:
    """behavior_bouts.csv'den ek özellikler hesaplar."""
    matches = list(DLC_DIR.glob(f"*/{subject_id.replace('_', '*', 0)}"
                                f"/*{subject_id}*_behavior_bouts.csv"))
    # glob with OpenField prefix
    if not matches:
        matches = list(DLC_DIR.glob(f"*/OpenField{subject_id}/*_behavior_bouts.csv"))
    if not matches:
        return {}

    bouts = pd.read_csv(matches[0])
    feats: dict = {}
    early = EARLY_CUTOFF_S

    for beh in ("rearing", "grooming"):
        sub = bouts[bouts["behaviour"] == beh]
        if len(sub) == 0:
            feats[f"{beh[:4]}_mean_bout_s"] = 0.0
            feats[f"{beh[:4]}_early_frac"]  = np.nan
            feats[f"{beh[:4]}_bout_cv"]     = np.nan
            continue
        dur = sub["duration_s"].to_numpy()
        feats[f"{beh[:4]}_mean_bout_s"] = float(dur.mean())
        feats[f"{beh[:4]}_bout_cv"]     = float(dur.std() / dur.mean()
                                                 if dur.mean() > 0 else np.nan)
        early_s = sub[sub["start_s"] < early]["duration_s"].sum()
        total_s = dur.sum()
        feats[f"{beh[:4]}_early_frac"] = float(early_s / total_s
                                                if total_s > 0 else np.nan)
    return feats


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_features(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        sid = row["subject_id"]           # e.g. MA1_1
        ses = float(row.get("session_duration_s", 178.0))

        extra = bout_extra_features(sid, ses)

        # grooming mean bout: basit hesap (bouts CSV yoksa yedek)
        g_count = float(row.get("groom_bout_count", 0) or 0)
        g_total = float(row.get("groom_total_s",    0) or 0)
        r_count = float(row.get("rear_bout_count",  0) or 0)
        r_total = float(row.get("rear_total_s",     0) or 0)

        groom_mean_bout = extra.get("groo_mean_bout_s",
                                    g_total / g_count if g_count > 0 else 0.0)
        rear_mean_bout  = extra.get("rear_mean_bout_s",
                                    r_total / r_count if r_count > 0 else 0.0)

        feat = {
            "subject_id":         sid,
            "cohort":             row.get("cohort", ""),
            "group":              row.get("group",  ""),
            # --- Thigmotaxis / locomotion ---
            "pct_periphery":      float(row.get("pct_time_periphery", np.nan)),
            "pct_center":         float(row.get("pct_time_center",    np.nan)),
            "center_entries":     float(row.get("center_zone_entries", np.nan)),
            "total_distance_px":  float(row.get("total_distance_px",   np.nan)),
            "mean_speed_px_s":    float(row.get("mean_speed_px_s",     np.nan)),
            "spatial_entropy":    float(row.get("spatial_entropy_norm",np.nan)),
            # --- Freeze ---
            "pct_freeze":         float(row.get("pct_time_freeze",     np.nan)),
            "freeze_bout_count":  float(row.get("freeze_bout_count",   np.nan)),
            # --- Rearing ---
            "rear_count":         float(r_count),
            "rear_total_s":       float(r_total),
            "rear_pct":           float(row.get("rear_pct_time",       np.nan)),
            "rear_mean_bout_s":   float(rear_mean_bout),
            "rear_rate_per_min":  float(r_count / ses * 60) if ses > 0 else np.nan,
            "rear_early_frac":    extra.get("rear_early_frac", np.nan),
            # --- Grooming ---
            "groom_count":        float(g_count),
            "groom_total_s":      float(g_total),
            "groom_pct":          float(row.get("groom_pct_time",      np.nan)),
            "groom_mean_bout_s":  float(groom_mean_bout),
            "groom_bout_cv":      extra.get("groo_bout_cv", np.nan),
            "groom_early_frac":   extra.get("groo_early_frac", np.nan),
            # --- Combined ---
            "rear_groom_ratio":   (float(r_total / g_total)
                                   if g_total > 0 else np.nan),
        }
        rows.append(feat)
        print(f"  {sid:<10s}  group={feat['group']:<22s}  "
              f"rear={r_count:.0f}  groom={g_count:.0f}  "
              f"groom_mean={groom_mean_bout:.2f}s")
    return pd.DataFrame(rows)


# ── İstatistik ────────────────────────────────────────────────────────────────

def kruskal_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    groups = df["group"].unique()
    rows = []
    for col in feature_cols:
        samples = [df.loc[df["group"] == g, col].dropna().values
                   for g in groups]
        samples = [s for s in samples if len(s) >= 2]
        if len(samples) < 2:
            continue
        try:
            stat, p = stats.kruskal(*samples)
        except Exception:
            continue
        # eta-squared yaklaşımı: (H - k + 1) / (n - k)
        n = sum(len(s) for s in samples)
        k = len(samples)
        eta2 = max(0.0, (stat - k + 1) / (n - k)) if n > k else 0.0
        rows.append({"feature": col, "H": round(stat, 3),
                     "p_value": round(p, 4), "eta2": round(eta2, 3),
                     "significant": p < 0.05})
    return pd.DataFrame(rows).sort_values("p_value")


# ── PCA ──────────────────────────────────────────────────────────────────────

def plot_pca(df: pd.DataFrame, feature_cols: list[str], out: Path) -> None:
    X = df[feature_cols].copy()
    # çok fazla NaN olan sütunları çıkar
    keep = X.columns[X.isna().mean() < 0.5]
    X = X[keep].fillna(X[keep].mean())
    if X.shape[1] < 2:
        print("[warn] PCA için yeterli feature yok")
        return

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(3, X.shape[1]))
    coords = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_ * 100

    # gruplar
    all_groups = df["group"].unique()
    color_map = dict(GROUP_COLORS)
    for i, g in enumerate(all_groups):
        if g not in color_map:
            color_map[g] = _FALLBACK[i % len(_FALLBACK)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (xi, yi, title) in zip(axes, [
        (0, 1, f"PC1 ({var[0]:.1f}%) vs PC2 ({var[1]:.1f}%)"),
        (0, 2 if coords.shape[1] > 2 else 1,
         f"PC1 ({var[0]:.1f}%) vs PC3 ({var[2]:.1f}%)"
         if coords.shape[1] > 2 else ""),
    ]):
        if not title:
            ax.set_visible(False)
            continue
        for g in all_groups:
            mask = df["group"] == g
            ax.scatter(coords[mask, xi], coords[mask, yi],
                       label=g, color=color_map[g], s=90, alpha=0.85,
                       edgecolors="white", linewidth=0.5)
            for idx in np.where(mask)[0]:
                ax.annotate(df["subject_id"].iloc[idx],
                            (coords[idx, xi], coords[idx, yi]),
                            fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"PC{xi+1}", fontsize=11)
        ax.set_ylabel(f"PC{yi+1}", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.legend(fontsize=8, framealpha=0.7)

    # loading arrows (PC1 vs PC2 panel)
    ax = axes[0]
    loadings = pca.components_[:2, :].T
    feat_names = list(keep)
    scale = 0.35 * np.abs(coords[:, :2]).max()
    top_idx = np.argsort(np.linalg.norm(loadings[:, :2], axis=1))[-8:]
    for i in top_idx:
        ax.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
        ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12,
                feat_names[i].replace("_", "\n"), fontsize=6, color="#333333",
                ha="center")

    fig.suptitle("OFT Anksiyete Profili — PCA", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


# ── Kutu grafikleri ───────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, feature_cols: list[str],
                  stats_df: pd.DataFrame, out: Path) -> None:
    all_groups  = df["group"].unique()
    color_map   = dict(GROUP_COLORS)
    for i, g in enumerate(all_groups):
        if g not in color_map:
            color_map[g] = _FALLBACK[i % len(_FALLBACK)]

    sig_feats = stats_df[stats_df["significant"]]["feature"].tolist()
    plot_feats = sig_feats if sig_feats else feature_cols[:12]

    ncols = 4
    nrows = int(np.ceil(len(plot_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, plot_feats):
        data   = [df.loc[df["group"] == g, feat].dropna().values
                  for g in all_groups]
        colors = [color_map[g] for g in all_groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        # scatter jitter
        for i, (d, c) in enumerate(zip(data, colors), 1):
            x = np.random.default_rng(i).uniform(i - 0.2, i + 0.2, len(d))
            ax.scatter(x, d, color=c, s=18, zorder=3, alpha=0.9,
                       edgecolors="white", linewidth=0.4)
        row = stats_df[stats_df["feature"] == feat]
        p_str = f"p={row['p_value'].values[0]:.3f}" if len(row) else ""
        ax.set_title(f"{feat}\n{p_str}", fontsize=8)
        ax.set_xticks(range(1, len(all_groups) + 1))
        ax.set_xticklabels([g[:6] for g in all_groups], fontsize=7, rotation=15)

    for ax in axes[len(plot_feats):]:
        ax.set_visible(False)

    fig.suptitle("OFT Grup Karşılaştırması (Kruskal-Wallis anlamlı özellikler)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


# ── Ana ──────────────────────────────────────────────────────────────────────

def main() -> None:
    metrics = load_all_metrics()
    print(f"\n[features] her hayvan için özellikler hesaplanıyor...")
    feat_df = build_features(metrics)

    out_csv = ROOT / "data" / "anxiety_features.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"\n[write] {out_csv.relative_to(ROOT)}: "
          f"{len(feat_df)} hayvan × {feat_df.shape[1]} sütun")

    # grup dağılımı
    print("\n[groups]")
    print(feat_df.groupby("group")["subject_id"].apply(list).to_string())

    meta_cols    = {"subject_id", "cohort", "group"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    print("\n[stats] Kruskal-Wallis testi...")
    stats_df = kruskal_table(feat_df, feature_cols)
    stats_path = REPORTS / "anxiety_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(stats_df.to_string(index=False))
    print(f"[write] {stats_path.relative_to(ROOT)}")

    print("\n[pca] oluşturuluyor...")
    plot_pca(feat_df, feature_cols, REPORTS / "anxiety_pca.png")

    print("\n[boxplots] oluşturuluyor...")
    plot_boxplots(feat_df, feature_cols, stats_df,
                  REPORTS / "anxiety_boxplots.png")

    print("\n[done]")


if __name__ == "__main__":
    main()
