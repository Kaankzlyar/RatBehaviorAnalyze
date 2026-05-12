"""
cohort_stats_epm.py
-------------------
Plus Maze / EPM verisinde kohort-duzeyinde istatistik.

Her EPM feature icin:
  - Kruskal-Wallis H testi (asimptotik + permutasyon p)
  - Epsilon-kare etki buyuklugu
  - Dunn post-hoc (pairwise, within-feature BH-FDR)

Tum featurelarda:
  - Benjamini-Hochberg FDR (cross-feature)
  - PERMANOVA (Oklid, z-skorlu EPM feature seti)

EPM yorumlamasi:
  - Acik kol (left + right) = anksiyolitik gosterge (artis = dusuk kaygi)
  - Kapali kol (top + bottom) = guveli alan
  - total_entries = lokomotor kovariat

Ciktilar:
  reports/cohort_epm_kw.csv
  reports/cohort_epm_dunn.csv
  reports/cohort_epm_permanova.csv

Kullanim:
  python analysis/plus_maze/cohort_stats_epm.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT  = Path(__file__).resolve().parent.parent.parent
INPUT = ROOT / "data" / "plus_maze_metrics_all.csv"
OUT   = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_COL = "cohort"
N_PERM    = 10_000
RNG       = np.random.default_rng(42)

# EPM icin analiz edilecek feature seti
EPM_FEATURES = [
    # Birincil EPM endpointleri
    "pct_open_arm",           # % acik kol suresi (Left+Right) — altin standart
    "pct_open_arm_entries",   # % acik kol girisi
    "anxiety_index_epm",      # (pct_open_arm + pct_open_arm_entries) / 2
    # Lokomotor kovariat
    "total_entries",          # toplam giris — motor aktivite
    "mean_speed_px_s",        # ortalama hiz
    "total_distance_px",      # toplam mesafe
    # Alternasyon / bilis
    "successive_alternation_pct",
    "perseveration_rate_pct",
    # Kol tercihi
    "arm_preference_index",
    # Bireysel kol yuzdeleri
    "pct_time_left",
    "pct_time_right",
    "pct_time_top",
    "pct_time_bottom",
    "pct_time_junction",
]

ID_COLS = {"subject_id", "cohort_id", "cohort", "session",
           "n_frames", "n_valid_frames", "session_duration_s",
           "most_visited_arm", "entry_sequence"}


# ─── helpers (OFT cohort_stats.py ile ayni) ───────────────────────────────────

def epsilon_squared(H: float, k: int, n: int) -> float:
    if n - k <= 0:
        return np.nan
    return (H - k + 1) / (n - k)


def kw_permutation_p(values: np.ndarray, labels: np.ndarray,
                     n_perm: int) -> tuple[float, float]:
    groups_obs = [values[labels == g] for g in np.unique(labels)]
    H_obs = stats.kruskal(*groups_obs).statistic
    count = 0
    for _ in range(n_perm):
        perm_labels = RNG.permutation(labels)
        perm_groups = [values[perm_labels == g] for g in np.unique(labels)]
        try:
            H_perm = stats.kruskal(*perm_groups).statistic
        except ValueError:
            continue
        if H_perm >= H_obs:
            count += 1
    return H_obs, (count + 1) / (n_perm + 1)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    sorted_q = q[order]
    for i in range(n - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    out = np.empty(n)
    out[order] = np.clip(sorted_q, 0, 1)
    return out


def dunn_posthoc(values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    n = len(values)
    ranks = stats.rankdata(values)
    groups = np.unique(labels)
    mean_ranks = {g: ranks[labels == g].mean() for g in groups}
    sizes = {g: int((labels == g).sum()) for g in groups}

    _, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1]
    C = 1 - (np.sum(ties ** 3 - ties) / (n ** 3 - n)) if len(ties) else 1.0
    sigma2 = (n * (n + 1) / 12.0) * C

    rows = []
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            se = np.sqrt(sigma2 * (1 / sizes[ga] + 1 / sizes[gb]))
            z = (mean_ranks[ga] - mean_ranks[gb]) / se if se > 0 else 0.0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({"group_a": ga, "group_b": gb, "z": z, "p_two_sided": p})
    return pd.DataFrame(rows)


def permanova(X: np.ndarray, labels: np.ndarray, n_perm: int) -> dict:
    n, _ = X.shape
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    iu = np.triu_indices(n, k=1)
    SST = D2[iu].sum() / n

    def ssw(lab):
        s = 0.0
        for g in np.unique(lab):
            idx = np.where(lab == g)[0]
            n_g = len(idx)
            if n_g < 2:
                continue
            sub = D2[np.ix_(idx, idx)]
            iu_g = np.triu_indices(n_g, k=1)
            s += sub[iu_g].sum() / n_g
        return s

    k = len(np.unique(labels))
    SSW_obs = ssw(labels)
    SSA_obs = SST - SSW_obs
    F_obs   = (SSA_obs / (k - 1)) / (SSW_obs / (n - k))
    R2      = SSA_obs / SST

    count = 0
    for _ in range(n_perm):
        perm  = RNG.permutation(labels)
        SSW_p = ssw(perm)
        SSA_p = SST - SSW_p
        F_p   = (SSA_p / (k - 1)) / (SSW_p / (n - k))
        if F_p >= F_obs:
            count += 1
    return {"pseudo_F": F_obs, "R2": R2, "p_perm": (count + 1) / (n_perm + 1),
            "n_perm": n_perm, "k": k, "n": n}


def effect_label(eps2: float) -> str:
    if np.isnan(eps2) or eps2 < 0.04:
        return "negligible"
    if eps2 < 0.25:
        return "small"
    if eps2 < 0.64:
        return "medium"
    return "large"


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(INPUT)
    print(f"[load] {INPUT.name}: {len(df)} subjects")
    print(f"       cohorts: {sorted(df[GROUP_COL].unique().tolist())}")

    # Sadece mevcut ve numerik feature'lari al
    feature_cols = [f for f in EPM_FEATURES
                    if f in df.columns
                    and pd.api.types.is_numeric_dtype(df[f])]
    print(f"[load] {len(feature_cols)} EPM feature analiz edilecek\n")

    labels = df[GROUP_COL].values
    k = len(np.unique(labels))
    n = len(df)

    # ── per-feature KW ────────────────────────────────────────────────────────
    kw_rows   = []
    dunn_long = []

    for feat in feature_cols:
        vals = df[feat].fillna(df[feat].median()).values.astype(float)
        if np.unique(vals).size < 2:
            continue

        H_obs, p_perm = kw_permutation_p(vals, labels, N_PERM)
        p_asym = stats.kruskal(*[vals[labels == g] for g in np.unique(labels)]).pvalue
        eps2   = epsilon_squared(H_obs, k, n)

        kw_rows.append({
            "feature":          feat,
            "H":                round(H_obs, 4),
            "p_asymptotic":     round(p_asym, 4),
            "p_permutation":    round(p_perm, 4),
            "epsilon_sq":       round(eps2, 4),
            "effect_size_label": effect_label(eps2),
        })
        print(f"  {feat:<35}  H={H_obs:.3f}  p_perm={p_perm:.4f}  eps2={eps2:.3f}")

        dunn_df  = dunn_posthoc(vals, labels)
        dunn_bh  = bh_fdr(dunn_df["p_two_sided"].values)
        dunn_df["p_bh_within_feature"] = dunn_bh.round(4)
        dunn_df.insert(0, "feature", feat)
        dunn_long.append(dunn_df)

    # BH-FDR cross-feature
    kw_df = pd.DataFrame(kw_rows)
    kw_df["q_bh"] = bh_fdr(kw_df["p_permutation"].values).round(4)
    kw_df = kw_df.sort_values("p_permutation").reset_index(drop=True)

    # ── PERMANOVA ─────────────────────────────────────────────────────────────
    # Sadece birincil EPM feature'lari uzerinde
    primary = ["pct_open_arm", "pct_open_arm_entries",
               "total_entries", "mean_speed_px_s"]
    primary = [f for f in primary if f in df.columns]
    X_raw = df[primary].fillna(df[primary].median()).values.astype(float)
    # z-skore
    std = X_raw.std(axis=0, ddof=1)
    std[std == 0] = 1
    X_z = (X_raw - X_raw.mean(axis=0)) / std

    print("\n[PERMANOVA] birincil EPM feature seti uzerinde...")
    perm_res = permanova(X_z, labels, N_PERM)
    perm_df  = pd.DataFrame([{
        "features":   "+".join(primary),
        "pseudo_F":   round(perm_res["pseudo_F"], 4),
        "R2":         round(perm_res["R2"], 4),
        "p_perm":     round(perm_res["p_perm"], 4),
        "n_perm":     perm_res["n_perm"],
        "k_groups":   perm_res["k"],
        "n_subjects": perm_res["n"],
    }])

    # ── kaydet ────────────────────────────────────────────────────────────────
    kw_path   = OUT / "cohort_epm_kw.csv"
    dunn_path = OUT / "cohort_epm_dunn.csv"
    perm_path = OUT / "cohort_epm_permanova.csv"

    kw_df.to_csv(kw_path, index=False)
    pd.concat(dunn_long, ignore_index=True).to_csv(dunn_path, index=False)
    perm_df.to_csv(perm_path, index=False)

    # ── ozet yazdir ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EPM KOHORT ISTATISTIGI OZETI")
    print(f"{'='*60}")
    print(kw_df[["feature", "H", "p_permutation", "epsilon_sq",
                 "effect_size_label", "q_bh"]].to_string(index=False))

    print(f"\nPERMANOVA  pseudo-F={perm_res['pseudo_F']:.3f}  "
          f"R2={perm_res['R2']:.3f}  p={perm_res['p_perm']:.4f}")

    print(f"\nKohort ortalamalari (birincil EPM endpointler):")
    summary_cols = [c for c in ["pct_open_arm", "pct_open_arm_entries",
                                 "anxiety_index_epm", "total_entries"]
                    if c in df.columns]
    print(df.groupby(GROUP_COL)[summary_cols].mean().round(2).to_string())

    print(f"\n[done]")
    print(f"  {kw_path}")
    print(f"  {dunn_path}")
    print(f"  {perm_path}")


if __name__ == "__main__":
    main()
