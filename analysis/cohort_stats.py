"""
Cohort-level statistical analysis for OFT metrics.

For each engineered feature in ``data/oft_metrics_all.csv``:
  - Kruskal-Wallis H test (asymptotic + Monte-Carlo permutation p)
  - Epsilon-squared effect size
  - Dunn post-hoc (pairwise) with within-feature BH-FDR

Across-features:
  - Benjamini-Hochberg FDR on KW p-values
  - PERMANOVA (pseudo-F, Euclidean distance on z-scored features)

Outputs:
  reports/cohort_kruskal_wallis.csv
  reports/cohort_dunn_posthoc.csv
  reports/cohort_permanova.csv

Run:
  python analysis/cohort_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "oft_metrics_all.csv"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_COL = "group"
ID_COLS = {"subject_id", "cohort", "group", "session_duration_s", "n_frames", "bouts_csv_found"}
N_PERM = 10_000
RNG = np.random.default_rng(42)


# ─── helpers ──────────────────────────────────────────────────────────────────

def epsilon_squared(H: float, k: int, n: int) -> float:
    """Tomczak & Tomczak (2014) — KW effect size."""
    if n - k <= 0:
        return np.nan
    return (H - k + 1) / (n - k)


def kw_permutation_p(values: np.ndarray, labels: np.ndarray, n_perm: int) -> tuple[float, float]:
    """Return (observed H, two-sided permutation p).
    Permutes labels among the same set of values."""
    groups_obs = [values[labels == g] for g in np.unique(labels)]
    H_obs = stats.kruskal(*groups_obs).statistic
    count = 0
    n = len(values)
    for _ in range(n_perm):
        perm_labels = RNG.permutation(labels)
        perm_groups = [values[perm_labels == g] for g in np.unique(labels)]
        try:
            H_perm = stats.kruskal(*perm_groups).statistic
        except ValueError:
            continue
        if H_perm >= H_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return H_obs, p


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # enforce monotonicity (from largest p to smallest)
    sorted_q = q[order]
    for i in range(n - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    out = np.empty(n)
    out[order] = np.clip(sorted_q, 0, 1)
    return out


def dunn_posthoc(values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Pairwise Dunn test with tie correction.

    Returns long-format DataFrame: group_a, group_b, z, p_two_sided.
    """
    n = len(values)
    ranks = stats.rankdata(values)
    groups = np.unique(labels)
    mean_ranks = {g: ranks[labels == g].mean() for g in groups}
    sizes = {g: int((labels == g).sum()) for g in groups}

    # tie correction term — sigma^2 = [N(N+1)/12] * C  where
    # C = 1 - sum(t_j^3 - t_j) / (N^3 - N)
    _, counts = np.unique(values, return_counts=True)
    ties = counts[counts > 1]
    if len(ties):
        C = 1 - (np.sum(ties ** 3 - ties) / (n ** 3 - n))
    else:
        C = 1.0
    sigma2 = (n * (n + 1) / 12.0) * C

    rows = []
    for i, ga in enumerate(groups):
        for gb in groups[i + 1 :]:
            se = np.sqrt(sigma2 * (1 / sizes[ga] + 1 / sizes[gb]))
            z = (mean_ranks[ga] - mean_ranks[gb]) / se if se > 0 else 0.0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({"group_a": ga, "group_b": gb, "z": z, "p_two_sided": p})
    return pd.DataFrame(rows)


def permanova(X: np.ndarray, labels: np.ndarray, n_perm: int) -> dict:
    """Pseudo-F PERMANOVA on Euclidean distances of (already-scaled) X.

    Anderson (2001) — F = (SSA/(k-1)) / (SSW/(N-k))
    """
    n, _ = X.shape
    # squared Euclidean distance matrix (use full matrix, ignore diagonal)
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
    F_obs = (SSA_obs / (k - 1)) / (SSW_obs / (n - k))
    R2 = SSA_obs / SST

    count = 0
    for _ in range(n_perm):
        perm = RNG.permutation(labels)
        SSW_p = ssw(perm)
        SSA_p = SST - SSW_p
        F_p = (SSA_p / (k - 1)) / (SSW_p / (n - k))
        if F_p >= F_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"pseudo_F": F_obs, "R2": R2, "p_perm": p, "n_perm": n_perm, "k": k, "n": n}


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(INPUT)
    print(f"[load] {INPUT.name}: {len(df)} subjects, groups={df[GROUP_COL].unique().tolist()}")

    feature_cols = [
        c for c in df.columns
        if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"[load] {len(feature_cols)} numeric features")

    labels = df[GROUP_COL].values
    k = len(np.unique(labels))
    n = len(df)

    # ── per-feature KW ────────────────────────────────────────────────────────
    kw_rows = []
    dunn_long = []
    for feat in feature_cols:
        vals = df[feat].values.astype(float)
        if np.unique(vals).size < 2:
            continue
        kw_asym = stats.kruskal(*[vals[labels == g] for g in np.unique(labels)])
        H_obs, p_perm = kw_permutation_p(vals, labels, N_PERM)
        eps = epsilon_squared(H_obs, k, n)
        kw_rows.append({
            "feature": feat,
            "H": H_obs,
            "p_asymptotic": kw_asym.pvalue,
            "p_permutation": p_perm,
            "epsilon_sq": eps,
            "effect_size_label": (
                "negligible" if eps < 0.01 else
                "small" if eps < 0.06 else
                "medium" if eps < 0.14 else
                "large"
            ),
        })

        dunn = dunn_posthoc(vals, labels)
        dunn["feature"] = feat
        # within-feature BH on Dunn pairwise p
        dunn["p_bh_within_feature"] = bh_fdr(dunn["p_two_sided"].values)
        dunn_long.append(dunn)

    kw_df = pd.DataFrame(kw_rows).sort_values("p_permutation")
    kw_df["q_bh"] = bh_fdr(kw_df["p_permutation"].values)
    kw_df = kw_df.reset_index(drop=True)

    dunn_df = pd.concat(dunn_long, ignore_index=True)
    dunn_df = dunn_df[["feature", "group_a", "group_b", "z", "p_two_sided", "p_bh_within_feature"]]

    # ── PERMANOVA on z-scored feature matrix ──────────────────────────────────
    X = df[feature_cols].values.astype(float)
    # robust scaling — z by feature, drop zero-variance columns
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    keep = stds > 0
    Xz = (X[:, keep] - means[keep]) / stds[keep]
    perm = permanova(Xz, labels, N_PERM)

    # ── write ─────────────────────────────────────────────────────────────────
    kw_path = OUT / "cohort_kruskal_wallis.csv"
    dunn_path = OUT / "cohort_dunn_posthoc.csv"
    perm_path = OUT / "cohort_permanova.csv"

    kw_df.to_csv(kw_path, index=False, float_format="%.4f")
    dunn_df.to_csv(dunn_path, index=False, float_format="%.4f")
    pd.DataFrame([perm]).to_csv(perm_path, index=False, float_format="%.4f")

    print(f"\n[write] {kw_path.relative_to(ROOT)}  ({len(kw_df)} rows)")
    print(f"[write] {dunn_path.relative_to(ROOT)}  ({len(dunn_df)} rows)")
    print(f"[write] {perm_path.relative_to(ROOT)}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n── Kruskal-Wallis (sorted by permutation p) ──")
    print(kw_df.to_string(index=False))

    print("\n── PERMANOVA (cohort ~ z-scored feature matrix, Euclidean) ──")
    for k_, v in perm.items():
        print(f"  {k_:>10} = {v:.4f}" if isinstance(v, float) else f"  {k_:>10} = {v}")

    sig = kw_df[kw_df["p_permutation"] < 0.05]
    print(f"\n[summary] features with p_perm < .05 (uncorrected): {len(sig)} / {len(kw_df)}")
    sig_q = kw_df[kw_df["q_bh"] < 0.05]
    print(f"[summary] features with q_bh   < .05 (FDR-corrected): {len(sig_q)} / {len(kw_df)}")


if __name__ == "__main__":
    main()
