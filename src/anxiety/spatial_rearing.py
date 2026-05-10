"""
Spatial Rearing Analysis — Center vs Wall
-----------------------------------------
Her rearing bout sırasında body_center koordinatının medyan konumunu hesaplar
ve arena iç bölgesinde mi (center) yoksa periferde mi (wall) olduğunu sınıflar.

Bilimsel motivasyon
-------------------
Toplam rear_count grup ayrımında marjinal (p=0.045) — fakat *nerede* şahlanıldığı
kalitatif olarak çok daha güçlü bir anksiyete sinyalidir:
  - Merkez rearing: düşük anksiyete (savunmasız karın açık).
  - Duvar rearing : kaçış arayışı / yüksek anksiyete.
İki grupta toplam sayı eşit olsa bile dağılım farkı niteliksel anlam taşır.

Yöntem
------
  - body_center keypoint'i referans noktası (rearing sırasında nose yukarı çıkıp
    kayar; body_center daha kararlı).
  - DLC likelihood < 0.6 olan kareler atılır.
  - Her bout için frame'lerin body_center medyan (x,y)'si hesaplanır.
  - Inner zone: arena (396,776,153,530) etrafında %20 margin (oft_metrics.py'deki
    aynı default — iç bölge 472–700, 228–455 px).
  - Medyan (x,y) inner zone içindeyse bout = center, değilse wall.

Çıktılar
--------
  data/spatial_rearing.csv             — per-subject ve per-bout sonuçlar
  reports/spatial_rearing_stats.csv    — Kruskal-Wallis + Mann-Whitney
  reports/figures/spatial_rearing_boxplot.png
  reports/figures/spatial_rearing_arena.png  — tüm bout'ların arena üzerinde dağılımı

Kullanım
--------
  python -m src.anxiety.spatial_rearing
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent.parent
DLC_DIR  = ROOT / "data" / "DLCfiltered"
DATA     = ROOT / "data"
REPORTS  = ROOT / "reports"
FIGS     = REPORTS / "figures"
for d in (REPORTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

ARENA      = (396.0, 776.0, 153.0, 530.0)   # XMIN XMAX YMIN YMAX (oft_metrics default)
MARGIN     = 0.20                            # iç bölge marjı
LIKELIHOOD = 0.6

GROUP_COLORS = {
    "Control":              "#2196F3",
    "Aspartame":            "#F44336",
    "Grapefruit":           "#4CAF50",
    "Aspartame+Grapefruit": "#FF9800",
}

COHORT_MAP = {
    "MA1": "Control", "MA2": "Control",
    "MA3": "Aspartame", "MA4": "Aspartame",
    "MA5": "Grapefruit", "MA6": "Grapefruit",
    "MA7": "Aspartame+Grapefruit", "MA8": "Aspartame+Grapefruit",
}


def inner_zone(arena: tuple, margin: float = MARGIN) -> tuple:
    x0, x1, y0, y1 = arena
    w, h = x1 - x0, y1 - y0
    return (x0 + margin * w, x1 - margin * w, y0 + margin * h, y1 - margin * h)


def load_body_center(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """DLC CSV'den body_center x,y; likelihood < threshold ise NaN."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = ["_".join(c[-2:]).strip() for c in df.columns.values]
    x  = df["body_center_x"].to_numpy(dtype=float)
    y  = df["body_center_y"].to_numpy(dtype=float)
    lk = df["body_center_likelihood"].to_numpy(dtype=float)
    x[lk < LIKELIHOOD] = np.nan
    y[lk < LIKELIHOOD] = np.nan
    return x, y


def classify_bout(x: np.ndarray, y: np.ndarray,
                  start: int, end: int,
                  inner: tuple) -> tuple[str, float, float]:
    """Bout boyunca body_center medyanı → 'center' | 'wall' | 'unknown'."""
    end = min(end + 1, len(x))
    seg_x = x[start:end]; seg_y = y[start:end]
    valid = ~(np.isnan(seg_x) | np.isnan(seg_y))
    if valid.sum() < 1:
        return ("unknown", float("nan"), float("nan"))
    mx = float(np.nanmedian(seg_x))
    my = float(np.nanmedian(seg_y))
    ix0, ix1, iy0, iy1 = inner
    label = "center" if (ix0 <= mx <= ix1 and iy0 <= my <= iy1) else "wall"
    return (label, mx, my)


def find_subject_dirs() -> list[Path]:
    """OpenFieldMA*_* klasörleri."""
    pat = re.compile(r"^OpenFieldMA\d+_\d+$")
    out = []
    for p in DLC_DIR.glob("**/OpenFieldMA*_*"):
        if p.is_dir() and pat.match(p.name):
            out.append(p)
    return sorted(out)


def process_subject(subj_dir: Path, inner: tuple) -> tuple[dict, list[dict]]:
    base = subj_dir.name             # e.g. OpenFieldMA1_1
    pose_csv  = subj_dir / f"{base}.csv"
    bouts_csv = subj_dir / f"{base}_behavior_bouts.csv"
    if not pose_csv.exists() or not bouts_csv.exists():
        return {}, []

    m = re.search(r"MA(\d+)_(\d+)", base)
    cohort = f"MA{m.group(1)}"
    sid    = f"{cohort}_{m.group(2)}"
    group  = COHORT_MAP.get(cohort, "Unknown")

    x, y = load_body_center(pose_csv)
    bouts = pd.read_csv(bouts_csv)
    rears = bouts[bouts["behaviour"] == "rearing"].copy()

    bout_rows: list[dict] = []
    for _, r in rears.iterrows():
        s = int(r["start_frame"]); e = int(r["end_frame"])
        label, mx, my = classify_bout(x, y, s, e, inner)
        bout_rows.append({
            "subject_id": sid,
            "cohort":     cohort,
            "group":      group,
            "bout":       int(r["bout"]),
            "start_s":    float(r["start_s"]),
            "duration_s": float(r["duration_s"]),
            "median_x":   round(mx, 1) if not np.isnan(mx) else np.nan,
            "median_y":   round(my, 1) if not np.isnan(my) else np.nan,
            "zone":       label,
        })

    n_total  = len(bout_rows)
    n_center = sum(1 for b in bout_rows if b["zone"] == "center")
    n_wall   = sum(1 for b in bout_rows if b["zone"] == "wall")
    n_unk    = sum(1 for b in bout_rows if b["zone"] == "unknown")
    s_center = sum(b["duration_s"] for b in bout_rows if b["zone"] == "center")
    s_wall   = sum(b["duration_s"] for b in bout_rows if b["zone"] == "wall")
    classified = n_center + n_wall

    summary = {
        "subject_id":            sid,
        "cohort":                cohort,
        "group":                 group,
        "rear_count_total":      n_total,
        "rear_count_center":     n_center,
        "rear_count_wall":       n_wall,
        "rear_count_unknown":    n_unk,
        "rear_center_frac":      round(n_center / classified, 4) if classified else float("nan"),
        "rear_total_s_center":   round(s_center, 2),
        "rear_total_s_wall":     round(s_wall, 2),
        "rear_center_minus_wall":(n_center - n_wall),
    }
    return summary, bout_rows


def stats_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    is_treated = df["group"].astype(str).str.lower() != "control"
    n_total = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 5 or vals.nunique() < 2:
            continue

        groups = [df.loc[df["group"] == g, col].dropna().values
                  for g in df["group"].unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            H, p_kw = stats.kruskal(*groups)
            k = len(groups)
            eta2 = max(0.0, (H - k + 1) / (n_total - k))
        else:
            H, p_kw, eta2 = np.nan, np.nan, np.nan

        ctrl = df.loc[~is_treated, col].dropna().values
        trt  = df.loc[ is_treated, col].dropna().values
        if len(ctrl) >= 2 and len(trt) >= 2:
            U, p_mw = stats.mannwhitneyu(trt, ctrl, alternative="two-sided")
            r_rb = 1.0 - (2.0 * U) / (len(ctrl) * len(trt))
            sd = np.sqrt(((len(ctrl) - 1) * ctrl.var(ddof=1)
                          + (len(trt) - 1) * trt.var(ddof=1))
                         / (len(ctrl) + len(trt) - 2)) if (len(ctrl) > 1 and len(trt) > 1) else 0.0
            d = float((trt.mean() - ctrl.mean()) / sd) if sd > 0 else np.nan
        else:
            U, p_mw, r_rb, d = np.nan, np.nan, np.nan, np.nan

        rows.append({
            "metric":        col,
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
    return pd.DataFrame(rows)


def plot_boxplots(df: pd.DataFrame, stats_df: pd.DataFrame, out: Path) -> None:
    cols = ["rear_center_frac", "rear_count_center", "rear_count_wall"]
    fig, axes = plt.subplots(1, len(cols), figsize=(5.0 * len(cols), 4.5))
    groups = ["Control", "Aspartame", "Grapefruit", "Aspartame+Grapefruit"]
    groups = [g for g in groups if g in df["group"].unique()]

    for ax, col in zip(axes, cols):
        data = [df.loc[df["group"] == g, col].dropna().values for g in groups]
        bp = ax.boxplot(data, labels=[g.replace("Aspartame+Grapefruit", "ASP+G") for g in groups],
                        patch_artist=True, widths=0.55)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(GROUP_COLORS.get(g, "#888"))
            patch.set_alpha(0.65)
        for i, vals in enumerate(data, start=1):
            if len(vals):
                ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.07, 0.07, len(vals)),
                           vals, color="black", s=14, zorder=3, alpha=0.75)
        row = stats_df[stats_df["metric"] == col]
        if len(row):
            r = row.iloc[0]
            ax.set_title(f"{col}\nKW p={r.kruskal_p:.3g}  η²={r.eta2:.2f}\n"
                         f"MW p={r.mw_p:.3g}  d={r.cohens_d:.2f}", fontsize=10)
        else:
            ax.set_title(col)
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Spatial Rearing — Center vs Wall by Group", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


def plot_arena(bouts_df: pd.DataFrame, inner: tuple, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    groups = ["Control", "Aspartame", "Grapefruit", "Aspartame+Grapefruit"]
    ax_x0, ax_x1, ax_y0, ax_y1 = ARENA
    ix0, ix1, iy0, iy1 = inner

    for ax, g in zip(axes, groups):
        sub = bouts_df[bouts_df["group"] == g]
        # arena
        ax.plot([ax_x0, ax_x1, ax_x1, ax_x0, ax_x0],
                [ax_y0, ax_y0, ax_y1, ax_y1, ax_y0], "k-", lw=1.4)
        # inner zone
        ax.plot([ix0, ix1, ix1, ix0, ix0],
                [iy0, iy0, iy1, iy1, iy0], "k--", lw=1, alpha=0.6)
        # bout positions
        center = sub[sub["zone"] == "center"]
        wall   = sub[sub["zone"] == "wall"]
        ax.scatter(wall["median_x"], wall["median_y"],
                   color=GROUP_COLORS.get(g, "#888"), edgecolor="white",
                   s=28, alpha=0.55, label=f"wall (n={len(wall)})")
        ax.scatter(center["median_x"], center["median_y"],
                   color=GROUP_COLORS.get(g, "#888"), edgecolor="black",
                   s=44, alpha=0.95, marker="*", label=f"center (n={len(center)})")
        ax.set_title(g, fontsize=11)
        ax.set_xlabel("x (px)")
        ax.legend(loc="upper right", fontsize=8)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("y (px)")
    fig.suptitle("Rearing Locations Across Arena (median body_center per bout)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out.relative_to(ROOT)}")


def main() -> None:
    inner = inner_zone(ARENA)
    print(f"[arena]  X {ARENA[0]:.0f}-{ARENA[1]:.0f}   Y {ARENA[2]:.0f}-{ARENA[3]:.0f}")
    print(f"[inner]  X {inner[0]:.0f}-{inner[1]:.0f}   Y {inner[2]:.0f}-{inner[3]:.0f}")

    subj_dirs = find_subject_dirs()
    print(f"[scan]   {len(subj_dirs)} subject klasörü")

    summaries: list[dict] = []
    all_bouts: list[dict] = []
    for sd in subj_dirs:
        s, b = process_subject(sd, inner)
        if not s:
            print(f"  [skip] {sd.name} — pose ya da bouts CSV eksik")
            continue
        summaries.append(s)
        all_bouts.extend(b)
        print(f"  {s['subject_id']:<8s}  group={s['group']:<22s}  "
              f"total={s['rear_count_total']:>3d}  "
              f"center={s['rear_count_center']:>3d}  "
              f"wall={s['rear_count_wall']:>3d}  "
              f"unk={s['rear_count_unknown']:>3d}  "
              f"frac={s['rear_center_frac']}")

    summary_df = pd.DataFrame(summaries)
    bouts_df   = pd.DataFrame(all_bouts)

    summary_csv = DATA / "spatial_rearing.csv"
    bouts_csv   = DATA / "spatial_rearing_bouts.csv"
    summary_df.to_csv(summary_csv, index=False)
    bouts_df.to_csv(bouts_csv, index=False)
    print(f"[write] {summary_csv.relative_to(ROOT)}")
    print(f"[write] {bouts_csv.relative_to(ROOT)}")

    metric_cols = ["rear_count_total", "rear_count_center", "rear_count_wall",
                   "rear_center_frac", "rear_total_s_center", "rear_total_s_wall",
                   "rear_center_minus_wall"]
    stats_df = stats_table(summary_df, metric_cols)

    stats_csv = REPORTS / "spatial_rearing_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"[write] {stats_csv.relative_to(ROOT)}")

    print("\n[stats — sorted by mw_p]")
    print(stats_df.sort_values("mw_p").to_string(index=False))

    print("\n[group summary]")
    for g in summary_df["group"].unique():
        sub = summary_df[summary_df["group"] == g]
        print(f"  {g:<22s}  n={len(sub):>2d}  "
              f"center_med={sub['rear_count_center'].median():.1f}  "
              f"wall_med={sub['rear_count_wall'].median():.1f}  "
              f"frac_med={sub['rear_center_frac'].median():.3f}")

    plot_boxplots(summary_df, stats_df, FIGS / "spatial_rearing_boxplot.png")
    plot_arena(bouts_df, inner, FIGS / "spatial_rearing_arena.png")

    # data/anxiety_features.csv'yi spatial sütunlarla genişletilmiş kopyaya yaz —
    # classifier --csv argümanı ile kullanabilir.
    feat_csv = DATA / "anxiety_features.csv"
    if feat_csv.exists():
        feats = pd.read_csv(feat_csv)
        spatial_cols = ["rear_count_center", "rear_count_wall",
                        "rear_center_frac", "rear_total_s_center",
                        "rear_total_s_wall", "rear_center_minus_wall"]
        merged = feats.merge(summary_df[["subject_id"] + spatial_cols],
                             on="subject_id", how="left")
        ext_csv = DATA / "anxiety_features_extended.csv"
        merged.to_csv(ext_csv, index=False)
        print(f"[write] {ext_csv.relative_to(ROOT)}  (+{len(spatial_cols)} spatial columns)")

    print("\n[done]")


if __name__ == "__main__":
    main()
