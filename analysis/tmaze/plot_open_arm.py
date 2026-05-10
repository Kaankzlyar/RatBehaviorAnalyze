"""
plot_open_arm.py
----------------
EPM cohort karsilastirmasi gorsellerini uretir.

Grafik 1 — Box + strip: pct_open_arm (acik kol % suresi) kohort bazinda
Grafik 2 — Box + strip: pct_open_arm_entries (acik kol % girisi)
Grafik 3 — Grouped bar: 4 kohort x 5 kol (bottom/left/right/top/junction) yuzdeleri
Grafik 4 — Scatter: pct_open_arm vs total_entries (lokomotor kovariat)

Ciktilar:
  reports/figures/epm_open_arm_by_cohort.png
  reports/figures/epm_arm_distribution.png
  reports/figures/epm_locomotor_covariate.png

Kullanim:
  python analysis/tmaze/plot_open_arm.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent.parent
DATA     = ROOT / "data" / "plus_maze_metrics_all.csv"
KW_PATH  = ROOT / "reports" / "cohort_epm_kw.csv"
FIGS     = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

COHORT_ORDER  = ["Control", "Aspartame", "Grapefruit", "ASP+Greyfurt"]
COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
}


def strip_jitter(ax, data_by_group, colors, x_positions, jitter=0.08):
    """Her grubun noktalarini hafif kaydirarak ciz."""
    rng = np.random.default_rng(42)
    for xi, (grp, vals) in zip(x_positions, data_by_group.items()):
        jit = rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(xi + jit, vals,
                   color=colors[grp], edgecolors="white",
                   linewidths=0.6, s=60, zorder=5, alpha=0.9)


def p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "~"
    return "ns"


def add_significance_bar(ax, x1, x2, y, p, h=0.5):
    """Iki grup arasina anlamlilik cubugu ekle."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h + 0.1, p_stars(p),
            ha="center", va="bottom", fontsize=11)


# ─── Grafik 1+2: Open arm % suresi ve giris ───────────────────────────────────

def plot_open_arm_box():
    df = pd.read_csv(DATA)
    kw = pd.read_csv(KW_PATH).set_index("feature") if KW_PATH.exists() else None

    metrics = [
        ("pct_open_arm",         "% Acik Kol Suresi",  "Acik kol (left + right)"),
        ("pct_open_arm_entries", "% Acik Kol Girisi",  "Acik kol giriş / toplam giriş"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (col, ylabel, title) in zip(axes, metrics):
        data_by_group = {
            g: df.loc[df["cohort"] == g, col].dropna().values
            for g in COHORT_ORDER if g in df["cohort"].values
        }
        x_pos = list(range(len(data_by_group)))
        groups = list(data_by_group.keys())

        # Boxplot
        bp = ax.boxplot(
            [data_by_group[g] for g in groups],
            positions=x_pos,
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="", linestyle="none"),
        )
        for patch, grp in zip(bp["boxes"], groups):
            patch.set_facecolor(COHORT_COLORS[grp])
            patch.set_alpha(0.6)

        strip_jitter(ax, data_by_group, COHORT_COLORS, x_pos)

        # p degeri annotation
        if kw is not None and col in kw.index:
            p_val = kw.loc[col, "p_permutation"]
            ep2   = kw.loc[col, "epsilon_sq"]
            ax.set_title(
                f"{title}\n"
                f"KW p={p_val:.3f} {p_stars(p_val)}   "
                f"ε²={ep2:.3f}",
                fontsize=11, fontweight="bold"
            )
        else:
            ax.set_title(title, fontsize=11, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(bottom=-1)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Referans cizgisi: chance = 50% (eger tum kollar esit kullanilsaydi)
        ax.axhline(50 / 2, color="gray", linestyle="--",
                   linewidth=1, alpha=0.5, label="Esit kol kullanimi (50%)")

    legend_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.7, label=g)
        for g in COHORT_ORDER
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Plus Maze — EPM Acik Kol Analizi (n=3/grup)\n"
        "Dikey kollar = kapali kol | Yatay kollar = acik kol",
        fontsize=13, fontweight="bold", y=1.05
    )
    plt.tight_layout()
    out = FIGS / "epm_open_arm_by_cohort.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── Grafik 3: Kol dagilimi stacked bar ───────────────────────────────────────

def plot_arm_distribution():
    df = pd.read_csv(DATA)
    arm_cols = ["pct_time_bottom", "pct_time_left",
                "pct_time_right", "pct_time_top", "pct_time_junction"]
    arm_labels = ["Bottom\n(kapali)", "Left\n(acik)", "Right\n(acik)",
                  "Top\n(kapali)", "Junction"]
    arm_colors = ["#607D8B", "#FF9800", "#FF5722", "#607D8B", "#9E9E9E"]

    means = (df.groupby("cohort")[arm_cols].mean()
               .reindex(COHORT_ORDER)
               .reset_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(COHORT_ORDER))
    bottom = np.zeros(len(COHORT_ORDER))

    for col, label, color in zip(arm_cols, arm_labels, arm_colors):
        vals = means[col].values
        ax.bar(x, vals, bottom=bottom, label=label,
               color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 3:
                ax.text(xi, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(COHORT_ORDER, fontsize=11)
    ax.set_ylabel("Ortalama Kol Suresi (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Plus Maze Kol Dagilimi — Kohort Ortalamasi\n"
                 "Acik kollar: Left + Right | Kapali kollar: Top + Bottom",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=True,
              bbox_to_anchor=(1.18, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGS / "epm_arm_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── Grafik 4: Scatter — acik kol vs lokomotor ────────────────────────────────

def plot_locomotor_covariate():
    df = pd.read_csv(DATA)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    pairs = [
        ("total_entries",   "pct_open_arm",
         "Toplam Giris (lokomotor)", "% Acik Kol Suresi"),
        ("mean_speed_px_s", "pct_open_arm",
         "Ortalama Hiz (px/s)",      "% Acik Kol Suresi"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        for _, row in df.iterrows():
            grp = row["cohort"]
            if grp not in COHORT_COLORS:
                continue
            ax.scatter(row[xcol], row[ycol],
                       color=COHORT_COLORS[grp],
                       s=90, edgecolors="white", linewidths=0.8,
                       zorder=4, alpha=0.9)
            ax.annotate(row["subject_id"].replace("PlusMaze", ""),
                        (row[xcol], row[ycol]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="gray")

        # Pearson r
        valid = df[[xcol, ycol]].dropna()
        if len(valid) > 2:
            from scipy.stats import pearsonr
            r, pval = pearsonr(valid[xcol], valid[ycol])
            ax.set_title(f"r = {r:.2f}  p = {pval:.3f}", fontsize=10)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.8, label=g)
        for g in COHORT_ORDER
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Acik Kol Suresi vs Lokomotor Kovariat\n"
                 "(Cruz 1994: saf anksiyolitik = acik kol artisi + sabit lokomotor)",
                 fontsize=11, fontweight="bold", y=1.06)
    plt.tight_layout()
    out = FIGS / "epm_locomotor_covariate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("EPM grafikleri olusturuluyor...\n")
    plot_open_arm_box()
    plot_arm_distribution()
    plot_locomotor_covariate()
    print("\n[done] tum grafikleri reports/figures/ altinda")
