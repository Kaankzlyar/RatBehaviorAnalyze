# -*- coding: utf-8 -*-
"""
plot_open_arm.py
----------------
EPM kohort karşılaştırması görsellerini üretir.

Grafik 1 — Box + strip: pct_open_arm (açık kol % süresi) kohort bazında
Grafik 2 — Box + strip: pct_open_arm_entries (açık kol % girişi)
Grafik 3 — Grouped bar: 4 kohort x 5 kol (bottom/left/right/top/junction) yüzdeleri
Grafik 4 — Scatter: pct_open_arm vs total_entries (lokomotor kovaryat)

Çıktılar:
  reports/figures/plus_maze/epm_open_arm_by_cohort.png
  reports/figures/plus_maze/epm_arm_distribution.png
  reports/figures/plus_maze/epm_locomotor_covariate.png

Kullanım:
  python analysis/plus_maze/plot_open_arm.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Türkçe karakter desteği için Calibri ──────────────────────────────────────
matplotlib.rcParams["font.family"]        = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"]         = 150

ROOT     = pathlib.Path(__file__).resolve().parent.parent.parent
DATA     = ROOT / "data" / "plus_maze_metrics_all.csv"
KW_PATH  = ROOT / "reports" / "cohort_epm_kw.csv"
FIGS     = ROOT / "reports" / "figures" / "plus_maze"
FIGS.mkdir(parents=True, exist_ok=True)

COHORT_ORDER  = ["Control", "Aspartame", "Grapefruit", "ASP+Greyfurt"]
COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
}


def strip_jitter(ax, data_by_group, colors, x_positions, jitter=0.10):
    """Her nokta = 1 sıçan. X ve Y yönünde hafif kaydırma ile örtüşme önlenir."""
    rng = np.random.default_rng(42)
    for xi, (grp, vals) in zip(x_positions, data_by_group.items()):
        jit_x = rng.uniform(-jitter, jitter, size=len(vals))
        # Aynı y değerindeki noktaları ayırt etmek için küçük y jitter
        jit_y = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + jit_x, vals + jit_y,
                   color=colors[grp], edgecolors="white",
                   linewidths=0.8, s=75, zorder=5, alpha=0.95)


def p_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "~"
    return "ns"


# ─── Grafik 1+2: Açık kol % süresi ve giriş ──────────────────────────────────

def plot_open_arm_box():
    df = pd.read_csv(DATA)
    kw = pd.read_csv(KW_PATH).set_index("feature") if KW_PATH.exists() else None

    metrics = [
        ("pct_open_arm",
         "% Açık Kol Süresi",
         "Açık Kol Süresi (Left + Right)"),
        ("pct_open_arm_entries",
         "% Açık Kol Girişi",
         "Açık Kol Girişi / Toplam Giriş"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.0))
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.97, wspace=0.32)

    for ax, (col, ylabel, title) in zip(axes, metrics):
        data_by_group = {
            g: df.loc[df["cohort"] == g, col].dropna().values
            for g in COHORT_ORDER if g in df["cohort"].values
        }
        x_pos  = list(range(len(data_by_group)))
        groups = list(data_by_group.keys())

        bp = ax.boxplot(
            [data_by_group[g] for g in groups],
            positions=x_pos,
            widths=0.48,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2.2),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            flierprops=dict(marker="", linestyle="none"),
        )
        for patch, grp in zip(bp["boxes"], groups):
            patch.set_facecolor(COHORT_COLORS[grp])
            patch.set_alpha(0.55)

        strip_jitter(ax, data_by_group, COHORT_COLORS, x_pos)

        if kw is not None and col in kw.index:
            p_val = kw.loc[col, "p_permutation"]
            ep2   = kw.loc[col, "epsilon_sq"]
            subtitle = (
                f"KW p = {p_val:.3f}  {p_stars(p_val)}     "
                f"ε² = {ep2:.3f}"
            )
        else:
            subtitle = ""

        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(bottom=-1)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        ax.axhline(25, color="#888888", linestyle="--",
                   linewidth=1, alpha=0.6, label="Eşit kol kullanımı (25%)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")

    # Kohort legend — figürün altında, x eksen etiketlerinin altında
    legend_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.80, label=g)
        for g in COHORT_ORDER
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.52, 0.01),
    )

    n_per_group = {g: len(data_by_group.get(g, [])) for g in COHORT_ORDER if g in data_by_group}
    n_str = "  |  ".join(f"{g} n={n}" for g, n in n_per_group.items())
    fig.suptitle(
        f"Plus Maze — EPM Açık Kol Analizi  ({n_str})\n"
        "Dikey kollar = kapalı kol  |  Yatay kollar = açık kol",
        fontsize=11, fontweight="bold", y=0.98,
    )
    fig.text(0.52, 0.085, "● Her nokta = 1 sıçan  |  Aynı değerdeki noktalar hafifçe kaydırılmıştır",
             ha="center", va="center", fontsize=9, color="#666666", style="italic")

    out = FIGS / "epm_open_arm_by_cohort.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── Grafik 3: Kol dağılımı stacked bar ──────────────────────────────────────

def plot_arm_distribution():
    df = pd.read_csv(DATA)
    arm_cols   = ["pct_time_bottom", "pct_time_left",
                  "pct_time_right",  "pct_time_top", "pct_time_junction"]
    arm_labels = [
        "Bottom (kapalı)",
        "Left (açık)",
        "Right (açık)",
        "Top (kapalı)",
        "Junction",
    ]
    arm_colors = ["#607D8B", "#FF9800", "#FF5722", "#455A64", "#9E9E9E"]

    means = (df.groupby("cohort")[arm_cols].mean()
               .reindex(COHORT_ORDER)
               .reset_index())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(right=0.78, top=0.88)

    x      = np.arange(len(COHORT_ORDER))
    bottom = np.zeros(len(COHORT_ORDER))

    # ── Largest-remainder düzeltmesi ─────────────────────────────────────────
    # Her kohortta 5 segment değerini 0.1% hassasiyetle yuvarla ve toplamı
    # tam %100 yapacak şekilde en büyük segmenti düzelt.
    # means'i cohort-indexed hâliyle kullan (reset_index öncesi)
    means_idx = means.set_index("cohort")
    display_map = {}  # (col, xi) -> gösterilecek float değer
    for xi, cohort in enumerate(COHORT_ORDER):
        if cohort not in means_idx.index:
            continue
        raw      = means_idx.loc[cohort, arm_cols].values.astype(float)
        rounded  = np.array([round(v, 1) for v in raw])
        residual = round(100.0 - float(rounded.sum()), 1)
        if residual != 0.0:
            largest_idx = int(np.argmax(rounded))
            rounded[largest_idx] = round(rounded[largest_idx] + residual, 1)
        for col_name, rv in zip(arm_cols, rounded):
            display_map[(col_name, xi)] = rv

    # Küçük segmentler için sonradan ok+etiket eklemek üzere sakla
    small_annotations = []  # (xi, seg_mid_y, value_str, color)

    for col, label, color in zip(arm_cols, arm_labels, arm_colors):
        vals = means[col].values
        ax.bar(x, vals, bottom=bottom, label=label,
               color=color, edgecolor="white", linewidth=0.8, alpha=0.88)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            seg_mid  = b + v / 2
            v_disp   = display_map.get((col, xi), round(v, 1))
            txt      = f"{v_disp:.1f}%"
            if v >= 3:
                ax.text(xi, seg_mid, txt,
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
            elif v >= 0.1:
                # Eşik altı segment: ok ile bar dışında göster
                small_annotations.append((xi, seg_mid, txt, color))
        bottom += vals

    # Küçük segmentler için annotate — ok bar'dan dışarı çıkıyor
    # Sağ tarafta yığılma yapmamak için xi'ye göre offset seç
    used_y = []  # çakışma önleme
    for xi, y_mid, txt, color in small_annotations:
        # Ok ucunun x konumu: barın sağ kenarı dışında
        x_tip  = xi + 0.26
        # y konumunu kullanılmış olanlardan kaçıracak şekilde ayarla
        y_txt = y_mid
        for used in used_y:
            if abs(y_txt - used) < 4:
                y_txt = used + 4
        used_y.append(y_txt)

        ax.annotate(
            txt,
            xy=(x_tip - 0.01, y_mid),        # ok ucu: segmentin ortası
            xytext=(x_tip + 0.35, y_txt),     # metin konumu
            fontsize=8.5, color="#333333", fontweight="bold",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=1.5,
                connectionstyle="arc3,rad=0.0",
            ),
            va="center", ha="left",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(COHORT_ORDER, fontsize=12)
    ax.set_ylabel("Ortalama Kol Süresi (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title(
        "Plus Maze Kol Dağılımı — Kohort Ortalaması\n"
        "Açık kollar: Left + Right  |  Kapalı kollar: Top + Bottom",
        fontsize=12, fontweight="bold",
    )
    ax.legend(
        loc="upper left",
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    out = FIGS / "epm_arm_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── Grafik 4: Scatter — açık kol vs lokomotor ───────────────────────────────

def plot_locomotor_covariate():
    df = pd.read_csv(DATA)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(top=0.80, bottom=0.12, wspace=0.30)

    pairs = [
        ("total_entries",   "pct_open_arm",
         "Toplam Giriş (Lokomotor Kovaryat)", "% Açık Kol Süresi"),
        ("mean_speed_px_s", "pct_open_arm",
         "Ortalama Hız (px/s)",               "% Açık Kol Süresi"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        for _, row in df.iterrows():
            grp = row["cohort"]
            if grp not in COHORT_COLORS:
                continue
            ax.scatter(row[xcol], row[ycol],
                       color=COHORT_COLORS[grp],
                       s=95, edgecolors="white", linewidths=0.9,
                       zorder=4, alpha=0.92)
            label_txt = str(row["subject_id"]).replace("PlusMaze", "")
            ax.annotate(
                label_txt,
                (row[xcol], row[ycol]),
                textcoords="offset points", xytext=(6, 3),
                fontsize=8, color="#555555",
            )

        valid = df[[xcol, ycol]].dropna()
        if len(valid) > 2:
            from scipy.stats import pearsonr
            r, pval = pearsonr(valid[xcol], valid[ycol])
            stars = p_stars(pval)
            ax.set_title(
                f"Pearson  r = {r:.2f}   p = {pval:.3f}  {stars}",
                fontsize=11,
            )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.85, label=g)
        for g in COHORT_ORDER
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=4, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "Açık Kol Süresi vs Lokomotor Kovaryat\n"
        "(Cruz 1994: saf anksiyolitik etki = açık kol ↑ + lokomotor sabit)",
        fontsize=12, fontweight="bold", y=1.04,
    )

    out = FIGS / "epm_locomotor_covariate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("EPM grafikleri oluşturuluyor...\n")
    plot_open_arm_box()
    plot_arm_distribution()
    plot_locomotor_covariate()
    print("\n[done] tüm grafikler reports/figures/ altında")
