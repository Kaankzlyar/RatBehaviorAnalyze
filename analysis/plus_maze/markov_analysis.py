# -*- coding: utf-8 -*-
"""
markov_analysis.py
------------------
Plus Maze kol-gecis Markov analizi.

Her kohort icin 4x4 gecis olasilik matrisi hesaplar ve isi haritasi uretir.
Diagonal = persiverasyon (ayni koldan ayni kola), off-diagonal = alternasyon.

Ciktilar:
  reports/markov_transition_matrices.csv   -- uzun format (cohort, from, to, count, prob)
  reports/markov_transition_counts.csv     -- uzun format (ham sayilar)
  reports/figures/plus_maze/markov_heatmaps.png      -- 2x2 kohort isi haritasi
  reports/figures/plus_maze/markov_perseveration.png -- persiverasyon + alternasyon bar chart

Kullanim:
  python analysis/plus_maze/markov_analysis.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.family"]        = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"]         = 150

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DATA    = ROOT / "data" / "plus_maze_metrics_all.csv"
REPORTS = ROOT / "reports"
FIGS    = ROOT / "reports" / "figures" / "plus_maze"
FIGS.mkdir(parents=True, exist_ok=True)

ARMS          = ["B", "T", "L", "R"]
ARM_LABELS    = ["Alt Kol\n(kapali)", "Ust Kol\n(kapali)", "Sol Kol\n(acik)", "Sag Kol\n(acik)"]
COHORT_ORDER  = ["Control", "Aspartame", "Grapefruit", "ASP+Greyfurt"]
COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
}
COHORT_TR = {
    "Control":      "Kontrol",
    "Aspartame":    "Aspartam",
    "Grapefruit":   "Greyfurt",
    "ASP+Greyfurt": "ASP+Greyfurt",
}


# ─── yardimci fonksiyonlar ────────────────────────────────────────────────────

def build_transition_matrix(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Bir liste entry_sequence stringinden 4x4 ham sayim matrisi olusturur.
    Donus: (count_matrix, prob_matrix)
    """
    mat = np.zeros((4, 4), dtype=float)
    for seq_str in sequences:
        if not isinstance(seq_str, str) or not seq_str.strip():
            continue
        tokens = [t.strip() for t in seq_str.split("->")]
        for a, b in zip(tokens, tokens[1:]):
            if a in ARMS and b in ARMS:
                mat[ARMS.index(a), ARMS.index(b)] += 1

    # Satir normalize et
    row_sums = mat.sum(axis=1, keepdims=True)
    prob = np.divide(mat, row_sums, where=row_sums > 0, out=np.zeros_like(mat))
    return mat, prob


def perseveration_rate(prob_mat: np.ndarray) -> float:
    """Diagonal ortalamasini (persiverasyon orani) dondurur."""
    return float(np.mean(np.diag(prob_mat)))


def open_arm_preference(prob_mat: np.ndarray) -> float:
    """L ve R kollarini hedef alan toplam gecis olasiligini hesaplar."""
    l_idx, r_idx = ARMS.index("L"), ARMS.index("R")
    return float(prob_mat[:, l_idx].sum() + prob_mat[:, r_idx].sum()) / 4.0


# ─── CSV kaydet ───────────────────────────────────────────────────────────────

def save_matrices_csv(cohort_mats: dict) -> None:
    records = []
    for cohort, (count_mat, prob_mat) in cohort_mats.items():
        for i, arm_from in enumerate(ARMS):
            for j, arm_to in enumerate(ARMS):
                records.append({
                    "cohort":      cohort,
                    "from_arm":    arm_from,
                    "to_arm":      arm_to,
                    "count":       int(count_mat[i, j]),
                    "probability": round(float(prob_mat[i, j]), 4),
                })
    df_out = pd.DataFrame(records)

    prob_path  = REPORTS / "markov_transition_matrices.csv"
    df_out.to_csv(prob_path, index=False, encoding="utf-8-sig")
    print(f"[ok] {prob_path}")

    count_path = REPORTS / "markov_transition_counts.csv"
    df_out[["cohort", "from_arm", "to_arm", "count"]].to_csv(
        count_path, index=False, encoding="utf-8-sig"
    )
    print(f"[ok] {count_path}")


# ─── Grafik 1: 2x2 isi haritasi ──────────────────────────────────────────────

def plot_heatmaps(cohort_mats: dict) -> None:
    BG       = "#FFFFFF"
    GRID_COL = "#E0E0E0"
    TXT      = "#212121"

    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(hspace=0.68, wspace=0.36,
                        top=0.84, bottom=0.08, left=0.07, right=0.97)

    for ax, cohort in zip(axes.flat, COHORT_ORDER):
        if cohort not in cohort_mats:
            ax.set_visible(False)
            continue

        count_mat, prob_mat = cohort_mats[cohort]
        n_trans = int(count_mat.sum())
        perv    = perseveration_rate(prob_mat)

        annot_arr = np.array(
            [[f"{prob_mat[i,j]:.2f}\n(n={int(count_mat[i,j])})"
              for j in range(4)] for i in range(4)]
        )

        ax.set_facecolor(BG)
        sns.heatmap(
            prob_mat,
            ax=ax,
            annot=annot_arr,
            fmt="",
            cmap="Blues",
            vmin=0.0, vmax=1.0,
            xticklabels=ARM_LABELS,
            yticklabels=ARM_LABELS,
            linewidths=1.0,
            linecolor=GRID_COL,
            cbar_kws={"shrink": 0.80, "pad": 0.03},
            annot_kws={"size": 10, "color": TXT},
        )

        cbar = ax.collections[0].colorbar
        cbar.set_label("Gecis Olasiligi", fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8)

        # Diagonal: perseveration — thick red border per cell
        for k in range(4):
            ax.add_patch(plt.Rectangle(
                (k, k), 1, 1,
                fill=False, edgecolor="#D32F2F", lw=2.8, zorder=5,
            ))

        # L + R columns together: open arm — green dashed bracket
        ax.add_patch(plt.Rectangle(
            (2, 0), 2, 4,
            fill=False, edgecolor="#388E3C", lw=2.2,
            linestyle="--", zorder=5,
        ))

        # Kohort basligi — iki satir: isim + istatistik
        ax.set_title(
            f"{COHORT_TR[cohort]}\nn = {n_trans}   |   Persiverasyon ort: {perv:.2f}",
            fontsize=12, fontweight="bold",
            color=COHORT_COLORS[cohort], pad=10,
        )

        ax.set_xlabel("Hedef Kol",   fontsize=10, labelpad=7,  color=TXT)
        ax.set_ylabel("Kaynak Kol", fontsize=10, labelpad=7,  color=TXT)
        ax.tick_params(axis="both", labelsize=9.5, colors=TXT)

    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="#D32F2F", linewidth=2.8,
                       label="Diagonal = Persiverasyon  (ayni kol tekrari)"),
        mpatches.Patch(facecolor="none", edgecolor="#388E3C", linewidth=2.2,
                       linestyle="--", label="Acik kol sutunlari  (Sol, Sag)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2, fontsize=10,
        frameon=True, facecolor="#F5F5F5", edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, 0.008),
    )

    fig.suptitle(
        "Plus Maze  —  Markov Gecis Olasilik Matrisleri\n"
        "Satir: kaynak kol   |   Sutun: hedef kol   |   Deger: satir-normalize olasilik   (n = ham sayi)",
        fontsize=14, fontweight="bold", y=0.975, color="#1A237E",
    )

    out = FIGS / "markov_heatmaps.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[ok] {out}")


# ─── Grafik 2: Persiverasyon + alternasyon bar chart ─────────────────────────

def plot_perseveration_bars(cohort_mats: dict) -> None:
    BG  = "#FFFFFF"
    TXT = "#212121"

    cohorts      = [c for c in COHORT_ORDER if c in cohort_mats]
    perv_rates   = []
    open_rates   = []
    closed_rates = []
    total_trans  = []

    for c in cohorts:
        count_mat, prob_mat = cohort_mats[c]
        perv_rates.append(perseveration_rate(prob_mat))
        l_idx, r_idx = ARMS.index("L"), ARMS.index("R")
        b_idx, t_idx = ARMS.index("B"), ARMS.index("T")
        open_rates.append(float(prob_mat[:, l_idx].mean() + prob_mat[:, r_idx].mean()))
        closed_rates.append(float(prob_mat[:, b_idx].mean() + prob_mat[:, t_idx].mean()))
        total_trans.append(int(count_mat.sum()))

    x     = np.arange(len(cohorts))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.10, right=0.97)

    BAR_COLORS = {
        "perv":   "#E53935",
        "open":   "#388E3C",
        "closed": "#1565C0",
    }

    b1 = ax.bar(x - width, perv_rates,    width,
                label="Persiverasyon  (diagonal ort.)",
                color=BAR_COLORS["perv"],   alpha=0.88, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x,          open_rates,   width,
                label="Acik kola gecis  (Sol + Sag ort.)",
                color=BAR_COLORS["open"],   alpha=0.88, edgecolor="white", linewidth=0.8)
    b3 = ax.bar(x + width,  closed_rates, width,
                label="Kapali kola gecis  (Alt + Ust ort.)",
                color=BAR_COLORS["closed"], alpha=0.88, edgecolor="white", linewidth=0.8)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.014,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=TXT,
            )

    # Cohort-colored n_transitions labels below x-axis
    for xi, (c, n) in enumerate(zip(cohorts, total_trans)):
        ax.text(
            xi, -0.085, f"n = {n}",
            ha="center", va="top", fontsize=9,
            color=COHORT_COLORS[c], fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    ax.axhline(0.25, color="#9E9E9E", linestyle="--", linewidth=1.3,
               alpha=0.75, label="Esit olasilik esigi  (0.25)")

    ax.set_xticks(x)
    ax.set_xticklabels([COHORT_TR[c] for c in cohorts], fontsize=12)
    for tick_lbl, c in zip(ax.get_xticklabels(), cohorts):
        tick_lbl.set_color(COHORT_COLORS[c])
        tick_lbl.set_fontweight("bold")

    ax.set_ylabel("Ortalama Gecis Olasiligi", fontsize=11, color=TXT, labelpad=8)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    ax.set_title(
        "Markov Gecis Oranlari  —  Kohort Karsilastirmasi\n"
        "Persiverasyon: ayni kol tekrari   |   Acik / Kapali: hedef kol tercihi",
        fontsize=13, fontweight="bold", color="#1A237E", pad=10,
    )

    ax.legend(fontsize=9.5, frameon=True, facecolor="#F5F5F5",
              edgecolor="#CCCCCC", loc="upper right", ncol=1)
    ax.grid(axis="y", alpha=0.28, linestyle="--", color="#BDBDBD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.tick_params(axis="y", labelsize=9.5, colors=TXT)

    out = FIGS / "markov_perseveration.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[ok] {out}")


# ─── ozet tablo yazdir ───────────────────────────────────────────────────────

def print_summary(cohort_mats: dict) -> None:
    print("\n=== Markov Analizi Ozeti ===")
    print(f"{'Kohort':<16} {'n_gecis':>8} {'Perv.':>8} {'Acik->':>8} {'Kapali->':>9}")
    print("-" * 52)
    for c in COHORT_ORDER:
        if c not in cohort_mats:
            continue
        count_mat, prob_mat = cohort_mats[c]
        n_t = int(count_mat.sum())
        pv  = perseveration_rate(prob_mat)
        l_idx, r_idx = ARMS.index("L"), ARMS.index("R")
        b_idx, t_idx = ARMS.index("B"), ARMS.index("T")
        oa = float(prob_mat[:, l_idx].mean() + prob_mat[:, r_idx].mean())
        ca = float(prob_mat[:, b_idx].mean() + prob_mat[:, t_idx].mean())
        print(f"{c:<16} {n_t:>8} {pv:>8.3f} {oa:>8.3f} {ca:>9.3f}")
    print()
    print("Persiverasyon: ayni kol ard ardina tekrari (diagonal ort.)")
    print("Acik->        : L veya R hedefli gecis olasiligi ortalamasi")
    print("Kapali->      : B veya T hedefli gecis olasiligi ortalamasi")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DATA)
    print(f"Veri yuklendi: {len(df)} satir, kolonlar: {list(df.columns)}")

    cohort_mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cohort in COHORT_ORDER:
        grp = df[df["cohort"] == cohort]
        if grp.empty:
            print(f"[uyari] {cohort} bulunamadi, atlaniyor.")
            continue
        seqs = grp["entry_sequence"].dropna().tolist()
        count_mat, prob_mat = build_transition_matrix(seqs)
        cohort_mats[cohort] = (count_mat, prob_mat)
        total = int(count_mat.sum())
        print(f"[ok] {cohort}: {len(seqs)} sican, {total} toplam gecis islendi.")

    save_matrices_csv(cohort_mats)
    plot_heatmaps(cohort_mats)
    plot_perseveration_bars(cohort_mats)
    print_summary(cohort_mats)
    print("\n[done] Tum ciktilar reports/ ve reports/figures/ altinda.")


if __name__ == "__main__":
    main()
