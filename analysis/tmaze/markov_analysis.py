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
  reports/figures/markov_heatmaps.png      -- 2x2 kohort isi haritasi
  reports/figures/markov_perseveration.png -- persiverasyon + alternasyon bar chart

Kullanim:
  python analysis/tmaze/markov_analysis.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.family"]        = "Calibri"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"]         = 150

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DATA    = ROOT / "data" / "plus_maze_metrics_all.csv"
REPORTS = ROOT / "reports"
FIGS    = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

ARMS          = ["B", "T", "L", "R"]
ARM_LABELS    = ["Bottom\n(kapali)", "Top\n(kapali)", "Left\n(acik)", "Right\n(acik)"]
COHORT_ORDER  = ["Control", "Aspartame", "Grapefruit", "ASP+Greyfurt"]
COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.subplots_adjust(hspace=0.42, wspace=0.32,
                        top=0.91, bottom=0.06, left=0.06, right=0.98)

    for ax, cohort in zip(axes.flat, COHORT_ORDER):
        if cohort not in cohort_mats:
            ax.set_visible(False)
            continue

        count_mat, prob_mat = cohort_mats[cohort]
        total_transitions = int(count_mat.sum())

        # SHAP benzeri: persiverasyon kutulari kirmizi, acik kol kutulari yesil
        annot_arr = np.array(
            [[f"{prob_mat[i,j]:.2f}\n(n={int(count_mat[i,j])})"
              for j in range(4)] for i in range(4)]
        )

        sns.heatmap(
            prob_mat,
            ax=ax,
            annot=annot_arr,
            fmt="",
            cmap="Blues",
            vmin=0, vmax=1,
            xticklabels=ARM_LABELS,
            yticklabels=ARM_LABELS,
            linewidths=0.5,
            linecolor="#cccccc",
            cbar_kws={"shrink": 0.75, "label": "Gecis Olasiligi"},
        )

        # Diagonal (persiverasyon) kutusu kirmizi kenarlikla isaretler
        for k in range(4):
            ax.add_patch(plt.Rectangle(
                (k, k), 1, 1,
                fill=False, edgecolor="#C62828", lw=2.2, zorder=3
            ))

        # Acik kol sutunlari (L=2, R=3) yesil kenarlikla isaretler
        for col_idx in [2, 3]:
            ax.add_patch(plt.Rectangle(
                (col_idx, 0), 1, 4,
                fill=False, edgecolor="#2E7D32", lw=1.8,
                linestyle="--", zorder=3
            ))

        perv = perseveration_rate(prob_mat)
        ax.set_title(
            f"{cohort}  (n_gecis={total_transitions})\n"
            f"Persiverasyon ortalamasi: {perv:.2f}",
            fontsize=11, fontweight="bold", color=COHORT_COLORS[cohort],
            pad=8,
        )
        ax.set_xlabel("Hedef Kol (TO)", fontsize=10)
        ax.set_ylabel("Kaynak Kol (FROM)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    # Legend: kirmizi cizgi = persiverasyon, yesil cizgi = acik kol
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="#C62828", linewidth=2,
                       label="Diagonal = Persiverasyon (ayni kol)"),
        mpatches.Patch(facecolor="none", edgecolor="#2E7D32", linewidth=1.5,
                       linestyle="--", label="Acik kol sutunlari (L, R)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=2, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "Plus Maze — Markov Gecis Olasilik Matrisleri\n"
        "Satir: kaynak kol  |  Sutun: hedef kol  |  Deger: satir-normalize olasilik",
        fontsize=13, fontweight="bold", y=0.98,
    )

    out = FIGS / "markov_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── Grafik 2: Persiverasyon + alternasyon bar chart ─────────────────────────

def plot_perseveration_bars(cohort_mats: dict) -> None:
    """
    Her kohort icin:
      - Persiverasyon orani (diagonal ort.)
      - Acik kola gecis orani (L+R sutunlari)
      - Kapali kola gecis orani (B+T sutunlari)
    """
    cohorts = [c for c in COHORT_ORDER if c in cohort_mats]
    perv_rates   = []
    open_rates   = []
    closed_rates = []
    total_trans  = []

    for c in cohorts:
        _, prob_mat = cohort_mats[c]
        count_mat, _ = cohort_mats[c]
        perv_rates.append(perseveration_rate(prob_mat))

        # Acik kol hedefi: L(idx=2) ve R(idx=3) sutunlarinin ortalamalari
        l_idx, r_idx = ARMS.index("L"), ARMS.index("R")
        b_idx, t_idx = ARMS.index("B"), ARMS.index("T")
        open_target   = float(prob_mat[:, l_idx].mean() + prob_mat[:, r_idx].mean())
        closed_target = float(prob_mat[:, b_idx].mean() + prob_mat[:, t_idx].mean())
        open_rates.append(open_target)
        closed_rates.append(closed_target)
        total_trans.append(int(count_mat.sum()))

    x = np.arange(len(cohorts))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.10, right=0.97)

    b1 = ax.bar(x - width, perv_rates,   width, label="Persiverasyon (diagonal ort.)",
                color="#E53935", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,          open_rates,  width, label="Acik kola gecis orani (L+R ort.)",
                color="#2E7D32", alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + width,  closed_rates,width, label="Kapali kola gecis orani (B+T ort.)",
                color="#1565C0", alpha=0.85, edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # Toplam gecis sayisi etiketler
    for xi, (c, n) in enumerate(zip(cohorts, total_trans)):
        ax.text(xi, -0.055, f"n_gecis={n}", ha="center", va="top",
                fontsize=8, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, fontsize=11)
    ax.set_ylabel("Ortalama Gecis Olasiligi", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.25, color="#888888", linestyle="--", linewidth=1,
               alpha=0.5, label="Esit olasilik esigi (0.25)")
    ax.set_title(
        "Markov Gecis Oranlarinin Kohort Karsilastirmasi\n"
        "Persiverasyon: ayni kol tekrari | Acik/Kapali: hedef kol tercihi",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    out = FIGS / "markov_perseveration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
