# -*- coding: utf-8 -*-
"""
ethological_features.py
-----------------------
Plus Maze DLC verisinden pose-bazli etolojik davranis tespiti.

Tespit edilen davranislar:
  Grooming  — Kendini temizleme: on pati buruna yakin + hareketsiz
  SAP       — Stretch-Attend Posture (uzanma-degerlendirme): vucut uzuyor + hareketsiz

NOT: Rearing (dikine kalkma) ustten kamera kaydinda guvenilir olarak tespit
edilemeдигinden (vucut kisalmasi grooming/kivrилma ile karisabilir) kapsam
disinda birakilmistir. Gelecek calisma: yan kamera ile dogrulanabilir.

Yontem:
  - Her kare (frame) icin binarize edilir (0/1)
  - Min bout suresi uygulanir, kisa araliklar birlestirilir
  - Kohort bazinda kutu grafikleri uretilir

Esikler (12 sican havuzundan hesaplandi):
  GROOMING : forepaw-nose < 23px  AND  hiz < 0.50 px/kare
  SAP      : vucut boyu > 102px   AND  hiz < 0.50 px/kare
  Minimum bout: 0.5s (15 kare @ 30fps)
  Birlestirme araligi: 0.33s (10 kare @ 30fps)

Ciktilar:
  reports/ethological_metrics_epm.csv
  reports/figures/ethological_grooming.png
  reports/figures/ethological_sap.png
  reports/figures/ethological_summary.png

Kullanim:
  python analysis/tmaze/ethological_features.py
"""

import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"]        = "Calibri"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"]         = 150
matplotlib.rcParams["pdf.fonttype"]       = 42
matplotlib.rcParams["ps.fonttype"]        = 42

# ── Proje yolu ─────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DLC_DIR = ROOT / "data" / "DLCfiltered"
REPORTS = ROOT / "reports"
FIGS    = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Sabitler ──────────────────────────────────────────────────────────────────
FPS                  = 30
LIKELIHOOD_THRESH    = 0.6

# Davranis esikleri (px cinsinden; 12 sican havuzundan)
GROOMING_FPN_THRESH  = 23.0   # forepaw-nose mesafesi p25
SPEED_STILL_THRESH   = 0.50   # px/kare — "hareketsiz" siniri
SAP_BL_THRESH        = 102.0  # vucut boyu p75 (uzama = SAP)

MIN_BOUT_FRAMES      = 15     # 0.5s
MERGE_GAP_FRAMES     = 10     # 0.33s

COHORT_ORDER  = ["Control", "Aspartame", "Grapefruit", "ASP+Greyfurt"]
COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
}
COHORT_DISPLAY = {
    "Control":      "Kontrol",
    "Aspartame":    "Aspartam",
    "Grapefruit":   "Greyfurt",
    "ASP+Greyfurt": "Aspartam+Greyfurt",
}

# subject_id -> cohort eslestirmesi (plus_maze_metrics_all.csv'den)
def load_cohort_map() -> dict:
    pm = pd.read_csv(ROOT / "data" / "plus_maze_metrics_all.csv")
    return dict(zip(pm["subject_id"].str.replace("PlusMaze", ""), pm["cohort"]))


# ── DLC CSV okuyucu ───────────────────────────────────────────────────────────

def load_dlc(csv_path: pathlib.Path) -> pd.DataFrame:
    """3 satirlik baslik (scorer/bodypart/coord) ile yukle, bodypart+coord MultiIndex dondur."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples([(b, c) for _, b, c in df.columns])
    return df


def get_keypoint(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """x, y dizilerini dondurur; likelihood < esik olan kareler NaN yapilir."""
    x  = df[name]["x"].astype(float).values.copy()
    y  = df[name]["y"].astype(float).values.copy()
    lk = df[name]["likelihood"].astype(float).values
    mask = lk < LIKELIHOOD_THRESH
    x[mask] = np.nan
    y[mask] = np.nan
    return x, y


# ── Mesafe hesaplama ──────────────────────────────────────────────────────────

def dist(x1, y1, x2, y2) -> np.ndarray:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ── Bout tespit motoru ────────────────────────────────────────────────────────

def detect_bouts(signal: np.ndarray,
                 min_frames: int = MIN_BOUT_FRAMES,
                 merge_gap: int  = MERGE_GAP_FRAMES) -> list[tuple[int, int]]:
    """
    1D binary (0/1) sinyalden bout baslangic-bitis listesi dondurur.
    Kisa araliklar (< merge_gap) birlestirilerek parlak bir sure filtresi uygulanir.
    """
    # NaN -> 0
    sig = np.nan_to_num(signal.astype(float)).astype(bool)

    # Baslangic ve bitis karelerini bul
    padded  = np.concatenate([[0], sig, [0]])
    changes = np.diff(padded.astype(int))
    starts  = np.where(changes == 1)[0]
    ends    = np.where(changes == -1)[0]   # exclusive

    bouts = list(zip(starts, ends))

    # Kisa aralik birlestirme
    merged = []
    for s, e in bouts:
        if merged and (s - merged[-1][1]) <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    # Min sure filtresi
    bouts_filtered = [(s, e) for s, e in merged if (e - s) >= min_frames]
    return bouts_filtered


# ── Metrik ozeti ──────────────────────────────────────────────────────────────

def summarize_bouts(bouts: list[tuple[int, int]],
                    total_frames: int,
                    fps: float = FPS) -> dict:
    n          = len(bouts)
    total_s    = sum(e - s for s, e in bouts) / fps
    mean_s     = (total_s / n) if n > 0 else 0.0
    max_s      = (max(e - s for s, e in bouts) / fps) if n > 0 else 0.0
    pct_time   = total_s / (total_frames / fps) * 100 if total_frames > 0 else 0.0
    frag_idx   = n / total_s if total_s > 0 else 0.0   # bouts/s
    return {
        "bout_count": n,
        "total_s":    round(total_s, 2),
        "pct_time":   round(pct_time, 2),
        "mean_s":     round(mean_s, 2),
        "max_s":      round(max_s, 2),
        "frag_idx":   round(frag_idx, 3),
    }


# ── Ana islem: bir sican ──────────────────────────────────────────────────────

def process_subject(csv_path: pathlib.Path) -> dict:
    df  = load_dlc(csv_path)
    n   = len(df)

    nx, ny   = get_keypoint(df, "nose")
    tx, ty   = get_keypoint(df, "tail_base")
    bx, by   = get_keypoint(df, "body_center")
    lfx, lfy = get_keypoint(df, "left_forepaw")
    rfx, rfy = get_keypoint(df, "right_forepaw")

    # Vucut boyu: nose -> tail_base
    body_len = dist(nx, ny, tx, ty)

    # Forepaw-nose: her iki patinin minimumu
    fpn = np.fmin(dist(lfx, lfy, nx, ny), dist(rfx, rfy, nx, ny))

    # Hiz: body_center (px/kare); iki kare arasi
    spd_full = np.concatenate([[np.nan], np.sqrt(np.diff(bx)**2 + np.diff(by)**2)])

    # NaN yokken geçerlilik maskesi
    valid = ~np.isnan(body_len) & ~np.isnan(fpn) & ~np.isnan(spd_full)

    # Grooming: forepaw buruna yakin + hareketsiz
    groom_sig = valid & (fpn < GROOMING_FPN_THRESH) & (spd_full < SPEED_STILL_THRESH)

    # SAP: vucut uzar + hareketsiz
    sap_sig   = valid & (body_len > SAP_BL_THRESH) & (spd_full < SPEED_STILL_THRESH)

    groom_bouts = detect_bouts(groom_sig.astype(float))
    sap_bouts   = detect_bouts(sap_sig.astype(float))

    result = {"n_frames": n, "session_s": round(n / FPS, 1)}
    for prefix, bouts in [("groom", groom_bouts), ("sap", sap_bouts)]:
        for k, v in summarize_bouts(bouts, n).items():
            result[f"{prefix}_{k}"] = v

    # Ham sinyal yuzdeleri (esik kontrolu icin)
    result["groom_signal_pct"] = round(float(np.nanmean(groom_sig)) * 100, 2)
    result["sap_signal_pct"]   = round(float(np.nanmean(sap_sig))   * 100, 2)

    return result


# ── Tum sicanlari isle ────────────────────────────────────────────────────────

def run_all() -> pd.DataFrame:
    cohort_map = load_cohort_map()
    pm_csvs    = sorted([
        p for p in DLC_DIR.rglob("*.csv")
        if "PlusMaze" in p.name and "metrics" not in p.name
    ])

    rows = []
    for csv_path in pm_csvs:
        m = re.search(r"PlusMaze(MA\d+_\d+)", csv_path.name, re.IGNORECASE)
        if not m:
            continue
        subj_id = m.group(1)
        cohort  = cohort_map.get(subj_id, "Unknown")

        print(f"  {subj_id:10s} ({cohort}) ...", end=" ")
        try:
            metrics = process_subject(csv_path)
            metrics["subject_id"] = subj_id
            metrics["cohort"]     = cohort
            rows.append(metrics)
            print(f"grooming={metrics['groom_bout_count']}bouts/{metrics['groom_total_s']:.1f}s  "
                  f"SAP={metrics['sap_bout_count']}bouts/{metrics['sap_total_s']:.1f}s")
        except Exception as e:
            print(f"HATA: {e}")

    df = pd.DataFrame(rows)
    cols = ["subject_id", "cohort", "n_frames", "session_s"] + [
        c for c in df.columns if c not in {"subject_id", "cohort", "n_frames", "session_s"}
    ]
    df = df[cols].sort_values(["cohort", "subject_id"])
    return df


# ── Gorsellestime: Secenek B — nokta + ortalama cizgisi + SD golgesi ─────────

def draw_dot_mean_sd(ax, xi: int, vals: np.ndarray, color: str,
                     rng: np.random.Generator,
                     w: float = 0.28, jitter: float = 0.12) -> None:
    """
    Tek bir grup icin:
      - SD golgesi (arka plan dikdortgen)
      - Ortalama yatay cizgisi
      - Ortalama elmas marker (4. nokta)
      - Bireysel noktalar (jittered)
      - Bireysel -> ortalama baglanti cizgileri
    """
    mean_v = float(np.mean(vals))
    sd_v   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # SD golgesi
    rect = plt.Rectangle(
        (xi - w, mean_v - sd_v), 2 * w, 2 * sd_v,
        color=color, alpha=0.13, zorder=1, linewidth=0,
    )
    ax.add_patch(rect)

    # Ortalama cizgisi
    ax.hlines(mean_v, xi - w, xi + w, colors=color, linewidth=3.0, zorder=3)

    # Bireysel noktalar
    jit_x = rng.uniform(-jitter, jitter, size=len(vals))
    for v, jx in zip(vals, jit_x):
        ax.plot([xi + jx, xi], [v, mean_v],
                color=color, alpha=0.28, linewidth=1.2, zorder=2)

    ax.scatter(xi + jit_x, vals,
               color=color, s=115, edgecolors="white",
               linewidths=1.3, zorder=5, alpha=0.95)

    # Ortalama — dolgu elmas (4. nokta)
    ax.scatter(xi, mean_v, marker="D", color="white", s=90,
               edgecolors=color, linewidths=2.2, zorder=6)


def plot_behavior_panel(df: pd.DataFrame,
                        metric_col: str,
                        ylabel: str,
                        title: str,
                        out_path: pathlib.Path) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.12, right=0.97)

    groups   = [g for g in COHORT_ORDER if g in df["cohort"].values]
    x_pos    = list(range(len(groups)))
    rng      = np.random.default_rng(42)

    for xi, grp in zip(x_pos, groups):
        vals = df.loc[df["cohort"] == grp, metric_col].dropna().values
        draw_dot_mean_sd(ax, xi, vals, COHORT_COLORS[grp], rng)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([COHORT_DISPLAY[g] for g in groups], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Sag ust: kohort renk göstergesi (Türkçe isimler)
    cohort_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.8,
                       label=COHORT_DISPLAY[g])
        for g in COHORT_ORDER
    ]
    ax.legend(handles=cohort_patches, loc="upper right", fontsize=10, frameon=False)

    # Alt: semboller için özel legend
    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
               markersize=11, label="= Fare"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor="#555555", markeredgewidth=2.2,
               markersize=10, label="= Grup Ortalaması"),
        Line2D([0], [0], color="#555555", linewidth=2.5, label="= Ortalama"),
    ]
    fig.legend(
        handles=marker_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


# ── Ozet 3-panel grafik ───────────────────────────────────────────────────────

def plot_summary(df: pd.DataFrame) -> None:
    metrics = [
        ("groom_pct_time", "% Oturum Süresi", "Grooming — Kendini Temizleme"),
        ("sap_pct_time",   "% Oturum Süresi", "SAP — Uzanma-Değerlendirme Duruşu"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.06, right=0.98, wspace=0.35)

    groups = [g for g in COHORT_ORDER if g in df["cohort"].values]
    x_pos  = list(range(len(groups)))

    rng = np.random.default_rng(42)
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for xi, grp in zip(x_pos, groups):
            vals = df.loc[df["cohort"] == grp, col].dropna().values
            draw_dot_mean_sd(ax, xi, vals, COHORT_COLORS[grp], rng)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [COHORT_DISPLAY[g] for g in groups],
            fontsize=9, rotation=0, ha="center",
        )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.8,
                       label=COHORT_DISPLAY[g])
        for g in COHORT_ORDER
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=4, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Plus Maze",
        fontsize=11, fontweight="bold", y=0.99,
    )

    out = FIGS / "ethological_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ── Kohort tablosu yazdir ─────────────────────────────────────────────────────

def print_cohort_summary(df: pd.DataFrame) -> None:
    print("\n=== Kohort Ortalamalari ===")
    cols = ["groom_bout_count", "groom_pct_time", "groom_mean_s",
            "sap_bout_count",   "sap_pct_time",   "sap_mean_s"]
    grp = df.groupby("cohort")[cols].mean().round(2).reindex(COHORT_ORDER)
    print(grp.to_string())
    print()
    print("Ham sinyal yuzdeleri (esik kontrolu):")
    print(df.groupby("cohort")[["groom_signal_pct", "sap_signal_pct"]].mean().round(1).reindex(COHORT_ORDER).to_string())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Plus Maze etolojik ozellikler hesaplaniyor...\n")
    print(f"Esikler: grooming forepaw-nose<{GROOMING_FPN_THRESH}px | "
          f"SAP body>{SAP_BL_THRESH}px | "
          f"hiz<{SPEED_STILL_THRESH}px/kare | min_bout={MIN_BOUT_FRAMES}kare ({MIN_BOUT_FRAMES/FPS:.1f}s)\n")

    df = run_all()

    # CSV kaydet
    out_csv = REPORTS / "ethological_metrics_epm.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {out_csv}")

    print_cohort_summary(df)

    # Grafik: her davranis icin ayri panel
    plot_behavior_panel(df, "groom_pct_time", "% Oturum Süresi",
                        "Grooming — Kendini Temizleme",
                        FIGS / "ethological_grooming.png")
    plot_behavior_panel(df, "sap_pct_time", "% Oturum Süresi",
                        "SAP — Uzanma-Değerlendirme Duruşu",
                        FIGS / "ethological_sap.png")
    plot_summary(df)

    print("\n[done] Tum ciktilar reports/ ve reports/figures/ altinda.")


if __name__ == "__main__":
    main()
