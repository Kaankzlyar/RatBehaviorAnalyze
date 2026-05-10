# -*- coding: utf-8 -*-
"""
behavioral_effect_analysis.py
------------------------------
4 kohort için OFT + EPM verilerini birleştirerek madde etkilerini ölçer.

Adım 1 — Cohen's d etki büyüklüğü tablosu  (her madde vs Control)
Adım 2 — Radar (örümcek ağı) grafiği       (8 metrik, 4 grup)
Adım 3 — % değişim + etki büyüklüğü bar chart

Çıktı klasörü: reports/effect_analysis/
  behavioral_effect_table.csv
  behavioral_radar.png
  behavioral_effect_bars.png
"""

import pathlib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams["font.family"]        = "Calibri"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "reports" / "effect_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_ORDER  = ["Control", "Aspartame", "Grapefruit", "ASP+Greyfurt"]
SUBSTANCES    = ["Aspartame", "Grapefruit", "ASP+Greyfurt"]
COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
}

# MA1=Control, MA3=Aspartame, MA5=Grapefruit, MA7=ASP+Greyfurt
COHORT_MAP = {
    "MA1": "Control", "MA3": "Aspartame",
    "MA5": "Grapefruit", "MA7": "ASP+Greyfurt",
}

# ── Veri yükle ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    pm  = pd.read_csv(ROOT / "data" / "plus_maze_metrics_all.csv")
    oft = pd.read_csv(ROOT / "data" / "oft_metrics_all.csv")

    # Sadece 4 ana kohort (plus maze ile eşleşen)
    oft = oft[oft["cohort"].isin(COHORT_MAP)].copy()
    oft["cohort4"] = oft["cohort"].map(COHORT_MAP)
    pm["cohort4"]  = pm["cohort"]

    # Subject ID eşleştirmesi: PlusMazeMA1_1 -> MA1_1
    pm["subject_id_short"] = pm["subject_id"].str.replace("PlusMaze", "", regex=False)
    oft["subject_id_short"] = oft["subject_id"]

    pm_cols = ["subject_id_short", "pct_open_arm", "pct_open_arm_entries",
               "anxiety_index_epm", "total_entries"]

    # Birleştir — suffixes ile çakışan sütunları ayırt et
    df = oft.merge(
        pm[pm_cols],
        on="subject_id_short",
        how="inner",
        suffixes=("", "_pm"),
    )
    # cohort4 OFT'den gelen ile aynı — pm'den gelenini düşür
    df = df.drop(columns=[c for c in df.columns if c.endswith("_pm")], errors="ignore")

    return df


# ── İstatistik yardımcıları ───────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: (mean_a - mean_b) / pooled_std"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    var_pool = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sd_pool  = np.sqrt(var_pool) if var_pool > 0 else np.nan
    return (a.mean() - b.mean()) / sd_pool if sd_pool and not np.isnan(sd_pool) else np.nan


def d_label(d: float) -> str:
    if np.isnan(d):      return "—"
    ad = abs(d)
    if ad < 0.2:         return "önemsiz"
    if ad < 0.5:         return "küçük"
    if ad < 0.8:         return "orta"
    if ad < 1.4:         return "büyük"
    return "çok büyük"


def effect_direction(col: str, d: float) -> str:
    """Pozitif d = madde grubu kontrolden YÜKSEK."""
    increase_good = {
        "pct_open_arm", "pct_open_arm_entries", "anxiety_index_epm",
        "pct_time_center", "total_distance_px", "mean_speed_px_s",
        "rear_pct_time",
    }
    decrease_good = {"pct_time_freeze", "pct_time_periphery"}
    if np.isnan(d):
        return "—"
    if col in increase_good:
        return "anksiyolitik ↓" if d > 0 else "anksiyojenik ↑"
    if col in decrease_good:
        return "anksiyolitik ↓" if d < 0 else "anksiyojenik ↑"
    return "↑" if d > 0 else "↓"


# ─── ADIM 1: Cohen's d etki büyüklüğü tablosu ────────────────────────────────

METRICS = {
    # EPM metrikleri
    "pct_open_arm":          "Açık Kol Süresi % (EPM)",
    "pct_open_arm_entries":  "Açık Kol Girişi % (EPM)",
    "anxiety_index_epm":     "Anksiyete İndeksi (EPM)",
    # OFT — anksiyete
    "pct_time_freeze":       "Donma Süresi % (OFT)",
    "pct_time_center":       "Merkez Zamanı % (OFT)",
    "pct_time_periphery":    "Çevre Zamanı % (OFT)",
    # OFT — lokomotor
    "total_distance_px":     "Toplam Mesafe (OFT)",
    "mean_speed_px_s":       "Ortalama Hız (OFT)",
    # OFT — davranış
    "rear_pct_time":         "Rearing % (OFT)",
    "groom_pct_time":        "Grooming % (OFT)",
}


def build_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    ctrl = df[df["cohort4"] == "Control"]
    rows = []
    for col, label in METRICS.items():
        if col not in df.columns:
            continue
        ctrl_vals = ctrl[col].dropna().values
        ctrl_mean = ctrl_vals.mean()
        ctrl_std  = ctrl_vals.std(ddof=1) if len(ctrl_vals) > 1 else np.nan

        for sub in SUBSTANCES:
            sub_vals = df[df["cohort4"] == sub][col].dropna().values
            if len(sub_vals) == 0:
                continue
            sub_mean = sub_vals.mean()
            pct_chg  = ((sub_mean - ctrl_mean) / ctrl_mean * 100
                        if ctrl_mean != 0 else np.nan)
            d        = cohens_d(sub_vals, ctrl_vals)
            rows.append({
                "metrik":           label,
                "madde":            sub,
                "kontrol_ort":      round(ctrl_mean, 2),
                "madde_ort":        round(sub_mean, 2),
                "pct_degisim":      round(pct_chg, 1) if not np.isnan(pct_chg) else np.nan,
                "cohens_d":         round(d, 2) if not np.isnan(d) else np.nan,
                "etki_buyuklugu":   d_label(d),
                "yon":              effect_direction(col, d),
            })
    return pd.DataFrame(rows)


# ─── ADIM 2: Radar grafiği ────────────────────────────────────────────────────

RADAR_METRICS = [
    ("pct_open_arm",       "Açık Kol\nSüresi %",      True),
    ("pct_open_arm_entries","Açık Kol\nGirişi %",     True),
    ("anxiety_index_epm",  "Anksiyete\nİndeksi",      True),
    ("pct_time_center",    "Merkez\nZamanı %",        True),
    ("pct_time_freeze",    "Donma %\n(ters)",         False),  # ters: az = iyi
    ("total_distance_px",  "Lokomotor\nMesafe",       True),
    ("rear_pct_time",      "Rearing %",               True),
    ("groom_pct_time",     "Grooming %",              True),
]


def plot_radar(df: pd.DataFrame):
    group_means = (df.groupby("cohort4")
                     [[m for m, _, _ in RADAR_METRICS]]
                     .mean()
                     .reindex(COHORT_ORDER))

    labels   = [lbl for _, lbl, _ in RADAR_METRICS]
    n        = len(RADAR_METRICS)
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles  += angles[:1]  # kapat

    # Min-max normalize (0-1), donma ters çevrilir
    norm_data = {}
    for col, _, is_higher_better in RADAR_METRICS:
        vals = group_means[col].values.astype(float)
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx != mn else 1.0
        normalized = (vals - mn) / rng
        if not is_higher_better:
            normalized = 1 - normalized  # ters: az donma = dışa doğru
        norm_data[col] = normalized

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7.5, color="#999999")
    ax.set_ylim(0, 1)
    ax.grid(color="#cccccc", linestyle="--", linewidth=0.7)
    ax.spines["polar"].set_visible(False)

    for i, cohort in enumerate(COHORT_ORDER):
        values = [norm_data[col][i] for col, _, _ in RADAR_METRICS]
        values += values[:1]
        color = COHORT_COLORS[cohort]
        ax.plot(angles, values, color=color, linewidth=2.2, label=cohort)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.30, 1.15),
        fontsize=11, frameon=False,
    )
    fig.suptitle(
        "Davranışsal Etki Profili — 4 Kohort\n"
        "Dışa doğru = Kontrole göre daha fazla etki / daha az anksiyete\n"
        "(Donma % ters çevrildi: dışa = az donma = düşük anksiyete)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    out = OUT_DIR / "behavioral_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── ADIM 3: 4 grup karşılaştırma — ham ortalama + Cohen's d ─────────────────

# Ham sütun adları → gösterim etiketi eşlemesi
METRIC_COL_MAP = {
    "Açık Kol Süresi % (EPM)":   "pct_open_arm",
    "Açık Kol Girişi % (EPM)":   "pct_open_arm_entries",
    "Anksiyete İndeksi (EPM)":   "anxiety_index_epm",
    "Donma Süresi % (OFT)":      "pct_time_freeze",
    "Merkez Zamanı % (OFT)":     "pct_time_center",
    "Toplam Mesafe (OFT)":       "total_distance_px",
    "Rearing % (OFT)":           "rear_pct_time",
    "Grooming % (OFT)":          "groom_pct_time",
}

SHORT_LABELS = [
    "Açık Kol\nSüresi\n(EPM)",
    "Açık Kol\nGirişi\n(EPM)",
    "Anksiyete\nİndeksi\n(EPM)",
    "Donma\nSüresi\n(OFT)",
    "Merkez\nZamanı\n(OFT)",
    "Lokomotor\nMesafe\n(OFT)",
    "Rearing\n(OFT)",
    "Grooming\n(OFT)",
]


def plot_effect_bars(df_raw: pd.DataFrame, table: pd.DataFrame):
    """
    Panel A — 4 grubun ham ortalaması, metrik başına min-max normalize edilmiş
              (0 = en düşük grup, 1 = en yüksek grup)
    Panel B — Cohen's d: 3 madde grubu vs Kontrol (Kontrol = 0 referans çizgisi)
    """
    selected_metrics = list(METRIC_COL_MAP.keys())
    df_sel = table[table["metrik"].isin(selected_metrics)].copy()

    # ── Panel A için grup ortalamalarını hesapla ve normalize et ─────────────
    metric_keys = list(METRIC_COL_MAP.keys())
    group_means = {}
    for cohort in COHORT_ORDER:
        sub  = df_raw[df_raw["cohort4"] == cohort]
        vals = [sub[col].mean() if col in sub.columns else np.nan
                for col in METRIC_COL_MAP.values()]
        group_means[cohort] = np.array(vals, dtype=float)

    raw_matrix = np.array([group_means[c] for c in COHORT_ORDER])  # (4, 8)
    col_min    = np.nanmin(raw_matrix, axis=0)
    col_max    = np.nanmax(raw_matrix, axis=0)
    col_rng    = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    norm_matrix = (raw_matrix - col_min) / col_rng  # 0–1

    MIN_VIS = 0.04   # küçük değerlerin yine de görünmesi için minimum bar yüksekliği

    # ── Figür ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.subplots_adjust(wspace=0.34, left=0.05, right=0.97, top=0.87, bottom=0.26)

    bar_w = 0.19
    x     = np.arange(len(metric_keys))

    # ── Panel A: 4 grup, her metrikte 4 renkli bar ───────────────────────────
    ax = axes[0]
    for i, cohort in enumerate(COHORT_ORDER):
        offset    = (i - 1.5) * bar_w
        norm_vals = norm_matrix[i]           # gerçek normalize değer
        raw_vals  = group_means[cohort]

        for j, (n_val, r_val) in enumerate(zip(norm_vals, raw_vals)):
            vis_h = max(float(n_val), MIN_VIS)   # görünür yükseklik
            bar = ax.bar(x[j] + offset, vis_h, bar_w * 0.92,
                         color=COHORT_COLORS[cohort],
                         alpha=0.85 if n_val >= MIN_VIS else 0.45,
                         label=cohort if j == 0 else "_nolegend_")

            # Ham değer etiketi — her bar'ın üstünde
            txt = (f"{r_val:.0f}" if r_val >= 100 else f"{r_val:.1f}")
            ax.text(x[j] + offset + bar_w * 0.92 / 2,
                    vis_h + 0.015,
                    txt,
                    ha="center", va="bottom",
                    fontsize=6.5, color="#333333",
                    rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=9)
    ax.set_ylabel("Normalize Edilmiş Ortalama\n(0 = en düşük grup, 1 = en yüksek grup)",
                  fontsize=10)
    ax.set_ylim(0, 1.55)
    ax.set_title("A — 4 Grup Karşılaştırması\n(Her metrikte 4 renk: Control · Aspartame · Grapefruit · ASP+Greyfurt)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel B: Cohen's d — 4 grup, Control = 0 bar ─────────────────────────
    ax = axes[1]

    # Control = 0 barı (açıkça göster)
    ctrl_bar_h = np.zeros(len(x))
    ax.bar(x - 1.5 * bar_w, ctrl_bar_h, bar_w * 0.92,
           color=COHORT_COLORS["Control"], alpha=0.85, label="Control  (d = 0, referans)")
    for j in range(len(x)):
        ax.text(x[j] - 1.5 * bar_w + bar_w * 0.92 / 2,
                0.08, "0", ha="center", va="bottom",
                fontsize=6.5, color=COHORT_COLORS["Control"], fontweight="bold")

    # 3 madde grubu
    for i, sub in enumerate(SUBSTANCES):
        sub_df = df_sel[df_sel["madde"] == sub].set_index("metrik")
        vals   = [float(sub_df.loc[m, "cohens_d"]) if m in sub_df.index else 0.0
                  for m in metric_keys]
        offset = (i - 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w * 0.92,
                        color=COHORT_COLORS[sub], alpha=0.85, label=sub)

    for val, lbl, ls in [(0.8, "büyük (0.8)", "--"), (1.4, "çok büyük (1.4)", ":")]:
        ax.axhline( val, color="#888888", linewidth=1, linestyle=ls, alpha=0.7)
        ax.axhline(-val, color="#888888", linewidth=1, linestyle=ls, alpha=0.7)
        ax.text(len(x) - 0.3, val + 0.06, lbl,
                fontsize=7, color="#888888", va="bottom")

    ax.axhline(0, color="black", linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=9)
    ax.set_ylabel("Cohen's d  (madde − kontrol)\nKontrol = 0 referans", fontsize=10)
    ax.set_title("B — Etki Büyüklüğü vs Kontrol (Cohen's d)\n"
                 "d = (madde_ort − kontrol_ort) / havuzlanmış_std",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Ortak legend — 4 grup
    handles = [mpatches.Patch(facecolor=COHORT_COLORS[c], alpha=0.85, label=c)
               for c in COHORT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Madde Etkisi — OFT + EPM Birleşik Analiz  (n = 3 / grup)\n"
        "A: Her metrikte 4 grup yan yana  |  B: Cohen's d (Kontrol referans alınarak hesaplanmış)",
        fontsize=11, fontweight="bold",
    )

    out = OUT_DIR / "behavioral_effect_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Veri yükleniyor...")
    df = load_data()
    print(f"  {len(df)} sıçan yüklendi | kohortlar: {df['cohort4'].value_counts().to_dict()}")

    # Adım 1 — Effect size tablosu
    print("\n[Adım 1] Cohen's d etki büyüklüğü tablosu...")
    table = build_effect_table(df)
    out_csv = OUT_DIR / "behavioral_effect_table.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] {out_csv}")

    # Özet yazdır
    print("\n=== ETKİ BÜYÜKLÜĞÜ ÖZETİ (EPM metrikleri) ===")
    epm_rows = table[table["metrik"].str.contains("EPM")]
    pivot = epm_rows.pivot(index="metrik", columns="madde",
                           values="cohens_d").reindex(columns=SUBSTANCES)
    print(pivot.round(2).to_string())

    print("\n=== % DEĞİŞİM ÖZETİ (EPM metrikleri) ===")
    pivot2 = epm_rows.pivot(index="metrik", columns="madde",
                            values="pct_degisim").reindex(columns=SUBSTANCES)
    print(pivot2.round(1).to_string())

    print("\n=== ETKİ BÜYÜKLÜĞÜ ETİKETLERİ ===")
    pivot3 = table.pivot(index="metrik", columns="madde",
                         values="etki_buyuklugu").reindex(columns=SUBSTANCES)
    print(pivot3.to_string())

    # Adım 2 — Radar grafiği
    print("\n[Adım 2] Radar grafiği...")
    plot_radar(df)

    # Adım 3 — Bar chart
    print("\n[Adım 3] 4 grup karşılaştırma + Cohen's d bar chart...")
    plot_effect_bars(df, table)

    print(f"\n{'='*55}")
    print(f"Tüm çıktılar: {OUT_DIR}")
    print(f"  behavioral_effect_table.csv")
    print(f"  behavioral_radar.png")
    print(f"  behavioral_effect_bars.png")


if __name__ == "__main__":
    main()
