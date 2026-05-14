# -*- coding: utf-8 -*-
"""
plus_maze_app.py  —  Plus Maze (EPM) analiz ve tahmin dashboard'u
Çalıştırmak için:
    cd webapp && streamlit run plus_maze_app.py
"""
from __future__ import annotations

import io
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Yol kurulumu ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PLUS_DIR  = ROOT / "analysis" / "plus_maze"
MODEL_DIR = ROOT / "models" / "epm_classifier"
sys.path.insert(0, str(PLUS_DIR))

from tmaze_metrics import compute_metrics, load_body_center  # noqa: E402

# ── Kol koordinatları ─────────────────────────────────────────────────────────
_arm_json = ROOT / "data" / "arm_coords.json"
if _arm_json.exists():
    with open(_arm_json, encoding="utf-8") as _f:
        DEFAULT_ARMS: dict[str, tuple] = {k: tuple(v) for k, v in json.load(_f).items()}
else:
    DEFAULT_ARMS = {
        "bottom_arm": (547, 603, 402, 713),
        "left_arm":   (260, 552, 347, 403),
        "right_arm":  (602, 894, 345, 402),
        "top_arm":    (546, 604,  33, 347),
    }

ZONE_HEX = {
    "bottom_arm": "#3B82F6",
    "left_arm":   "#10B981",
    "right_arm":  "#F59E0B",
    "top_arm":    "#A855F7",
}
ARM_LABELS = {
    "bottom_arm": "Alt Kol (kapalı)",
    "left_arm":   "Sol Kol (açık)",
    "right_arm":  "Sağ Kol (açık)",
    "top_arm":    "Üst Kol (kapalı)",
}
FPS = 30.0

FEATURE_COLS = [
    "pct_open_arm",
    "anxiety_index_epm",
    "pct_open_arm_entries",
    "total_entries",
    "successive_alternation_pct",
    "perseveration_rate_pct",
    "mean_speed_px_s",
    "arm_preference_index",
    "pct_time_junction",
]

# Türkçe ad, açıklama ve hesaplama formülleri — Karar Katkıları altındaki
# tıklanabilir kartlarda kullanılır.
FEATURE_INFO: dict[str, dict[str, str]] = {
    "pct_open_arm": {
        "name":       "Açık Kol Süresi (%)",
        "name_short": "Açık Kol Süresi",
        "desc": "Farenin açık kollarda (sol + sağ) geçirdiği zamanın "
                "seansa oranıdır. EPM testinde anksiyete seviyesinin en güçlü "
                "göstergelerindendir; düşük değer yüksek anksiyeteye işaret eder.",
        "formula": "pct_time_left + pct_time_right",
    },
    "anxiety_index_epm": {
        "name":       "Anksiyete İndeksi (EPM)",
        "name_short": "Anksiyete İndeksi",
        "desc": "Açık kol süresi ve açık kola giriş yüzdesinin ortalaması olan "
                "bütünleşik bir anksiyete göstergesidir. Yüksek değer düşük "
                "anksiyeteyi (daha çok keşif) ifade eder.",
        "formula": "(pct_open_arm + pct_open_arm_entries) / 2",
    },
    "pct_open_arm_entries": {
        "name":       "Açık Kola Giriş Yüzdesi (%)",
        "name_short": "Açık Kola Giriş %",
        "desc": "Toplam giriş sayısı içinde açık kollara yapılanların oranıdır. "
                "Düşük olması kaçınma/anksiyete davranışını işaret eder.",
        "formula": "(left_entries + right_entries) / total_entries × 100",
    },
    "total_entries": {
        "name":       "Toplam Kol Girişi",
        "name_short": "Toplam Giriş",
        "desc": "Farenin hareketi boyunca herhangi bir kola yaptığı toplam "
                "giriş sayısıdır. Genel keşif ve lokomotor aktivitenin "
                "göstergesidir. Bir giriş sayılması için ilgili kolda en az "
                "~0.1 sn kalınmış olmalıdır.",
        "formula": "left + right + top + bottom kol girişlerinin toplamı",
    },
    "successive_alternation_pct": {
        "name":       "Ardışık Alternasyon (%)",
        "name_short": "Ardışık Alternasyon",
        "desc": "Ardışık iki girişin farklı kollara olma yüzdesidir. Çalışma "
                "belleği ve yenilik araştırma davranışını yansıtır; yüksek değer "
                "esnek keşfin işaretidir.",
        "formula": "(farklı_ardışık_giriş / (total_entries − 1)) × 100",
    },
    "perseveration_rate_pct": {
        "name":       "Perseverasyon Oranı (%)",
        "name_short": "Perseverasyon",
        "desc": "Ardışık olarak aynı kola tekrar giriş yapma oranıdır. Yüksek "
                "değerler katı / tekrarlayıcı davranışı ve bilişsel esneklik "
                "azalmasını işaret edebilir.",
        "formula": "(aynı_kola_ardışık_giriş / (total_entries − 1)) × 100",
    },
    "mean_speed_px_s": {
        "name":       "Ortalama Hız (px/sn)",
        "name_short": "Ortalama Hız",
        "desc": "Farenin saniyedeki ortalama yer değiştirme miktarıdır "
                "(piksel cinsinden). Genel motor aktivitenin doğrudan ölçüsüdür.",
        "formula": "ortalama( √(dx² + dy²) × fps ),  geçerli kareler üzerinden",
    },
    "arm_preference_index": {
        "name":       "Kol Tercih İndeksi",
        "name_short": "Kol Tercih İndeksi",
        "desc": "Kollar arasındaki giriş dağılımının dengesizliğidir. 0 ≈ eşit "
                "dağılım (dengeli keşif), 1 ≈ tek bir kola yönelim "
                "(tercih / yanlılık).",
        "formula": "(max_kol_girişi − min_kol_girişi) / total_entries",
    },
    "pct_time_junction": {
        "name":       "Kavşakta Geçen Süre (%)",
        "name_short": "Kavşakta Geçen Süre",
        "desc": "Farenin EPM'nin orta (junction) bölgesinde geçirdiği zaman "
                "yüzdesidir. Yüksek değerler duraksama / karar verme davranışını "
                "işaret edebilir.",
        "formula": "junction_kare_sayısı / geçerli_kare_sayısı × 100",
    },
}

# ── Sayfa konfigürasyonu ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EPM Analiz & Tahmin",
    page_icon="🐀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* App background */
.stApp { background: #0b0f1a; }
.main .block-container { padding-top: 1.5rem; max-width: 1280px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1320 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.06) !important;
    border: 1.5px dashed rgba(99,102,241,0.35) !important;
    border-radius: 14px !important;
    transition: all 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.75) !important;
    background: rgba(99,102,241,0.1) !important;
}

/* File uploader — Türkçe etiketler, dosya boyutu satırını gizle */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
    content: "Dosyayı buraya sürükleyip bırakın";
    color: rgba(255,255,255,0.6);
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.005em;
}
section[data-testid="stFileUploaderDropzone"] button {
    color: transparent !important;
    position: relative;
    min-width: 110px !important;
}
section[data-testid="stFileUploaderDropzone"] button::after {
    content: "Dosya Seç";
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    color: rgba(255,255,255,0.88);
    font-weight: 500;
    font-size: 0.875rem;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #1e3a5f !important;
    border: 1px solid rgba(99,149,210,0.35) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2.5rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35) !important;
    letter-spacing: 0.01em !important;
    color: #cbd5e1 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #254d7a !important;
    border-color: rgba(99,149,210,0.6) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
    color: #e2e8f0 !important;
}

/* Metric widgets */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: rgba(255,255,255,0.45) !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(255,255,255,0.4) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border: none !important;
    padding: 0.75rem 1.4rem !important;
    border-radius: 0 !important;
    letter-spacing: 0.01em !important;
}
.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #6366f1 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 100px !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 100px !important;
    height: 6px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }
.dvn-scroller { border-radius: 12px !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Alert/Info boxes */
.stAlert { border-radius: 10px !important; }

/* Expander */
.streamlit-expanderHeader {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.85rem !important;
    background: rgba(255,255,255,0.02) !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }

/* Hide chrome */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Yardımcı HTML bileşenleri ─────────────────────────────────────────────────

def html_badge(text: str, color: str = "#6366f1") -> str:
    return (
        f'<span style="background:rgba({_hex_to_rgb(color)},0.15);'
        f'border:1px solid rgba({_hex_to_rgb(color)},0.35);'
        f'border-radius:100px;padding:3px 12px;font-size:0.72rem;'
        f'font-weight:500;color:{color};letter-spacing:0.04em;">{text}</span>'
    )


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def html_stat_row(label: str, value: str) -> str:
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
        f'<span style="font-size:0.8rem;color:rgba(255,255,255,0.45);font-weight:500;">{label}</span>'
        f'<span style="font-size:0.85rem;color:#f1f5f9;font-weight:600;">{value}</span>'
        f'</div>'
    )


def html_pred_card(label: str, proba: float, is_treated: bool) -> str:
    if is_treated:
        border = "#EF4444"
        bg     = "linear-gradient(135deg,rgba(239,68,68,0.12),rgba(220,38,38,0.06))"
    else:
        border = "#10B981"
        bg     = "linear-gradient(135deg,rgba(16,185,129,0.12),rgba(5,150,105,0.06))"
    display_label = f"{label} Grubu"
    return (
        f'<div style="background:{bg};border:1.5px solid rgba({_hex_to_rgb(border)},0.45);'
        f'border-radius:16px;padding:1.75rem 1.25rem;text-align:center;margin-bottom:1rem;">'
        f'<div style="font-size:0.74rem;font-weight:700;color:rgba(255,255,255,0.55);'
        f'letter-spacing:0.18em;margin:0 auto 10px;text-align:center;">TAHMİN</div>'
        f'<div style="font-size:1.9rem;font-weight:800;color:{border};letter-spacing:-0.5px;">{display_label}</div>'
        f'<div style="font-size:0.85rem;color:rgba(255,255,255,0.45);margin-top:8px;">'
        f'Güven: <strong style="color:{border}">{proba:.1%}</strong></div>'
        f'</div>'
    )


# Türkçe locale farkındalıklı upper(): "i" → "İ", "ı" → "I".
_TR_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})


def tr_upper(s: str) -> str:
    return s.translate(_TR_UPPER_MAP).upper()


def html_section(title: str) -> str:
    label = tr_upper(title)
    return (
        f'<div style="font-size:1.3rem;font-weight:800;color:#e2e8f0;'
        f'letter-spacing:0.16em;'
        f'margin:1.6rem 0 0.9rem 0;display:flex;align-items:center;gap:16px;'
        f'justify-content:center;">'
        f'<div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>'
        f'<span>{label}</span>'
        f'<div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>'
        f'</div>'
    )


# ── Matplotlib grafikleri ─────────────────────────────────────────────────────

_DARK_BG  = "#0b0f1a"
_PANEL_BG = "#0f1320"
_GRID_CLR = "#1e2438"
_MUTED    = "#475569"


def _style_ax_dark(ax, fig=None):
    bg = _PANEL_BG
    if fig:
        fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(bg)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_CLR)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color("#94a3b8")


def fig_overview(body_x, body_y, arms: dict, pred_info: dict) -> plt.Figure:
    pred_color = "#EF4444" if pred_info["pred"] == 1 else "#10B981"

    all_x = [v for (x0, x1, y0, y1) in arms.values() for v in (x0, x1)]
    all_y = [v for (x0, x1, y0, y1) in arms.values() for v in (y0, y1)]
    margin = 40

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=_DARK_BG)
    _style_ax_dark(ax, fig)
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(max(all_y) + margin, min(all_y) - margin)
    ax.grid(alpha=0.06, color="white", linewidth=0.5, zorder=0)

    for arm_key, (x0, x1, y0, y1) in arms.items():
        c = ZONE_HEX[arm_key]
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1-x0, y1-y0,
            linewidth=0, facecolor=c, alpha=0.10, zorder=1,
        ))
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1-x0, y1-y0,
            linewidth=1.5, edgecolor=c, facecolor="none", alpha=0.45, zorder=2,
        ))
        # Etiketler: yörüngeyle çakışmamak için dış kenara konumlandırılır
        cx, cy = (x0+x1)/2, (y0+y1)/2
        lbl = ARM_LABELS[arm_key]
        if arm_key == "top_arm":
            tx, ty, ha, va = cx, y0 + 10, "center", "top"
        elif arm_key == "bottom_arm":
            tx, ty, ha, va = cx, y1 - 10, "center", "bottom"
        elif arm_key == "left_arm":
            tx, ty, ha, va = x0 + 10, cy, "left", "center"
        else:  # right_arm
            tx, ty, ha, va = x1 - 10, cy, "right", "center"
        ax.text(
            tx, ty, lbl,
            ha=ha, va=va, fontsize=7.5, fontweight="600",
            color=c, alpha=0.85, zorder=3,
        )

    valid = ~(np.isnan(body_x) | np.isnan(body_y))
    if valid.sum() > 5:
        xi, yi = body_x[valid], body_y[valid]
        t_norm = np.linspace(0, 1, len(xi))
        points   = np.array([xi, yi]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        base_rgb = mcolors.to_rgb(pred_color)
        dark_rgb = tuple(c * 0.20 for c in base_rgb)
        cmap = mcolors.LinearSegmentedColormap.from_list("traj", [dark_rgb, pred_color])
        lc = LineCollection(segments, cmap=cmap, linewidth=1.3, alpha=0.80, zorder=4)
        lc.set_array(t_norm[:-1])
        lc.set_clim(0, 1)
        ax.add_collection(lc)
        ax.scatter(xi[0],  yi[0],  color="#10B981", s=80, zorder=8,
                   edgecolors=_DARK_BG, linewidths=1.5, label="Başlangıç")
        ax.scatter(xi[-1], yi[-1], color="#EF4444", s=70, zorder=8,
                   marker="D", edgecolors=_DARK_BG, linewidths=1.5, label="Bitiş")
        leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.25,
                        facecolor=_PANEL_BG, edgecolor=_GRID_CLR, labelcolor="#94a3b8")

    ax.set_xlabel("x (piksel)", fontsize=9)
    ax.set_ylabel("y (piksel)", fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig


def fig_probability_bars(pred_info: dict) -> plt.Figure:
    prob_c = pred_info["proba_control"]
    prob_t = pred_info["proba_treated"]
    is_treated = pred_info["pred"] == 1

    fig, ax = plt.subplots(figsize=(5.5, 1.6), facecolor=_DARK_BG)
    _style_ax_dark(ax, fig)

    labels = ["P(Kontrol)", "P(Tedavi)"]
    values = [prob_c, prob_t]
    colors = ["#3B82F6", "#EF4444"]
    alphas = [0.85 if not is_treated else 0.30, 0.85 if is_treated else 0.30]

    bars = ax.barh(labels, values, color=colors, alpha=1.0, height=0.45,
                   edgecolor="none")
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    for bar, val, clr in zip(bars, values, colors):
        ax.text(val + 0.025, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10, fontweight="700", color=clr)

    ax.set_xlim(0, 1.18)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(left=False)
    for label in ax.get_yticklabels():
        label.set_color("#94a3b8")
        label.set_fontsize(9)
    fig.tight_layout(pad=0.8)
    return fig


def fig_feature_contributions(pred_info: dict) -> plt.Figure:
    tops   = pred_info["top_contributors"]
    names  = [
        FEATURE_INFO.get(c["feature"], {}).get(
            "name_short", c["feature"].replace("_", " ").title()
        )
        for c in tops
    ]
    pushes = [c["push"] for c in tops]
    colors = ["#EF4444" if p > 0 else "#3B82F6" for p in pushes]

    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=_DARK_BG)
    _style_ax_dark(ax, fig)

    bars = ax.barh(names[::-1], pushes[::-1], color=colors[::-1],
                   alpha=0.85, height=0.55, edgecolor="none")
    ax.axvline(0, color=_MUTED, linewidth=0.8, alpha=0.6, zorder=3)

    # Sayı etiketleri için biraz daha geniş bir x ekseni — etiketler
    # bar uçlarıyla / y-ekseni adlarıyla çakışmasın.
    max_abs = max((abs(p) for p in pushes), default=1.0) or 1.0
    pad = max_abs * 0.22
    ax.set_xlim(-max_abs - pad, max_abs + pad)
    offset = max_abs * 0.025

    for bar, val in zip(bars, pushes[::-1]):
        ha = "left" if val >= 0 else "right"
        dx = offset if val >= 0 else -offset
        ax.text(val + dx, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9,
                color="#cbd5e1", fontweight="600")

    ax.set_xlabel("Karar Katkısı", fontsize=10, color="#94a3b8")
    ax.grid(axis="x", alpha=0.08, color="white", zorder=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color("#e2e8f0")
        lbl.set_fontsize(10)
        lbl.set_fontweight("500")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor="#EF4444", alpha=0.85, label="→ Tedavi"),
                 Patch(facecolor="#3B82F6", alpha=0.85, label="→ Kontrol")],
        loc="lower right", fontsize=9, framealpha=0.2,
        facecolor=_PANEL_BG, edgecolor=_GRID_CLR, labelcolor="#94a3b8",
    )
    # Sol kenar boşluğunu artırarak uzun Türkçe isimlere yer aç.
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.18)
    return fig


def fig_arm_distribution(row: dict) -> plt.Figure:
    arms   = ["Sol\n(açık)", "Sağ\n(açık)", "Alt\n(kapalı)", "Üst\n(kapalı)", "Kavşak"]
    times  = [
        row.get("pct_time_left",     0) or 0,
        row.get("pct_time_right",    0) or 0,
        row.get("pct_time_bottom",   0) or 0,
        row.get("pct_time_top",      0) or 0,
        row.get("pct_time_junction", 0) or 0,
    ]
    colors = ["#10B981", "#F59E0B", "#3B82F6", "#A855F7", "#64748B"]

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=_DARK_BG)
    _style_ax_dark(ax, fig)

    x     = np.arange(len(arms))
    bars  = ax.bar(x, times, color=colors, alpha=0.80, width=0.55, edgecolor="none")

    for bar, val in zip(bars, times):
        bx = bar.get_x() + bar.get_width() / 2
        ax.text(bx, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="700", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=9)
    ax.set_ylabel("Zaman (%)", fontsize=9)
    ax.set_ylim(0, max(times) * 1.3 + 5)
    ax.grid(axis="y", alpha=0.07, color="white", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for lbl in ax.get_xticklabels():
        lbl.set_color("#94a3b8")

    fig.tight_layout(pad=1.0)
    return fig


# ── EPM türetilmiş metrikler ──────────────────────────────────────────────────

def add_derived(row: dict) -> dict:
    total = row["total_entries"] or float("nan")
    row["pct_open_arm"]   = row["pct_time_left"] + row["pct_time_right"]
    row["pct_closed_arm"] = row["pct_time_top"]  + row["pct_time_bottom"]
    oe = (row["left_entries"] + row["right_entries"]) / total * 100 if total else float("nan")
    row["pct_open_arm_entries"]   = oe
    row["pct_closed_arm_entries"] = (row["top_entries"] + row["bottom_entries"]) / total * 100 if total else float("nan")
    row["anxiety_index_epm"] = (row["pct_open_arm"] + (oe if not np.isnan(oe) else 0)) / 2
    return row


# ── Model yükleme ─────────────────────────────────────────────────────────────

def _train_from_csv():
    data_csv = ROOT / "data" / "plus_maze_metrics_all.csv"
    if not data_csv.exists():
        return None, None
    df = pd.read_csv(data_csv)
    df["label"] = (df["cohort"] != "Control").astype(int)
    if any(c not in df.columns for c in FEATURE_COLS):
        return None, None
    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values
    imp = SimpleImputer(strategy="median")
    X_i = imp.fit_transform(X)
    sc  = StandardScaler()
    X_s = sc.fit_transform(X_i)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_s, y)
    return {"imputer": imp, "scaler": sc, "features": FEATURE_COLS}, clf


@st.cache_resource
def load_model():
    sp, mp = MODEL_DIR / "scaler_epm.pkl", MODEL_DIR / "lr_epm.pkl"
    if sp.exists() and mp.exists():
        try:
            with open(sp, "rb") as f: bundle = pickle.load(f)
            with open(mp, "rb") as f: clf    = pickle.load(f)
            return bundle, clf
        except Exception:
            pass
    return _train_from_csv()


# ── Tahmin ────────────────────────────────────────────────────────────────────

def predict(row: dict, bundle: dict, clf) -> dict:
    feats   = bundle["features"]
    x       = np.array([[row.get(c, np.nan) for c in feats]], dtype=float)
    x_i     = bundle["imputer"].transform(x)
    x_s     = bundle["scaler"].transform(x_i)
    # scikit-learn 1.7+ uyumluluğu — eski pkl'larda olmayan multi_class
    # attribute'ünü predict_proba çağırırken set et (binary için "auto" yeterli).
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    pred    = int(clf.predict(x_s)[0])
    proba   = clf.predict_proba(x_s)[0]
    coef    = clf.coef_.ravel()
    contrib = coef * x_s[0]
    order   = np.argsort(-np.abs(contrib))
    top = [
        {
            "feature": feats[i],
            "value":   float(x[0, i]) if not np.isnan(x[0, i]) else None,
            "z":       round(float(x_s[0, i]), 2),
            "coef":    round(float(coef[i]), 3),
            "push":    round(float(contrib[i]), 3),
            "toward":  "Tedavi" if contrib[i] > 0 else "Kontrol",
        }
        for i in order[:5]
    ]
    return {
        "pred":             pred,
        "proba_control":    float(proba[0]),
        "proba_treated":    float(proba[1]),
        "top_contributors": top,
        "feature_cols":     feats,
    }


# ── Görselleştirme (subprocess) ───────────────────────────────────────────────

def run_visuals(csv_path: Path, arms: dict, out_dir: Path) -> dict:
    arm_args = []
    for key in ("bottom_arm", "left_arm", "right_arm", "top_arm"):
        arm_args += [f"--{key.replace('_','-')}"] + [str(v) for v in arms[key]]
    common = [
        "--csv",         str(csv_path),
        "--out-dir",     str(out_dir),
        "--likelihood",  "0.6",
        "--jump-thresh", "60",
        "--smooth",      "5",
    ] + arm_args
    for script in ("orbit_plot.py", "activity_heatmap.py"):
        try:
            res = subprocess.run(
                [sys.executable, str(PLUS_DIR / script)] + common,
                capture_output=True, text=True, timeout=120,
            )
            if res.returncode != 0:
                st.warning(f"{script}: {res.stderr[-300:] if res.stderr else 'bilinmiyor'}")
        except subprocess.TimeoutExpired:
            st.warning(f"{script} zaman aşımına uğradı (>120s)")
    stem    = csv_path.stem
    key_map = {
        "_plus_maze_orbit.png":     "orbit",
        "_plus_maze_bodyparts.png": "bodyparts",
        "_heatmap_kde.png":         "heatmap_kde",
        "_heatmap_histogram.png":   "heatmap_histogram",
    }
    images = {}
    for f in out_dir.iterdir():
        if f.name.startswith(stem) and f.suffix == ".png":
            for suffix, key in key_map.items():
                if f.name.endswith(suffix):
                    images[key] = f.read_bytes()
    return images


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;'
        'letter-spacing:0.14em;margin:0.6rem 0 0.7rem 0;text-align:center;">'
        'MODEL BİLGİSİ</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        html_stat_row("Algoritma",    "Lojistik Regresyon") +
        html_stat_row("Eğitim seti",  "37 sıçan (LOOCV)") +
        html_stat_row("AUC",          "0.733") +
        html_stat_row("F1 (Tedavi)",  "0.821") +
        html_stat_row("Duyarlılık",   "76.7%"),
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:1rem 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem;font-weight:600;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">Kullanım</div>', unsafe_allow_html=True)
    for i, step in enumerate(["DLC filtered CSV yükle", "Çalıştır butonuna bas", "Sonuçları incele & indir"], 1):
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:center;padding:6px 0;">'
            f'<div style="width:22px;height:22px;border-radius:50%;background:rgba(255,255,255,0.05);'
            f'border:1px solid rgba(255,255,255,0.14);display:flex;align-items:center;justify-content:center;'
            f'font-size:0.7rem;font-weight:600;color:#94a3b8;flex-shrink:0;">{i}</div>'
            f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.55);">{step}</span></div>',
            unsafe_allow_html=True,
        )



# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a5f 0%,#254d7a 55%,#0c1e36 100%);
     border-radius:16px;padding:1.75rem 2rem;margin-bottom:1.5rem;
     border:1px solid rgba(99,149,210,0.28);
     box-shadow:0 8px 28px rgba(12,30,54,0.55);">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:2.8rem;line-height:1;">🐀</div>
    <div>
      <h1 style="margin:0;font-size:1.6rem;font-weight:800;color:white;letter-spacing:-0.5px;">
        Elevated Plus Maze  —  Analiz Paneli
      </h1>
      <p style="margin:4px 0 0 0;font-size:0.82rem;color:rgba(255,255,255,0.45);">
        DeepLabCut pose verisi · Kaygı metrikleri · ML tabanlı grup tahmini
      </p>
    </div>
  </div>
  <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap;">
    <span style="background:rgba(99,102,241,0.18);border:1px solid rgba(99,102,241,0.35);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#818cf8;letter-spacing:0.04em;">Lojistik Regresyon</span>
    <span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.30);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#34d399;letter-spacing:0.04em;">LOOCV · n=37</span>
    <span style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.30);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#fbbf24;letter-spacing:0.04em;">AUC 0.733</span>
    <span style="background:rgba(168,85,247,0.15);border:1px solid rgba(168,85,247,0.30);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#c084fc;letter-spacing:0.04em;">F1 0.821</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DOSYA YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────

upload_col, btn_col = st.columns([3, 1], gap="medium")
with upload_col:
    uploaded = st.file_uploader(
        "DLC filtered pose CSV (tek sıçan)",
        type=["csv"],
        label_visibility="collapsed",
        help="DeepLabCut tarafından üretilmiş filtrelenmiş CSV dosyası",
    )
with btn_col:
    st.markdown('<div style="height:0.35rem;"></div>', unsafe_allow_html=True)
    run_clicked = st.button("▶  Çalıştır", type="primary", use_container_width=True)

if uploaded is None:
    st.markdown("""
    <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
         border-radius:12px;padding:1.25rem 1.5rem;margin-top:0.5rem;
         display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.5rem;">📂</span>
      <span style="font-size:0.875rem;color:rgba(255,255,255,0.45);">
        Analiz başlatmak için bir CSV dosyası yükleyin, ardından
        <strong style="color:#818cf8;">Çalıştır</strong> butonuna basın.
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_key = f"{uploaded.name}:{uploaded.size}"
has_cache = st.session_state.get("plus_maze_cache_key") == file_key

if not run_clicked and not has_cache:
    st.markdown(
        f'<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);'
        f'border-radius:10px;padding:0.9rem 1.2rem;margin-top:0.5rem;'
        f'display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:1.1rem;">📄</span>'
        f'<span style="font-size:0.875rem;color:#34d399;font-weight:500;">{uploaded.name}</span>'
        f'<span style="color:rgba(255,255,255,0.3);font-size:0.8rem;margin-left:auto;">'
        f'Hazır · Çalıştır butonuna bas</span></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL KONTROLÜ
# ─────────────────────────────────────────────────────────────────────────────

bundle, clf = load_model()
if bundle is None:
    st.error(
        "Model yüklenemedi — `data/plus_maze_metrics_all.csv` dosyasının "
        "varlığını kontrol edin."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if has_cache and not run_clicked:
    cached = st.session_state["plus_maze_results"]
    row            = cached["row"]
    pred_info      = cached["pred_info"]
    images         = cached["images"]
    overview_bytes = cached["overview_bytes"]
else:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / uploaded.name
        csv_path.write_bytes(uploaded.getvalue())

        prog = st.progress(0.0)
        status_ph = st.empty()

        def _step(pct: float, msg: str):
            prog.progress(pct)
            status_ph.markdown(
                f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.4);'
                f'margin:4px 0 8px 2px;">{msg}</div>',
                unsafe_allow_html=True,
            )

        try:
            _step(0.12, "⚙  EPM metrikleri hesaplanıyor…")
            row = add_derived(compute_metrics(str(csv_path), DEFAULT_ARMS, fps=FPS))

            _step(0.35, "📍 Trajectory okunuyor…")
            body_x, body_y = load_body_center(
                str(csv_path), likelihood_thresh=0.6, jump_thresh=60.0, smooth=5
            )

            _step(0.55, "🤖 Model tahmini yapılıyor…")
            pred_info = predict(row, bundle, clf)

            _step(0.72, "🎨 Görselleştirmeler üretiliyor…")
            images = run_visuals(csv_path, DEFAULT_ARMS, tmp_path)

            fig_ov = fig_overview(body_x, body_y, DEFAULT_ARMS, pred_info)
            buf = io.BytesIO()
            fig_ov.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                           facecolor=_DARK_BG)
            plt.close(fig_ov)
            overview_bytes = buf.getvalue()

            _step(1.0, "✓  Tamamlandı")
            prog.empty()
            status_ph.empty()

        except Exception as e:
            st.error(f"Pipeline hatası: {type(e).__name__}: {e}")
            st.exception(e)
            st.stop()

    st.session_state["plus_maze_cache_key"] = file_key
    st.session_state["plus_maze_results"]   = {
        "row":            row,
        "pred_info":      pred_info,
        "images":         images,
        "overview_bytes": overview_bytes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SONUÇLAR
# ─────────────────────────────────────────────────────────────────────────────

pred_label = "Tedavi"  if pred_info["pred"] == 1 else "Kontrol"
top_proba  = pred_info["proba_treated"] if pred_info["pred"] == 1 else pred_info["proba_control"]
subject    = Path(uploaded.name).stem
is_treated = pred_info["pred"] == 1

st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

tab_pred, tab_metrics, tab_viz, tab_dl = st.tabs(
    ["🎯  Tahmin", "📊  Metrikler", "🖼  Görselleştirme", "⬇  İndir"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TAHMİN
# ─────────────────────────────────────────────────────────────────────────────

with tab_pred:
    col_img, col_right = st.columns([1.35, 1], gap="large")

    with col_img:
        st.markdown(html_section("Hareket Haritası"), unsafe_allow_html=True)
        st.image(overview_bytes, use_container_width=True)

    with col_right:
        st.markdown(html_section("Tahmin Sonucu"), unsafe_allow_html=True)
        st.markdown(html_pred_card(pred_label, top_proba, is_treated), unsafe_allow_html=True)

        # Probability bars
        buf_p = io.BytesIO()
        fp = fig_probability_bars(pred_info)
        fp.savefig(buf_p, format="png", dpi=130, bbox_inches="tight",
                   facecolor=_DARK_BG)
        plt.close(fp)
        st.image(buf_p.getvalue(), use_container_width=True)

        # P(Kontrol) + P(Tedavi) = 1 doğrulaması
        _pc = pred_info["proba_control"]
        _pt = pred_info["proba_treated"]
        st.markdown(
            f"<div style='text-align:center;color:#94a3b8;font-size:0.85rem;"
            f"margin:0.15rem 0 0.6rem 0;font-family:\"JetBrains Mono\",monospace;'>"
            f"P(Kontrol) + P(Tedavi) = "
            f"<span style='color:#3B82F6;font-weight:600;'>{_pc:.3f}</span> + "
            f"<span style='color:#EF4444;font-weight:600;'>{_pt:.3f}</span> = "
            f"<span style='color:#f1f5f9;font-weight:700;'>{(_pc + _pt):.3f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(html_section("Temel Metrikler"), unsafe_allow_html=True)
        basic_metrics = [
            ("Açık Kol Süresi", f"%{row['pct_open_arm']:.1f}",
             "Farenin açık kollarda (sol + sağ) geçirdiği zamanın yüzdesi. "
             "Düşük değer yüksek anksiyeteye işaret eder."),
            ("Anksiyete İndeksi", f"{row['anxiety_index_epm']:.2f}",
             "Açık kol süresi ile açık kola giriş yüzdesinin ortalaması. "
             "Yüksek değer düşük anksiyeteyi (daha çok keşif) ifade eder."),
            ("Toplam Giriş", str(int(row["total_entries"])),
             "Farenin hareketi boyunca herhangi bir kola yaptığı toplam "
             "giriş sayısı. Genel keşif ve lokomotor aktivitenin "
             "göstergesidir."),
            ("Ardışık Alternasyon",
             f"%{row.get('successive_alternation_pct', 0) or 0:.1f}",
             "Ardışık iki girişin farklı kollara olma yüzdesi. "
             "Çalışma belleği ve esnek keşif davranışını yansıtır."),
        ]
        mr1 = st.columns(2)
        mr2 = st.columns(2)
        for slot, (label, val, desc) in zip([*mr1, *mr2], basic_metrics):
            with slot:
                with st.expander(f"{label}   ·   {val}", expanded=False):
                    st.markdown(
                        f"<div style='font-size:0.95rem;color:#cbd5e1;"
                        f"line-height:1.55;'>{desc}</div>",
                        unsafe_allow_html=True,
                    )

    # ── Karar Katkıları — tam genişlik, alt satırda ──────────────────────────
    st.markdown(html_section("Karar Katkıları"), unsafe_allow_html=True)
    buf_f = io.BytesIO()
    ff = fig_feature_contributions(pred_info)
    ff.savefig(buf_f, format="png", dpi=130, bbox_inches="tight",
               facecolor=_DARK_BG)
    plt.close(ff)
    st.image(buf_f.getvalue(), use_container_width=True)

    for c in pred_info["top_contributors"]:
        feat = c["feature"]
        info = FEATURE_INFO.get(feat)
        push  = c["push"]
        toward = "Tedavi Grubu" if push > 0 else "Kontrol Grubu"
        arrow_color = "#EF4444" if push > 0 else "#3B82F6"
        display_name = info["name"] if info else feat.replace("_", " ").title()
        value = c.get("value")
        value_str = f"{value:.3f}" if isinstance(value, (int, float)) else "—"

        header = f"{display_name}   ·   katkı {push:+.3f}   →   {toward}"
        with st.expander(header, expanded=False):
            if info:
                st.markdown(
                    f"<div style='font-size:0.72rem;font-weight:700;"
                    f"letter-spacing:0.14em;color:#94a3b8;"
                    f"margin:0.15rem 0 0.4rem 0;'>AÇIKLAMA</div>"
                    f"<div style='font-size:0.98rem;color:#e2e8f0;"
                    f"line-height:1.6;margin-bottom:0.9rem;'>{info['desc']}</div>"
                    f"<div style='font-size:0.72rem;font-weight:700;"
                    f"letter-spacing:0.14em;color:#94a3b8;"
                    f"margin:0.2rem 0 0.4rem 0;'>NASIL HESAPLANIR</div>",
                    unsafe_allow_html=True,
                )
                st.code(info["formula"], language="text")
            else:
                st.markdown(
                    "<div style='font-size:0.98rem;color:#cbd5e1;'>"
                    "Bu özellik için açıklama tanımlanmamış.</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-size:0.92rem;color:rgba(255,255,255,0.78);"
                f"line-height:1.6;border-top:1px solid rgba(255,255,255,0.07);"
                f"padding-top:0.7rem;margin-top:0.7rem;'>"
                f"Bu denek için ölçülen değer "
                f"<strong style='color:#f1f5f9;'>{value_str}</strong> "
                f"(z-skoru {c['z']:+.2f}). Modelin lojistik regresyon katsayısı "
                f"(<strong style='color:#f1f5f9;'>{c['coef']:+.3f}</strong>) "
                f"ile çarpılınca <strong style='color:{arrow_color};'>{push:+.3f}</strong> "
                f"büyüklüğünde, <strong style='color:{arrow_color};'>{toward}</strong> "
                f"yönünde bir karar katkısı üretir."
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — METRİKLER
# ─────────────────────────────────────────────────────────────────────────────

with tab_metrics:
    st.markdown(html_section("Kol Bazlı Zaman Dağılımı"), unsafe_allow_html=True)
    buf_a = io.BytesIO()
    fa = fig_arm_distribution(row)
    fa.savefig(buf_a, format="png", dpi=140, bbox_inches="tight",
               facecolor=_DARK_BG)
    plt.close(fa)
    st.image(buf_a.getvalue(), use_container_width=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(html_section("Model Özellikleri"), unsafe_allow_html=True)
        
        for k in FEATURE_COLS:
            info = FEATURE_INFO.get(k, {})
            name = info.get("name", k.replace("_", " ").title())
            val  = row.get(k)
            if isinstance(val, float):
                val_str = f"{val:.3f}"
            elif val is None:
                val_str = "—"
            else:
                val_str = str(val)

            with st.expander(f"{name}   ·   {val_str}", expanded=False):
                desc = info.get("desc")
                formula = info.get("formula")
                if desc:
                    st.markdown(
                        f"<div style='font-size:0.72rem;font-weight:700;"
                        f"letter-spacing:0.14em;color:#94a3b8;"
                        f"margin:0.1rem 0 0.4rem 0;'>AÇIKLAMA</div>"
                        f"<div style='font-size:0.95rem;color:#e2e8f0;"
                        f"line-height:1.55;margin-bottom:0.7rem;'>{desc}</div>",
                        unsafe_allow_html=True,
                    )
                if formula:
                    st.markdown(
                        "<div style='font-size:0.72rem;font-weight:700;"
                        "letter-spacing:0.14em;color:#94a3b8;"
                        "margin:0.1rem 0 0.35rem 0;'>NASIL HESAPLANIR</div>",
                        unsafe_allow_html=True,
                    )
                    st.code(formula, language="text")
                if not desc and not formula:
                    st.markdown(
                        "<div style='font-size:0.95rem;color:#cbd5e1;'>"
                        "Bu özellik için açıklama tanımlanmamış.</div>",
                        unsafe_allow_html=True,
                    )

    with mc2:
        st.markdown(html_section("Kol Detayları"), unsafe_allow_html=True)
        arm_rows = [
            ("Sol Kol (açık)",   "#10B981", row.get('pct_time_left',     0) or 0, int(row.get("left_entries",   0) or 0)),
            ("Sağ Kol (açık)",   "#F59E0B", row.get('pct_time_right',    0) or 0, int(row.get("right_entries",  0) or 0)),
            ("Alt Kol (kapalı)", "#3B82F6", row.get('pct_time_bottom',   0) or 0, int(row.get("bottom_entries", 0) or 0)),
            ("Üst Kol (kapalı)", "#A855F7", row.get('pct_time_top',      0) or 0, int(row.get("top_entries",    0) or 0)),
            ("Kavşak",           "#64748B", row.get('pct_time_junction', 0) or 0, None),
        ]
        max_pct = max((p for _, _, p, _ in arm_rows), default=1.0) or 1.0

        cards_html = ""
        for name, color, pct, ent in arm_rows:
            ent_str = str(ent) if ent is not None else "—"
            bar_w = (pct / max_pct) * 100.0
            cards_html += (
                f"<div style='background:rgba(255,255,255,0.03);"
                f"border:1px solid rgba(255,255,255,0.07);border-radius:12px;"
                f"padding:0.75rem 0.95rem;margin-bottom:8px;'>"
                f"  <div style='display:flex;justify-content:space-between;"
                f"align-items:center;margin-bottom:8px;'>"
                f"    <div style='display:flex;align-items:center;gap:10px;'>"
                f"      <div style='width:10px;height:10px;border-radius:50%;"
                f"background:{color};box-shadow:0 0 8px {color}66;'></div>"
                f"      <span style='color:#e2e8f0;font-weight:600;"
                f"font-size:0.95rem;'>{name}</span>"
                f"    </div>"
                f"    <div style='display:flex;gap:18px;font-size:0.85rem;'>"
                f"      <span style='color:#94a3b8;'>Giriş: "
                f"<strong style='color:#f1f5f9;font-weight:700;'>{ent_str}</strong></span>"
                f"      <span style='color:#94a3b8;'>Süre: "
                f"<strong style='color:{color};font-weight:700;'>{pct:.1f}%</strong></span>"
                f"    </div>"
                f"  </div>"
                f"  <div style='height:6px;background:rgba(255,255,255,0.05);"
                f"border-radius:100px;overflow:hidden;'>"
                f"    <div style='height:100%;background:{color};"
                f"width:{bar_w:.2f}%;border-radius:100px;"
                f"box-shadow:0 0 6px {color}55;'></div>"
                f"  </div>"
                f"</div>"
            )
        st.markdown(cards_html, unsafe_allow_html=True)

        with st.expander("Giriş dizisi"):
            seq = row.get("entry_sequence", "")
            st.code(seq if seq else "(giriş yok)")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────────

with tab_viz:
    if not images:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
             border-radius:12px;padding:1.25rem;text-align:center;color:rgba(255,255,255,0.5);">
          ⚠️  Görseller üretilemedi.
          <code>orbit_plot.py</code> veya <code>activity_heatmap.py</code> çıktısını kontrol edin.
        </div>
        """, unsafe_allow_html=True)
    else:
        viz_defs = [
            ("orbit",       "Hareket Rotası"),
            ("heatmap_kde", "Isı Haritası"),
        ]
        for key, title in viz_defs:
            if key not in images:
                continue
            st.markdown(
                f'<div style="text-align:center;font-size:1.15rem;'
                f'font-weight:700;color:#e2e8f0;letter-spacing:0.08em;'
                f'margin:0.6rem 0 0.6rem 0;">{title}</div>',
                unsafe_allow_html=True,
            )
            _, cimg, _ = st.columns([1, 3, 1])
            with cimg:
                st.image(images[key], use_container_width=True)
            st.markdown('<div style="height:1.25rem;"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — İNDİR
# ─────────────────────────────────────────────────────────────────────────────

def _stick_to_dl_tab() -> None:
    st.session_state["_return_to_dl_tab"] = True


with tab_dl:
    safe_row = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in row.items() if not k.startswith("_")
    }
    safe_row.update({
        "pred_label":    pred_label,
        "proba_control": round(pred_info["proba_control"], 4),
        "proba_treated": round(pred_info["proba_treated"], 4),
    })
    csv_bytes = pd.DataFrame([safe_row]).to_csv(
        index=False, encoding="utf-8-sig").encode("utf-8-sig")
    json_bytes = json.dumps(
        {
            "subject":          subject,
            "csv_file":         uploaded.name,
            "pred_label":       pred_label,
            "proba_control":    round(pred_info["proba_control"], 4),
            "proba_treated":    round(pred_info["proba_treated"], 4),
            "top_contributors": pred_info["top_contributors"],
            "features_used":    pred_info["feature_cols"],
            "metrics":          safe_row,
        },
        indent=2, ensure_ascii=False,
    ).encode("utf-8")

    st.markdown(html_section("Rapor Dosyaları"), unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3, gap="medium")

    with dc1:
        st.download_button(
            "⬇  Metrikler (CSV)", csv_bytes,
            file_name=f"{subject}_epm_metrics.csv", mime="text/csv",
            use_container_width=True,
            key="dl_csv",
            on_click=_stick_to_dl_tab,
        )
        with st.expander("Önizleme"):
            preview_df = pd.DataFrame([safe_row]).T.reset_index()
            preview_df.columns = ["Alan", "Değer"]
            st.dataframe(
                preview_df, hide_index=True,
                use_container_width=True, height=320,
            )

    with dc2:
        st.download_button(
            "⬇  Tahmin Raporu (JSON)", json_bytes,
            file_name=f"{subject}_epm_report.json", mime="application/json",
            use_container_width=True,
            key="dl_json",
            on_click=_stick_to_dl_tab,
        )
        with st.expander("Önizleme"):
            st.json(json.loads(json_bytes.decode("utf-8")), expanded=False)

    with dc3:
        st.download_button(
            "⬇  Genel Bakış (PNG)", overview_bytes,
            file_name=f"{subject}_epm_overview.png", mime="image/png",
            use_container_width=True,
            key="dl_overview",
            on_click=_stick_to_dl_tab,
        )
        with st.expander("Önizleme"):
            st.image(overview_bytes, use_container_width=True)

    if images:
        st.markdown(html_section("Analiz Görselleri"), unsafe_allow_html=True)
        viz_map = [
            ("orbit",       "Hareket Rotası"),
            ("heatmap_kde", "Isı Haritası"),
        ]
        viz_map = [(k, lbl) for k, lbl in viz_map if k in images]
        if viz_map:
            dcols = st.columns(len(viz_map), gap="medium")
            for (key, lbl), col in zip(viz_map, dcols):
                with col:
                    st.download_button(
                        f"⬇  {lbl} (PNG)", images[key],
                        file_name=f"{subject}_{key}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{key}",
                        on_click=_stick_to_dl_tab,
                    )
                    with st.expander("Önizleme"):
                        st.image(images[key], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# RERUN sonrası İndir sekmesine geri dön (download_button rerun tetikler)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.pop("_return_to_dl_tab", False):
    st.components.v1.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const targetLabel = 'İndir';
          const click = () => {
            const tabs = doc.querySelectorAll('button[role="tab"]');
            for (const t of tabs) {
              if ((t.innerText || '').includes(targetLabel)) {
                if (t.getAttribute('aria-selected') !== 'true') t.click();
                return true;
              }
            }
            return false;
          };
          if (!click()) {
            let tries = 0;
            const iv = setInterval(() => {
              if (click() || ++tries > 20) clearInterval(iv);
            }, 50);
          }
        })();
        </script>
        """,
        height=0,
    )
