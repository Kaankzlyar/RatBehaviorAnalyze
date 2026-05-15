# -*- coding: utf-8 -*-
"""
predict_anxiety_v3_gui.py — Open Field anksiyete tahmin ve davranış dashboard'u

Tek farelik DLC pose CSV'si yüklenir; davranış tespiti, OFT metrikleri,
mekansal rearing ve ML tabanlı **Kontrol / Tedavi** sınıflandırması yapılır.
İsteğe bağlı olarak run_analysis.py üzerinden tam open-field analiz
pipeline'ı (timeline, orbit, ısı haritası, bodypart) da çalıştırılır.

Çalıştırmak için:
    streamlit run scripts/predict_anxiety_v3_gui.py
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predict_anxiety_v2 as v2  # noqa: E402


# ── Sayfa konfigürasyonu ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Open Field — Tahmin Paneli",
    page_icon="🐀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0b0f1a; }
.main .block-container { padding-top: 1.5rem; max-width: 1280px; }

[data-testid="stSidebar"] {
    background: #0f1320 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

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

.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 100px !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 100px !important;
    height: 6px !important;
}

[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }
.dvn-scroller { border-radius: 12px !important; }

hr { border-color: rgba(255,255,255,0.06) !important; }
.stAlert { border-radius: 10px !important; }

.streamlit-expanderHeader {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.85rem !important;
    background: rgba(255,255,255,0.02) !important;
    border-radius: 8px !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }

#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── HTML bileşenleri ──────────────────────────────────────────────────────────

_TR_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})


def tr_upper(s: str) -> str:
    return s.translate(_TR_UPPER_MAP).upper()


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


# ── Open-Field özelliklerinin Türkçe açıklamaları ─────────────────────────────

FEATURE_INFO: dict[str, dict[str, str]] = {
    "pct_center": {
        "name":       "Merkez Süresi (%)",
        "name_short": "Merkez Süresi",
        "desc": "Farenin arenanın iç bölgesinde (center zone) geçirdiği zamanın "
                "yüzdesidir. Düşük değer çevreye yapışma (thigmotaxis) ve yüksek "
                "anksiyeteye işaret eder.",
        "formula": "center_frame_sayısı / geçerli_kare_sayısı × 100",
    },
    "pct_periphery": {
        "name":       "Çevre Süresi (%)",
        "name_short": "Çevre Süresi",
        "desc": "Farenin arenanın dış bandında (duvar kenarında) geçirdiği zamanın "
                "yüzdesidir. Yüksek değer thigmotaxis davranışını işaret eder.",
        "formula": "periphery_frame_sayısı / geçerli_kare_sayısı × 100",
    },
    "pct_freeze": {
        "name":       "Donakalma (%)",
        "name_short": "Donakalma",
        "desc": "Hız eşiğinin altında kalan kare oranıdır. Yüksek değer korku / "
                "anksiyeteye bağlı immobilizasyona işaret eder.",
        "formula": "frozen_frame_sayısı / toplam_kare × 100",
    },
    "spatial_entropy": {
        "name":       "Mekansal Entropi",
        "name_short": "Mekansal Entropi",
        "desc": "Farenin arenayı ne kadar dengeli kullandığını ölçer (Shannon "
                "entropisi, 0–1 normalize). Düşük değer dar bir alana takılı "
                "kalmayı, yüksek değer geniş keşfi gösterir.",
        "formula": "H(p) / log(N_grid)  —  ızgara üzerindeki ziyaret olasılığı",
    },
    "center_entries": {
        "name":       "Merkeze Giriş Sayısı",
        "name_short": "Merkez Girişi",
        "desc": "Farenin çevreden merkez bölgesine geçiş sayısıdır. Düşük değer "
                "merkez kaçınmasını / anksiyeteyi işaret eder.",
        "formula": "periphery → center geçişlerinin sayısı",
    },
    "total_distance_px": {
        "name":       "Toplam Mesafe (px)",
        "name_short": "Toplam Mesafe",
        "desc": "Farenin hareketi boyunca kat ettiği toplam yol (piksel). Genel "
                "lokomotor aktivitenin ölçüsüdür.",
        "formula": "Σ √(dx² + dy²)",
    },
    "mean_speed_px_s": {
        "name":       "Ortalama Hız (px/sn)",
        "name_short": "Ortalama Hız",
        "desc": "Geçerli karelerde ortalama yer değiştirme miktarı. Genel motor "
                "aktivitenin doğrudan ölçüsüdür.",
        "formula": "ortalama( √(dx² + dy²) × fps )",
    },
    "freeze_bout_count": {
        "name":       "Donakalma Bout Sayısı",
        "name_short": "Donakalma Bout",
        "desc": "Donakalma sayılan ardışık kare bloklarının sayısı. Sayıca "
                "fazla donakalma bouts'u kaygılı / temkinli davranış göstergesidir.",
        "formula": "freeze rejimine giriş–çıkış geçişlerinin sayısı",
    },
    "rear_count": {
        "name":       "Ayakta Durma Sayısı",
        "name_short": "Rearing Sayısı",
        "desc": "Farenin arka ayakları üzerinde dikildiği (rearing) bout sayısı. "
                "Keşfetme ve dikey aktivite göstergesidir.",
        "formula": "DLC keypoint'lerinden tespit edilen rearing bout'larının sayısı",
    },
    "rear_total_s": {
        "name":       "Toplam Rearing Süresi (sn)",
        "name_short": "Rearing Süresi",
        "desc": "Rearing davranışında geçirilen toplam süre (saniye).",
        "formula": "Σ rearing_bout_duration",
    },
    "rear_pct": {
        "name":       "Rearing Yüzdesi (%)",
        "name_short": "Rearing %",
        "desc": "Farenin hareketinin toplam süresine oranla rearing'de "
                "geçirilen süre yüzdesi.",
        "formula": "rear_total_s / session_s × 100",
    },
    "rear_mean_bout_s": {
        "name":       "Ortalama Rearing Bout Süresi (sn)",
        "name_short": "Ort. Rearing Bout",
        "desc": "Bir rearing bout'unun ortalama uzunluğu. Uzun bouts daha "
                "kararlı, kısa bouts daha çekingen keşfi yansıtabilir.",
        "formula": "rear_total_s / rear_count",
    },
    "rear_rate_per_min": {
        "name":       "Rearing Sıklığı (dk⁻¹)",
        "name_short": "Rearing Sıklığı",
        "desc": "Dakika başına rearing bout sayısı. Keşfetme dürtüsünün "
                "zaman-normalize ölçüsü.",
        "formula": "60 × rear_count / session_s",
    },
    "rear_early_frac": {
        "name":       "Erken Rearing Oranı",
        "name_short": "Erken Rearing",
        "desc": "Farenin hareketinin erken bölümündeki (ilk 90 saniye) "
                "rearing'in toplam rearing'e oranı. Yüksek değer hareketin "
                "başında yoğun keşfi, düşük değer sonradan ısınan keşfi gösterir.",
        "formula": "erken_pencerede_rear_süresi / toplam_rear_süresi",
    },
    "rear_count_center": {
        "name":       "Merkezde Rearing Sayısı",
        "name_short": "Merkez Rearing",
        "desc": "Merkez bölgesinde gerçekleşen rearing bout sayısı. Yüksek "
                "değer cesur / düşük anksiyeteli keşfi gösterir.",
        "formula": "merkez bölgesinde başlayan rearing bout'larının sayısı",
    },
    "rear_count_wall": {
        "name":       "Duvarda Rearing Sayısı",
        "name_short": "Duvar Rearing",
        "desc": "Çevre (duvar) bölgesinde gerçekleşen rearing bout sayısı. "
                "Yüksek değer çevreye yapışan, daha kaygılı keşfi gösterebilir.",
        "formula": "çevre bölgesinde başlayan rearing bout'larının sayısı",
    },
    "rear_center_frac": {
        "name":       "Merkez Rearing Oranı",
        "name_short": "Merkez Rearing %",
        "desc": "Tüm rearing'ler içinde merkezde olanların oranı (0–1). Yüksek "
                "değer = merkezde dikilmekten çekinmiyor = düşük anksiyete.",
        "formula": "rear_count_center / (rear_count_center + rear_count_wall)",
    },
    "groom_count": {
        "name":       "Tımarlanma Sayısı",
        "name_short": "Grooming Sayısı",
        "desc": "Grooming (kendini tımarlama) bout sayısı. Stres altında "
                "kompulsif olarak artabilir.",
        "formula": "DLC keypoint'lerinden tespit edilen grooming bout sayısı",
    },
    "groom_total_s": {
        "name":       "Toplam Grooming Süresi (sn)",
        "name_short": "Grooming Süresi",
        "desc": "Grooming davranışında geçirilen toplam saniye.",
        "formula": "Σ grooming_bout_duration",
    },
    "groom_pct": {
        "name":       "Grooming Yüzdesi (%)",
        "name_short": "Grooming %",
        "desc": "Farenin hareketinin toplam süresine oranla grooming'de "
                "geçirilen süre yüzdesi.",
        "formula": "groom_total_s / session_s × 100",
    },
    "groom_mean_bout_s": {
        "name":       "Ortalama Grooming Bout Süresi (sn)",
        "name_short": "Ort. Groom Bout",
        "desc": "Bir grooming bout'unun ortalama uzunluğu.",
        "formula": "groom_total_s / groom_count",
    },
    "groom_bout_cv": {
        "name":       "Grooming Bout CV",
        "name_short": "Groom Bout CV",
        "desc": "Grooming bout sürelerinin varyasyon katsayısı (std/mean). "
                "Yüksek değer düzensiz bout süreleri (kompulsif kalıp) gösterebilir.",
        "formula": "std(groom_durations) / mean(groom_durations)",
    },
    "groom_early_frac": {
        "name":       "Erken Grooming Oranı",
        "name_short": "Erken Grooming",
        "desc": "Farenin hareketinin erken bölümündeki (ilk 90 saniye) "
                "grooming'in toplama oranı.",
        "formula": "erken_pencerede_groom_süresi / toplam_groom_süresi",
    },
    "rear_groom_ratio": {
        "name":       "Rearing / Grooming Oranı",
        "name_short": "Rear / Groom",
        "desc": "Toplam rearing süresinin toplam grooming süresine oranı. "
                "Yüksek değer keşif baskın, düşük değer stres davranışı baskın.",
        "formula": "rear_total_s / groom_total_s",
    },
    "rear_per_100px": {
        "name":       "100 px Başına Rearing",
        "name_short": "Rearing / 100px",
        "desc": "Birim hareket başına rearing — yatay aktiviteden bağımsız bir "
                "dikey aktivite ölçüsü.",
        "formula": "100 × rear_count / total_distance_px",
    },
    "comfort_ratio": {
        "name":       "Konfor Oranı",
        "name_short": "Konfor Oranı",
        "desc": "Grooming süresinin merkez süresine oranı. Yüksek değer "
                "merkezde rahat hissettiğini, düşük değer merkezde tedirginliği "
                "gösterebilir.",
        "formula": "groom_total_s / pct_center",
    },
    "rear_total_s_center": {
        "name":       "Merkezde Toplam Rearing (sn)",
        "name_short": "Merkez Rear (sn)",
        "desc": "Merkez bölgesinde gerçekleşen rearing'lerin toplam süresi.",
        "formula": "Σ merkez_rearing_bout_duration",
    },
    "rear_total_s_wall": {
        "name":       "Duvarda Toplam Rearing (sn)",
        "name_short": "Duvar Rear (sn)",
        "desc": "Çevre / duvar bölgesinde gerçekleşen rearing'lerin toplam süresi.",
        "formula": "Σ duvar_rearing_bout_duration",
    },
    "rear_center_minus_wall": {
        "name":       "Merkez − Duvar Rearing",
        "name_short": "Merkez − Duvar",
        "desc": "Merkezdeki ve duvardaki rearing bout sayıları arasındaki fark. "
                "Pozitif değer merkez tercih, negatif değer duvar tercih.",
        "formula": "rear_count_center − rear_count_wall",
    },
}


# ── Matplotlib stil ───────────────────────────────────────────────────────────

_DARK_BG  = "#0b0f1a"
_PANEL_BG = "#0f1320"
_GRID_CLR = "#1e2438"
_MUTED    = "#475569"


def _style_ax_dark(ax, fig=None):
    if fig:
        fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_CLR)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color("#94a3b8")


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
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.18)
    return fig


# ── Open-field analiz subprocess wrapper ──────────────────────────────────────

def run_open_field_analysis(
    csv_path: Path,
    arena: tuple,
    inner_zone: tuple,
    fps: float = 30.0,
    skip_behavior: bool = False,
    skip_orbit: bool = False,
    skip_heatmap: bool = False,
    skip_bodypart: bool = False,
    progress_container=None,
) -> dict:
    """run_analysis.py'i alt-process olarak çalıştırır; üretilen PNG/CSV'leri
    aynı dizinde tarar ve key → Path eşlemesi döndürür."""
    results = {
        "behavior_bouts":   None,
        "behavior_frames":  None,
        "behavior_timeline":None,
        "orbit_grid":       None,
        "thigmotaxis":      None,
        "heatmap_kde":      None,
        "heatmap_histogram":None,
        "bodypart_heatmaps":None,
    }

    common_args = [
        str(ROOT / "analysis" / "open_field" / "run_analysis.py"),
        "--arena", *[str(x) for x in arena],
        "--inner-zone", *[str(x) for x in inner_zone],
        "--csv", str(csv_path),
        "--fps", str(fps),
        "--likelihood", "0.6",
        "--jump-thresh", "60",
        "--smooth", "5",
    ]
    if skip_behavior:  common_args.append("--skip-behavior")
    if skip_orbit:     common_args.append("--skip-orbit")
    if skip_heatmap:   common_args.append("--skip-heatmap")
    if skip_bodypart:  common_args.append("--skip-bodypart")

    cmd = [sys.executable] + common_args
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(ROOT / "analysis" / "open_field"),
    )
    for line in process.stdout:
        if progress_container and ("[*]" in line or "[OK]" in line or "Step" in line):
            progress_container.write(f"🔄 {line.rstrip()}")
    _, stderr = process.communicate(timeout=600)
    if process.returncode != 0:
        raise RuntimeError(f"Analysis failed: {stderr}")

    csv_dir = csv_path.parent
    stem = csv_path.stem
    suffix_map = {
        "_behavior_bouts.csv":   "behavior_bouts",
        "_behavior_frames.csv":  "behavior_frames",
        "_behavior_timeline.png":"behavior_timeline",
        "_orbit_grid.png":       "orbit_grid",
        "_thigmotaxis.png":      "thigmotaxis",
        "_heatmap_kde.png":      "heatmap_kde",
        "_heatmap_histogram.png":"heatmap_histogram",
        "_bodypart_heatmaps.png":"bodypart_heatmaps",
    }
    # Bireysel per-bodypart PNG'leri ayrı listelerde topla
    results["orbit_individuals"]    = []
    results["bodypart_individuals"] = []
    for file in csv_dir.glob(f"{stem}*"):
        if "_orbit_bp_" in file.name and file.suffix == ".png":
            bp = file.stem.split("_orbit_bp_", 1)[1]
            results["orbit_individuals"].append((bp, file))
            continue
        if "_bodypart_bp_" in file.name and file.suffix == ".png":
            bp = file.stem.split("_bodypart_bp_", 1)[1]
            results["bodypart_individuals"].append((bp, file))
            continue
        for suf, key in suffix_map.items():
            if file.name.endswith(suf):
                results[key] = file
    # Tutarlı bir sırada tut
    results["orbit_individuals"].sort(key=lambda x: x[0])
    results["bodypart_individuals"].sort(key=lambda x: x[0])
    return results


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
        html_stat_row("Algoritma",     "Lojistik Regresyon") +
        html_stat_row("Eğitim seti",   "29 sıçan (LOOCV)") +
        html_stat_row("Özellik sayısı","12") +
        html_stat_row("AUC",           "0.683") +
        html_stat_row("F1 (Makro)",   "0.72") +
        html_stat_row("F1 (Ağırlıklı)",  "0.83") +
        html_stat_row("Doğruluk",      "%82.8"),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:1px;background:rgba(255,255,255,0.06);margin:1rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:600;color:rgba(255,255,255,0.3);'
        'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">'
        'Kullanım</div>',
        unsafe_allow_html=True,
    )
    for i, step in enumerate(
        ["DLC filtered CSV yükle",
         "Çalıştır butonuna bas",
         "Sonuçları incele & indir"], 1
    ):
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:center;padding:6px 0;">'
            f'<div style="width:22px;height:22px;border-radius:50%;background:rgba(255,255,255,0.05);'
            f'border:1px solid rgba(255,255,255,0.14);display:flex;align-items:center;'
            f'justify-content:center;font-size:0.7rem;font-weight:600;color:#94a3b8;'
            f'flex-shrink:0;">{i}</div>'
            f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.55);">{step}</span></div>',
            unsafe_allow_html=True,
        )


# ── Pipeline varsayılanları (artık sidebar'da seçilemiyor) ───────────────────
fps             = float(v2.DEFAULT_FPS)
tag             = "rearonly"
run_analysis    = True
arena_xmin, arena_xmax = 397, 777
arena_ymin, arena_ymax = 156, 535
skip_behavior   = False
skip_orbit      = False
fast_mode       = False   # tüm adımlar (heatmap + bodypart dahil)
auto_inner      = True
save_to_reports = True
inner_xmin = inner_xmax = inner_ymin = inner_ymax = None


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
        Open Field  —  Tahmin Paneli
      </h1>
      <p style="margin:4px 0 0 0;font-size:0.82rem;color:rgba(255,255,255,0.45);">
        DeepLabCut pose verisi · Davranış (rearing / grooming) · Mekansal metrikler · ML tabanlı grup tahmini
      </p>
    </div>
  </div>
  <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap;">
    <span style="background:rgba(99,102,241,0.18);border:1px solid rgba(99,102,241,0.35);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#818cf8;letter-spacing:0.04em;">Lojistik Regresyon</span>
    <span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.30);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#34d399;letter-spacing:0.04em;">LOOCV · n=29</span>
    <span style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.30);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#fbbf24;letter-spacing:0.04em;">AUC 0.683</span>
    <span style="background:rgba(168,85,247,0.15);border:1px solid rgba(168,85,247,0.30);
          border-radius:100px;padding:3px 12px;font-size:0.72rem;font-weight:500;
          color:#c084fc;letter-spacing:0.04em;">F1 0.72</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DOSYA YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────

upload_col, btn_col = st.columns([3, 1], gap="medium")
with upload_col:
    uploaded = st.file_uploader(
        "DLC filtered pose CSV (tek fare)",
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

# ── Yüklenen dosyanın kimliğini hash'le; aynı dosyada rerun olunca cache'ten oku ──
_h = hashlib.sha256()
_h.update(uploaded.name.encode("utf-8"))
_h.update(b"|")
_h.update(uploaded.getvalue())
current_key = _h.hexdigest()

prev_key = st.session_state.get("v3_results_key")
if prev_key is not None and prev_key != current_key:
    # Yeni bir dosya yüklendi → eski sonuçları unut
    st.session_state.pop("v3_results", None)
    st.session_state.pop("v3_results_key", None)

cache_hit = (
    st.session_state.get("v3_results_key") == current_key
    and "v3_results" in st.session_state
)

if not cache_hit and not run_clicked:
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
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

analysis_results: dict = {}
analysis_images: dict[str, bytes] = {}
available_outputs: set[str] = set()
fig_images: dict[str, bytes] = {}

if cache_hit:
    # Aynı dosya hâlâ yüklü; pipeline'ı yeniden çalıştırma, cache'ten yükle.
    _r = st.session_state["v3_results"]
    pred_info         = _r["pred_info"]
    feat              = _r["feat"]
    report_text       = _r["report_text"]
    json_bytes        = _r["json_bytes"]
    txt_name          = _r["txt_name"]
    json_name         = _r["json_name"]
    fig_images        = _r["fig_images"]
    analysis_images   = _r["analysis_images"]
    available_outputs = _r["available_outputs"]
    analysis_results  = _r.get("analysis_results", {})
else:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / uploaded.name
        csv_path.write_bytes(uploaded.getvalue())

        status_ph = st.empty()
        prog = st.progress(0.0)

        def _step(p: float, msg: str):
            prog.progress(min(max(p, 0.0), 1.0))
            status_ph.markdown(
                f'<div style="font-size:0.85rem;color:rgba(255,255,255,0.5);'
                f'padding:0.3rem 0;">{msg}</div>',
                unsafe_allow_html=True,
            )

        try:
            if run_analysis:
                _step(0.05, "🧪 Open-field analiz pipeline başlıyor…")
                if auto_inner:
                    w = arena_xmax - arena_xmin
                    h = arena_ymax - arena_ymin
                    inner_xmin = arena_xmin + 0.20 * w
                    inner_xmax = arena_xmax - 0.20 * w
                    inner_ymin = arena_ymin + 0.20 * h
                    inner_ymax = arena_ymax - 0.20 * h

                arena_bounds = (arena_xmin, arena_xmax, arena_ymin, arena_ymax)
                inner_zone_bounds = (inner_xmin, inner_xmax, inner_ymin, inner_ymax)

                progress_output = st.empty()
                analysis_results = run_open_field_analysis(
                    csv_path,
                    arena_bounds,
                    inner_zone_bounds,
                    fps=fps,
                    skip_behavior=skip_behavior,
                    skip_orbit=skip_orbit,
                    skip_heatmap=fast_mode,
                    skip_bodypart=fast_mode,
                    progress_container=progress_output,
                )
                for key in ("behavior_timeline", "orbit_grid", "thigmotaxis",
                            "heatmap_histogram", "heatmap_kde", "bodypart_heatmaps"):
                    p = analysis_results.get(key)
                    if p and p.exists():
                        try:
                            analysis_images[key] = p.read_bytes()
                            available_outputs.add(key)
                        except Exception:
                            pass
                progress_output.empty()

            _step(0.30, "📍 Davranış bout'ları (rearing / grooming) çıkarılıyor")
            bouts_df, n_frames = v2.detect_bouts(csv_path, fps=fps)

            _step(0.50, "📐 OFT metrikleri (lokomosyon / thigmotaxis / freeze / entropi)")
            oft, body_x, body_y = v2.compute_oft_metrics(csv_path, fps=fps)

            _step(0.65, "🎯 Mekansal rearing (merkez vs. duvar)")
            spatial = v2.compute_spatial_rearing(bouts_df, body_x, body_y)
            feat = v2.build_feature_dict(bouts_df, oft, spatial, oft["session_s"])

            _step(0.80, f"🧠 Model yükleme ve tahmin (tag={tag})")
            bundle, model = v2.load_model(tag)
            pred_info = v2.predict(feat, bundle, model)

            _step(0.90, "📊 Rapor ve görseller üretiliyor")

            subject = csv_path.stem
            out_dir = tmp_path / "out"
            out_dir.mkdir()

            report_buf = StringIO()
            v2.render_summary(report_buf, csv_path, n_frames, feat, pred_info)
            report_text = report_buf.getvalue()
            txt_name = f"{subject}_anxiety_v2_report.txt"

            fig_path = out_dir / f"{subject}_anxiety_v2_overview.png"
            v2.plot_overview(csv_path, body_x, body_y, spatial, pred_info, fig_path)
            fig_images["anxiety_overview"] = fig_path.read_bytes()

            json_payload = {
                "subject":         subject,
                "csv_path":        uploaded.name,
                "n_frames":        n_frames,
                "session_s":       oft["session_s"],
                "model_tag":       tag,
                "pred":            pred_info["pred"],
                "label":           v2.label_str(pred_info["pred"]),
                "proba_control":   round(pred_info["proba_control"], 4),
                "proba_treated":   round(pred_info["proba_treated"], 4),
                "top_contributors": pred_info["top_contributors"],
                "features":        {k: (None if isinstance(val, float) and np.isnan(val) else val)
                                     for k, val in feat.items()},
                "feature_cols_used": pred_info["feature_cols"],
                "n_rearing_bouts":  int((bouts_df["behaviour"] == "rearing").sum()),
                "n_grooming_bouts": int((bouts_df["behaviour"] == "grooming").sum()),
                "spatial_rearing": {
                    "rear_count_center":   spatial["rear_count_center"],
                    "rear_count_wall":     spatial["rear_count_wall"],
                    "rear_count_unknown":  spatial["rear_count_unknown"],
                    "rear_center_frac":    (None if spatial["rear_center_frac"] != spatial["rear_center_frac"]
                                            else round(spatial["rear_center_frac"], 4)),
                },
                "generated_at":    datetime.now().isoformat(timespec="seconds"),
            }
            json_bytes = json.dumps(json_payload, indent=2, ensure_ascii=False).encode("utf-8")
            json_name = f"{subject}_anxiety_v2_report.json"

            if save_to_reports:
                target = v2.DEFAULT_OUT
                target.mkdir(parents=True, exist_ok=True)
                (target / json_name).write_bytes(json_bytes)
                (target / txt_name).write_text(report_text, encoding="utf-8")
                (target / f"{subject}_anxiety_v2_overview.png").write_bytes(fig_images["anxiety_overview"])
                for key, img_bytes in analysis_images.items():
                    (target / f"{subject}_{key}.png").write_bytes(img_bytes)

            _step(1.0, "✓  Tamamlandı")
            prog.empty()
            status_ph.empty()

        except FileNotFoundError as e:
            st.error("Model dosyaları bulunamadı.")
            st.code(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Pipeline hatası: {type(e).__name__}: {e}")
            st.exception(e)
            st.stop()

    # Pipeline başarıyla bitti — sonuçları cache'le ki sonraki rerun (download
    # butonu vs.) pipeline'ı tekrar çalıştırmasın.
    st.session_state["v3_results"] = {
        "pred_info":         pred_info,
        "feat":              feat,
        "report_text":       report_text,
        "json_bytes":        json_bytes,
        "txt_name":          txt_name,
        "json_name":         json_name,
        "fig_images":        fig_images,
        "analysis_images":   analysis_images,
        "available_outputs": available_outputs,
        "analysis_results":  analysis_results,
    }
    st.session_state["v3_results_key"] = current_key


# ─────────────────────────────────────────────────────────────────────────────
# SONUÇLAR
# ─────────────────────────────────────────────────────────────────────────────

is_treated = pred_info["pred"] == 1
pred_label = "Tedavi" if is_treated else "Kontrol"
top_proba  = pred_info["proba_treated"] if is_treated else pred_info["proba_control"]
subject    = Path(uploaded.name).stem

st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

# Sekme paneli için fade-in animasyonu — aktif tab değiştiğinde içerik
# anında geçmek yerine yumuşakça belirir, böylece "flash" hissi azalır.
st.markdown("""
<style>
[data-baseweb="tab-panel"] {
    animation: tab-fade-in 0.22s ease-out;
}
@keyframes tab-fade-in {
    from { opacity: 0; transform: translateY(2px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# st.fragment (Streamlit 1.37+) sayesinde sekme değişimi yalnızca bu
# fonksiyonu yeniden çalıştırır — hero / upload / pipeline kısmı yeniden
# render edilmez, bu da sekmeler arası "flash up/down" sıçramasını ortadan
# kaldırır. Eski sürümde fragment yoksa no-op decorator'a düşer (eski davranış).
_fragment = getattr(st, "fragment", None) or (lambda f: f)


@_fragment
def _render_tab_section():
    tab_pred, tab_beh, tab_metrics, tab_dl = st.tabs(
        ["🎯  Tahmin", "🧠  Davranış", "📊  Metrikler", "⬇  İndir"]
    )
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 1 — TAHMİN
    # ─────────────────────────────────────────────────────────────────────────────
    
    with tab_pred:
        col_img, col_right = st.columns([1.35, 1], gap="large")
    
        with col_img:
            st.markdown(html_section("Hareket Haritası"), unsafe_allow_html=True)
            st.image(fig_images["anxiety_overview"], use_container_width=True)
    
        with col_right:
            st.markdown(html_section("Tahmin Sonucu"), unsafe_allow_html=True)
            st.markdown(html_pred_card(pred_label, top_proba, is_treated), unsafe_allow_html=True)
    
            buf_p = io.BytesIO()
            fp = fig_probability_bars(pred_info)
            fp.savefig(buf_p, format="png", dpi=130, bbox_inches="tight", facecolor=_DARK_BG)
            plt.close(fp)
            st.image(buf_p.getvalue(), use_container_width=True)
    
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
                ("Merkez Süresi", f"%{feat['pct_center']:.1f}",
                 "Arenanın iç bölgesinde geçen sürenin yüzdesi. Düşük değer "
                 "merkez kaçınmasına / anksiyeteye işaret eder."),
                ("Çevre Süresi", f"%{feat['pct_periphery']:.1f}",
                 "Duvar kenarında geçen sürenin yüzdesi. Yüksek değer "
                 "thigmotaxis (duvara yapışma) davranışını işaret eder."),
                ("Donakalma", f"%{feat['pct_freeze']:.1f}",
                 "Hız eşiğinin altında kalan kare oranı. Yüksek değer kaygı "
                 "kaynaklı immobilizasyonu ima eder."),
                ("Toplam Rearing", f"{int(feat['rear_count'])}",
                 "Farenin hareketi boyunca arka ayaklar üzerinde dikilme (rearing) "
                 "bout sayısı. Keşfetme ve dikey aktivite göstergesidir."),
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
    
        # Karar Katkıları — tam genişlik
        st.markdown(html_section("Karar Katkıları"), unsafe_allow_html=True)
        buf_f = io.BytesIO()
        ff = fig_feature_contributions(pred_info)
        ff.savefig(buf_f, format="png", dpi=130, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(ff)
        st.image(buf_f.getvalue(), use_container_width=True)
    
        for c in pred_info["top_contributors"]:
            feat_key = c["feature"]
            info = FEATURE_INFO.get(feat_key)
            push  = c["push"]
            toward = "Tedavi Grubu" if push > 0 else "Kontrol Grubu"
            arrow_color = "#EF4444" if push > 0 else "#3B82F6"
            display_name = info["name"] if info else feat_key.replace("_", " ").title()
            value = c.get("value")
            value_str = f"{value:.3f}" if isinstance(value, (int, float)) else "—"
    
            header = f"{display_name}   ·   katkı {push:+.3f}   →   {toward}"
            with st.expander(header, expanded=False):
                if info:
                    st.markdown(
                        f"<div style='font-size:0.72rem;font-weight:700;"
                        f"letter-spacing:0.14em;color:#94a3b8;"
                        f"margin:0.1rem 0 0.4rem 0;'>AÇIKLAMA</div>"
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
                    f"(<strong style='color:{arrow_color};'>{c['coef']:+.3f}</strong>) "
                    f"ile çarpılınca <strong style='color:{arrow_color};'>{push:+.3f}</strong> "
                    f"büyüklüğünde, <strong style='color:{arrow_color};'>{toward}</strong> "
                    f"yönünde bir karar katkısı üretir."
                    f"</div>",
                    unsafe_allow_html=True,
                )
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 2 — DAVRANIŞ
    # ─────────────────────────────────────────────────────────────────────────────
    
    with tab_beh:
        if not available_outputs:
            st.info(
                "Bu sekme, sol paneldeki **Detaylı analizi çalıştır** seçeneği "
                "aktif olduğunda dolar. Şu an Open-Field analiz pipeline'ı "
                "çalıştırılmadığı için davranışsal görseller üretilmedi."
            )
        else:
            # ── Davranış Zaman Çizgisi ───────────────────────────────────────────
            if "behavior_timeline" in available_outputs:
                st.markdown(html_section("Davranış Zaman Çizgisi"), unsafe_allow_html=True)
                _, cimg, _ = st.columns([1, 3, 1])
                with cimg:
                    st.image(analysis_images["behavior_timeline"], use_container_width=True)
                st.markdown('<div style="height:1.25rem;"></div>', unsafe_allow_html=True)
    
            # ── Thigmotaksis — yan istatistik tablosu (dikey ortalı) ─────────────
            if "thigmotaxis" in available_outputs:
                # Başlık literal-uppercase — tr_upper'ın "Thigmotaxis"'i "THİGMOTAXİS"
                # yapmasını engellemek için doğrudan uppercase verilir.
                st.markdown(html_section("THIGMOTAXIS"), unsafe_allow_html=True)
                try:
                    col_img, col_stat = st.columns(
                        [2, 1], gap="medium", vertical_alignment="center",
                    )
                except TypeError:
                    # Streamlit < 1.36 fallback (vertical_alignment yok)
                    col_img, col_stat = st.columns([2, 1], gap="medium")
                with col_img:
                    st.image(analysis_images["thigmotaxis"], use_container_width=True)
                with col_stat:
                    _pp  = feat.get("pct_periphery") or 0.0
                    _pc  = feat.get("pct_center") or 0.0
                    _pj  = max(0.0, 100.0 - _pp - _pc)
                    rows_html = (
                        html_stat_row("Çevre (Thigmotaxis)", f"%{_pp:.1f}") +
                        html_stat_row("Merkez",                f"%{_pc:.1f}") +
                        html_stat_row("Diğer / kenar",         f"%{_pj:.1f}")
                    )
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.03);"
                        f"border:1px solid rgba(255,255,255,0.07);border-radius:12px;"
                        f"padding:0.85rem 1rem;'>{rows_html}"
                        f"<div style='font-size:0.85rem;color:rgba(255,255,255,0.65);"
                        f"line-height:1.55;margin-top:0.8rem;border-top:1px solid "
                        f"rgba(255,255,255,0.06);padding-top:0.7rem;'>"
                        f"<strong style='color:#fbbf24;'>Thigmotaxis</strong> = "
                        f"Farenin arenanın <em>iç kısmının dışında</em> "
                        f"(duvar yakını) geçirdiği zamanın yüzdesidir. Yüksek değer "
                        f"kaçınma/anksiyete; düşük değer cesur ve merkezi keşfi "
                        f"işaret eder."
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown('<div style="height:1.25rem;"></div>', unsafe_allow_html=True)
    
            # ── Isı Haritası (eski "KDE Aktivite Haritası") ──────────────────────
            if "heatmap_kde" in available_outputs:
                st.markdown(html_section("Isı Haritası"), unsafe_allow_html=True)
                _, cimg, _ = st.columns([1, 3, 1])
                with cimg:
                    st.image(analysis_images["heatmap_kde"], use_container_width=True)
                st.markdown('<div style="height:1.25rem;"></div>', unsafe_allow_html=True)
    
            # ── Davranış Bout Tablosu ────────────────────────────────────────────
            bouts_path = analysis_results.get("behavior_bouts")
            if bouts_path and bouts_path.exists():
                st.markdown(html_section("Davranış Bout Tablosu"), unsafe_allow_html=True)
                try:
                    bouts_data = pd.read_csv(bouts_path)
                    st.dataframe(
                        bouts_data.head(20),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(f"Toplam bout sayısı: {len(bouts_data)}")
                except Exception as e:
                    st.warning(f"Bout dosyası okunamadı: {e}")
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 3 — METRİKLER
    # ─────────────────────────────────────────────────────────────────────────────
    
    with tab_metrics:
        # ── Üst satır: MODEL ÖZELLİKLERİ — tam genişlik ──────────────────────────
        st.markdown(html_section("Model Özellikleri"), unsafe_allow_html=True)
        model_cols = pred_info.get("feature_cols", []) or list(feat.keys())
        feat_cols = st.columns(2, gap="medium")
        for i, k in enumerate(model_cols):
            info = FEATURE_INFO.get(k, {})
            name = info.get("name", k.replace("_", " ").title())
            val = feat.get(k)
            if isinstance(val, float):
                val_str = "—" if np.isnan(val) else f"{val:.3f}"
            elif val is None:
                val_str = "—"
            else:
                val_str = str(val)
    
            with feat_cols[i % 2]:
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
    
        # ── Alt satır: DAVRANIŞ ÖZETLERİ  ⟂  MEKANSAL REARING ────────────────────
        mc1, mc2 = st.columns(2, gap="medium")
    
        with mc1:
            st.markdown(html_section("Davranış Özetleri"), unsafe_allow_html=True)
    
            beh_cards = [
                ("Rearing",  "#10B981",
                 int(feat['rear_count']),
                 feat.get('rear_total_s', 0.0) or 0.0,
                 feat.get('rear_pct', 0.0) or 0.0),
                ("Grooming", "#A855F7",
                 int(feat['groom_count']),
                 feat.get('groom_total_s', 0.0) or 0.0,
                 feat.get('groom_pct', 0.0) or 0.0),
                ("Freezing", "#F59E0B",
                 int(feat.get('freeze_bout_count', 0) or 0),
                 None,
                 feat.get('pct_freeze', 0.0) or 0.0),
            ]
    
            cards_html = ""
            for name, color, count, total_s, pct in beh_cards:
                total_str = "—" if total_s is None else f"{total_s:.1f} sn"
                pct_str = f"%{pct:.1f}"
                cards_html += (
                    f"<div style='background:rgba(255,255,255,0.03);"
                    f"border:1px solid rgba(255,255,255,0.07);border-radius:12px;"
                    f"padding:0.85rem 1rem;margin-bottom:8px;'>"
                    f"  <div style='display:flex;justify-content:space-between;"
                    f"align-items:center;'>"
                    f"    <div style='display:flex;align-items:center;gap:10px;'>"
                    f"      <div style='width:10px;height:10px;border-radius:50%;"
                    f"background:{color};box-shadow:0 0 8px {color}66;'></div>"
                    f"      <span style='color:#e2e8f0;font-weight:600;"
                    f"font-size:0.95rem;'>{name}</span>"
                    f"    </div>"
                    f"    <div style='display:flex;gap:18px;font-size:0.85rem;'>"
                    f"      <span style='color:#94a3b8;'>Adet: "
                    f"<strong style='color:#f1f5f9;font-weight:700;'>{count}</strong></span>"
                    f"      <span style='color:#94a3b8;'>Süre: "
                    f"<strong style='color:#f1f5f9;font-weight:700;'>{total_str}</strong></span>"
                    f"      <span style='color:#94a3b8;'>%: "
                    f"<strong style='color:{color};font-weight:700;'>{pct_str}</strong></span>"
                    f"    </div>"
                    f"  </div>"
                    f"</div>"
                )
            st.markdown(cards_html, unsafe_allow_html=True)
    
        with mc2:
            # Başlık literal-uppercase — tr_upper'ın "Rearing"'i "REARİNG" yapmasını
            # engellemek için doğrudan büyük harf string verilir.
            st.markdown(html_section("MEKANSAL REARING"), unsafe_allow_html=True)
            rc = int(feat.get('rear_count_center', 0) or 0)
            rw = int(feat.get('rear_count_wall', 0) or 0)
            rcf = feat.get('rear_center_frac')
            rcf_str = "—" if (rcf is None or (isinstance(rcf, float) and np.isnan(rcf))) else f"{rcf:.1%}"
            c1, c2, c3 = st.columns(3)
            c1.metric("Merkez",        rc)
            c2.metric("Duvar",         rw)
            c3.metric("Merkez Oranı",  rcf_str)
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 4 — İNDİR
    # ─────────────────────────────────────────────────────────────────────────────
    
    with tab_dl:
        st.markdown(html_section("Rapor Dosyaları"), unsafe_allow_html=True)
    
        dc1, dc2, dc3 = st.columns(3, gap="medium")
    
        with dc1:
            st.download_button(
                "⬇  Rapor (TXT)", report_text,
                file_name=txt_name, mime="text/plain",
                use_container_width=True, key="dl_txt",
            )
            with st.expander("Önizleme"):
                st.code(report_text, language="text")
    
        with dc2:
            st.download_button(
                "⬇  Tahmin Raporu (JSON)", json_bytes,
                file_name=json_name, mime="application/json",
                use_container_width=True, key="dl_json",
            )
            with st.expander("Önizleme"):
                st.json(json.loads(json_bytes.decode("utf-8")), expanded=False)
    
        with dc3:
            st.download_button(
                "⬇  Genel Bakış (PNG)", fig_images["anxiety_overview"],
                file_name=f"{subject}_anxiety_v2_overview.png",
                mime="image/png",
                use_container_width=True, key="dl_overview",
            )
            with st.expander("Önizleme"):
                st.image(fig_images["anxiety_overview"], use_container_width=True)
    
        if available_outputs:
            st.markdown(html_section("Analiz Görselleri"), unsafe_allow_html=True)
            viz_map = [
                ("behavior_timeline", "Zaman Çizgisi"),
                ("orbit_grid",        "Yörünge"),
                ("thigmotaxis",       "Thigmotaxis"),
                ("heatmap_kde",       "Isı Haritası"),
                ("heatmap_histogram", "Histogram"),
                ("bodypart_heatmaps", "Bodypart Izgarası"),
            ]
            viz_map = [(k, lbl) for k, lbl in viz_map if k in available_outputs]
            for i in range(0, len(viz_map), 3):
                row = viz_map[i:i+3]
                cols = st.columns(len(row), gap="medium")
                for (key, lbl), col in zip(row, cols):
                    with col:
                        st.download_button(
                            f"⬇  {lbl} (PNG)", analysis_images[key],
                            file_name=f"{subject}_{key}.png",
                            mime="image/png",
                            use_container_width=True, key=f"dl_{key}",
                        )
                        with st.expander("Önizleme"):
                            st.image(analysis_images[key], use_container_width=True)


_render_tab_section()

