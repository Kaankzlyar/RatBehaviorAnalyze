# -*- coding: utf-8 -*-
"""
plus_maze_app.py
----------------
Plus Maze (EPM) analiz ve tahmin dashboard'u.

Çalıştırmak için:
    cd webapp
    streamlit run plus_maze_app.py
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── Kol koordinatları (arm_coords.json'dan ya da varsayılan) ──────────────────
_arm_json = ROOT / "data" / "arm_coords.json"
if _arm_json.exists():
    with open(_arm_json, encoding="utf-8") as _f:
        _raw = json.load(_f)
    DEFAULT_ARMS: dict[str, tuple] = {k: tuple(v) for k, v in _raw.items()}
else:
    DEFAULT_ARMS = {
        "bottom_arm": (547, 603, 402, 713),
        "left_arm":   (260, 552, 347, 403),
        "right_arm":  (602, 894, 345, 402),
        "top_arm":    (546, 604,  33, 347),
    }

ZONE_COLORS = {
    "bottom_arm": "#4FC3F7",
    "left_arm":   "#66BB6A",
    "right_arm":  "#FFA726",
    "top_arm":    "#BA68C8",
}

ARM_LABELS = {
    "bottom_arm": "Alt (kapalı)",
    "left_arm":   "Sol (açık)",
    "right_arm":  "Sağ (açık)",
    "top_arm":    "Üst (kapalı)",
}

FPS = 30.0

# ── Sayfa ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EPM Analiz & Tahmin",
    page_icon="🐀",
    layout="wide",
)
st.title("🐀 Plus Maze (EPM) Analiz & Tahmin")
st.caption(
    "DLC CSV yükle → EPM metrikleri → Kontrol / Tedavi tahmini + görselleştirme. "
    "Model: Lojistik Regresyon, 37 sıçan LOOCV AUC=0.733, F1(Tedavi)=0.821."
)


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


# ── Model yükleme (pkl başarısız olursa anında eğit) ─────────────────────────

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
    sp = MODEL_DIR / "scaler_epm.pkl"
    mp = MODEL_DIR / "lr_epm.pkl"
    if sp.exists() and mp.exists():
        try:
            with open(sp, "rb") as f:
                bundle = pickle.load(f)
            with open(mp, "rb") as f:
                clf = pickle.load(f)
            return bundle, clf
        except Exception:
            pass  # numpy/sklearn sürüm uyumsuzluğu → anında eğit
    return _train_from_csv()


# ── Tahmin ────────────────────────────────────────────────────────────────────

def predict(row: dict, bundle: dict, clf) -> dict:
    feats   = bundle["features"]
    x       = np.array([[row.get(c, np.nan) for c in feats]], dtype=float)
    x_i     = bundle["imputer"].transform(x)
    x_s     = bundle["scaler"].transform(x_i)
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


# ── Genel bakış grafiği ───────────────────────────────────────────────────────

def fig_overview(body_x, body_y, arms: dict, pred_info: dict) -> plt.Figure:
    pred_color = "#E53935" if pred_info["pred"] == 1 else "#1E88E5"
    pred_label = "Tedavi" if pred_info["pred"] == 1 else "Kontrol"

    # Eksen sınırlarını kollardan otomatik hesapla
    all_x = [v for (x0, x1, y0, y1) in arms.values() for v in (x0, x1)]
    all_y = [v for (x0, x1, y0, y1) in arms.values() for v in (y0, y1)]
    margin = 25
    xmin_b, xmax_b = min(all_x) - margin, max(all_x) + margin
    ymin_b, ymax_b = min(all_y) - margin, max(all_y) + margin

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin_b, xmax_b)
    ax.set_ylim(ymax_b, ymin_b)   # y ekseni ters (piksel koordinatı)

    # Kol dikdörtgenleri — dolu dolgu + belirgin kenar
    for arm_key, (x0, x1, y0, y1) in arms.items():
        c = ZONE_COLORS[arm_key]
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.5, edgecolor=c,
            facecolor=c, alpha=0.30, zorder=1,
        )
        ax.add_patch(rect)
        # Etiket — beyaz arka planlı metin
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2,
            ARM_LABELS[arm_key],
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=3,
            bbox=dict(facecolor=c, alpha=0.75,
                      edgecolor="none", boxstyle="round,pad=0.3"),
        )

    # Trajectory
    valid = ~(np.isnan(body_x) | np.isnan(body_y))
    if valid.any():
        ax.plot(body_x[valid], body_y[valid],
                color=pred_color, alpha=0.55, lw=0.9,
                zorder=2, label="body_center")
        ax.scatter(body_x[valid][0],  body_y[valid][0],
                   color="#43A047", s=90, zorder=5, label="Başlangıç")
        ax.scatter(body_x[valid][-1], body_y[valid][-1],
                   color="#E53935", s=90, zorder=5, marker="X", label="Bitiş")

    ax.set_title(
        f"{pred_label}  —  P(Tedavi) = {pred_info['proba_treated']:.2f}",
        fontsize=13, fontweight="bold", color=pred_color, pad=10,
    )
    ax.set_xlabel("x (piksel)", fontsize=10)
    ax.set_ylabel("y (piksel)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ── Trajectory + Heatmap görselleri (subprocess) ─────────────────────────────

def run_visuals(csv_path: Path, arms: dict, out_dir: Path) -> dict:
    arm_args = []
    for key in ("bottom_arm", "left_arm", "right_arm", "top_arm"):
        arm_args += [f"--{key.replace('_', '-')}"] + [str(v) for v in arms[key]]

    common = [
        "--csv",        str(csv_path),
        "--out-dir",    str(out_dir),
        "--likelihood", "0.6",
        "--jump-thresh", "60",
        "--smooth",     "5",
        "--fps",        str(FPS),
    ] + arm_args

    for script in ("orbit_plot.py", "activity_heatmap.py"):
        try:
            subprocess.run(
                [sys.executable, str(PLUS_DIR / script)] + common,
                capture_output=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            pass

    images = {}
    stem = csv_path.stem
    key_map = {
        "orbit.png":             "orbit",
        "bodyparts.png":         "bodyparts",
        "heatmap_kde.png":       "heatmap_kde",
        "heatmap_histogram.png": "heatmap_histogram",
    }
    for f in out_dir.iterdir():
        if f.name.startswith(stem) and f.suffix == ".png":
            for suffix, key in key_map.items():
                if f.name.endswith(suffix):
                    images[key] = f.read_bytes()
    return images


# ── Sidebar — sadece dosya yükleme bilgisi ────────────────────────────────────

with st.sidebar:
    st.header("Hakkında")
    st.markdown(
        "**Model:** Lojistik Regresyon  \n"
        "**Eğitim:** 37 sıçan (LOOCV)  \n"
        "**AUC:** 0.733  \n"
        "**F1 (Tedavi):** 0.821  \n\n"
        "**Kol koordinatları** `data/arm_coords.json` dosyasından otomatik yüklenir."
    )
    st.divider()
    st.markdown(
        "**Kullanım:**  \n"
        "1. CSV dosyasını yükle  \n"
        "2. **Çalıştır** butonuna bas  \n"
        "3. Sonuçları incele ve indir"
    )


# ── Dosya yükleme ─────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "DLC filtered pose CSV (tek sıçan)", type=["csv"]
)
if uploaded is None:
    st.info("CSV dosyası yükleyin, ardından **Çalıştır** butonuna basın.")
    st.stop()

if not st.button("▶  Çalıştır", type="primary"):
    st.stop()

# ── Model kontrol ─────────────────────────────────────────────────────────────

bundle, clf = load_model()
if bundle is None:
    st.error(
        "Model yüklenemedi ve eğitim verisi de bulunamadı.  \n"
        "`data/plus_maze_metrics_all.csv` dosyasının varlığını kontrol edin."
    )
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    csv_path = tmp_path / uploaded.name
    csv_path.write_bytes(uploaded.getvalue())

    progress = st.progress(0.0, "Başlıyor…")

    try:
        progress.progress(0.15, "[1/4] EPM metrikleri hesaplanıyor…")
        row = compute_metrics(str(csv_path), DEFAULT_ARMS, fps=FPS)
        row = add_derived(row)

        progress.progress(0.35, "[2/4] Trajectory okunuyor…")
        body_x, body_y = load_body_center(
            str(csv_path), likelihood_thresh=0.6, jump_thresh=60.0, smooth=5
        )

        progress.progress(0.55, "[3/4] Model tahmini…")
        pred_info = predict(row, bundle, clf)

        progress.progress(0.70, "[4/4] Görseller üretiliyor…")
        images = run_visuals(csv_path, DEFAULT_ARMS, tmp_path)

        # Overview grafiğini belleğe al (temp dir kapanmadan önce)
        fig_ov = fig_overview(body_x, body_y, DEFAULT_ARMS, pred_info)
        _buf = io.BytesIO()
        fig_ov.savefig(_buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig_ov)
        overview_bytes = _buf.getvalue()

        progress.progress(1.0, "Tamamlandı ✓")

    except Exception as e:
        st.error(f"Pipeline hatası: {type(e).__name__}: {e}")
        st.exception(e)
        st.stop()


# ── Görüntü ───────────────────────────────────────────────────────────────────

pred_label = "Tedavi" if pred_info["pred"] == 1 else "Kontrol"
top_proba  = pred_info["proba_treated"] if pred_info["pred"] == 1 else pred_info["proba_control"]
subject    = Path(uploaded.name).stem

st.divider()

tab_pred, tab_metrics, tab_viz, tab_dl = st.tabs(
    ["EPM Tahmini", "EPM Metrikleri", "Görselleştirmeler", "İndir"]
)

# ── Tab 1: EPM Tahmini ────────────────────────────────────────────────────────

with tab_pred:
    col_img, col_meta = st.columns([1.3, 1])

    with col_img:
        st.subheader("Genel Bakış")
        st.image(overview_bytes, use_container_width=True)

    with col_meta:
        st.subheader("Tahmin Sonucu")
        color_md = "red" if pred_info["pred"] == 1 else "blue"
        st.markdown(f"## :{color_md}[{pred_label}]")
        st.metric("Güven", f"%{top_proba * 100:.0f}")

        c1, c2 = st.columns(2)
        c1.metric("P(Kontrol)", f"{pred_info['proba_control']:.3f}")
        c2.metric("P(Tedavi)",  f"{pred_info['proba_treated']:.3f}")

        st.subheader("Temel EPM Metrikleri")
        oa  = row["pct_open_arm"]
        ai  = row["anxiety_index_epm"]
        te  = row["total_entries"]
        oae = row.get("pct_open_arm_entries", 0) or 0
        alt = row.get("successive_alternation_pct", 0) or 0
        st.markdown(
            f"- Açık kol süresi: **%{oa:.1f}**\n"
            f"- Anksiyete indeksi: **{ai:.2f}**\n"
            f"- Toplam giriş: **{te}**\n"
            f"- Açık kol girişi: **%{oae:.1f}**\n"
            f"- Alternasyon: **%{alt:.1f}**"
        )

        st.subheader("Kararı En Çok Etkileyen Özellikler")
        for i, c in enumerate(pred_info["top_contributors"][:3], 1):
            sign  = "+" if c["push"] >= 0 else "−"
            val_s = f"{c['value']:.2f}" if c["value"] is not None else "—"
            st.markdown(
                f"{i}. **{c['feature']}** = {val_s}  "
                f"(z={c['z']:+.2f}, etki={sign}{abs(c['push']):.2f}) "
                f"→ {c['toward']}"
            )


# ── Tab 2: EPM Metrikleri ─────────────────────────────────────────────────────

with tab_metrics:
    st.subheader("Tüm EPM Metrikleri")

    highlight = [
        "pct_open_arm", "anxiety_index_epm", "pct_open_arm_entries",
        "total_entries", "successive_alternation_pct", "perseveration_rate_pct",
        "mean_speed_px_s", "arm_preference_index", "pct_time_junction",
    ]
    metric_rows = [
        {
            "Metrik": k.replace("_", " ").title(),
            "Değer":  f"{v:.2f}" if isinstance(v, float) else str(v),
        }
        for k, v in row.items() if k in highlight
    ]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Kol Bazlı Zaman Dağılımı")
    arm_tbl = {
        "Kol": ["Sol (açık)", "Sağ (açık)", "Alt (kapalı)", "Üst (kapalı)", "Kavşakta"],
        "Süre (%)": [
            row["pct_time_left"], row["pct_time_right"],
            row["pct_time_bottom"], row["pct_time_top"],
            row["pct_time_junction"],
        ],
        "Giriş": [
            row["left_entries"], row["right_entries"],
            row["bottom_entries"], row["top_entries"], "—",
        ],
    }
    st.dataframe(pd.DataFrame(arm_tbl), use_container_width=True, hide_index=True)

    with st.expander("Giriş dizisi"):
        seq = row.get("entry_sequence", "")
        st.code(seq if seq else "(giriş yok)")


# ── Tab 3: Görselleştirmeler ──────────────────────────────────────────────────

with tab_viz:
    if not images:
        st.warning("Görsel üretilemedi. orbit_plot.py veya activity_heatmap.py hatasını kontrol edin.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            if "orbit" in images:
                st.markdown("### Kol-Renkli Trajectory")
                st.image(images["orbit"], use_container_width=True)
            if "heatmap_kde" in images:
                st.markdown("### KDE Aktivite Haritası")
                st.image(images["heatmap_kde"], use_container_width=True)
        with col_b:
            if "bodyparts" in images:
                st.markdown("### Vücut Noktaları (10 keypoint)")
                st.image(images["bodyparts"], use_container_width=True)
            if "heatmap_histogram" in images:
                st.markdown("### Histogram Yoğunluk")
                st.image(images["heatmap_histogram"], use_container_width=True)


# ── Tab 4: İndir ──────────────────────────────────────────────────────────────

with tab_dl:
    st.subheader("Dosyaları İndir")

    safe_row = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in row.items()
        if not k.startswith("_")
    }
    safe_row["pred_label"]    = pred_label
    safe_row["proba_control"] = round(pred_info["proba_control"], 4)
    safe_row["proba_treated"] = round(pred_info["proba_treated"], 4)

    csv_bytes  = pd.DataFrame([safe_row]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
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

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇ Metrikler (CSV)",
        csv_bytes,
        file_name=f"{subject}_epm_metrics.csv",
        mime="text/csv",
    )
    c2.download_button(
        "⬇ Tahmin Raporu (JSON)",
        json_bytes,
        file_name=f"{subject}_epm_report.json",
        mime="application/json",
    )
    c3.download_button(
        "⬇ Genel Bakış (PNG)",
        overview_bytes,
        file_name=f"{subject}_epm_overview.png",
        mime="image/png",
    )

    if images:
        st.divider()
        st.subheader("Analiz Görselleri")
        viz_map = [
            ("orbit",             "Trajectory"),
            ("bodyparts",         "Vücut Noktaları"),
            ("heatmap_kde",       "KDE Haritası"),
            ("heatmap_histogram", "Histogram"),
        ]
        cols = st.columns(len(viz_map))
        for (key, lbl), col in zip(viz_map, cols):
            if key in images:
                col.download_button(
                    f"⬇ {lbl} (PNG)",
                    images[key],
                    file_name=f"{subject}_{key}.png",
                    mime="image/png",
                )
