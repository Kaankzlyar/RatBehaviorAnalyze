# -*- coding: utf-8 -*-
"""
plus_maze_app.py
----------------
Plus Maze (EPM) analiz ve tahmin dashboard'u.

Pipeline:
  DLC CSV yukle
    -> tmaze_metrics.py (compute_metrics) -> EPM metrikleri
    -> models/epm_classifier/lr_epm.pkl   -> Control / Treated tahmini
    -> orbit_plot.py  (subprocess)        -> trajectory PNG
    -> activity_heatmap.py (subprocess)   -> heatmap PNG

Calistirmak icin:
    cd webapp
    streamlit run plus_maze_app.py
"""
from __future__ import annotations

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
ROOT     = Path(__file__).resolve().parent.parent
PLUS_DIR = ROOT / "analysis" / "plus_maze"
MODEL_DIR = ROOT / "models" / "epm_classifier"
sys.path.insert(0, str(PLUS_DIR))

from tmaze_metrics import compute_metrics, load_body_center  # noqa: E402

# ── Varsayilan kol koordinatlari ──────────────────────────────────────────────
_arm_json = ROOT / "data" / "arm_coords.json"
if _arm_json.exists():
    with open(_arm_json) as _f:
        _raw = json.load(_f)
    DEFAULT_ARMS = {k: tuple(v) for k, v in _raw.items()}
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
    "junction":   "#888888",
}

ARM_LABELS = {
    "bottom_arm": "Alt (kapali)",
    "left_arm":   "Sol (acik)",
    "right_arm":  "Sag (acik)",
    "top_arm":    "Ust (kapali)",
}

# ── Sayfa ayarlari ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EPM Analiz & Tahmin",
    page_icon="🐀",
    layout="wide",
)
st.title("🐀 Plus Maze (EPM) Analiz & Tahmin")
st.caption(
    "DLC CSV yukle → EPM metrikleri → Control / Treated tahmini + gorsellestirme. "
    "Model: LogisticRegression, 37 subject LOOCV AUC=0.733, F1(Treated)=0.821."
)


# ── Yardimci: EPM turetilmis metrikler ───────────────────────────────────────

def add_derived(row: dict) -> dict:
    """batch_metrics.py ile ayni formul."""
    total = row["total_entries"] or float("nan")
    row["pct_open_arm"]  = row["pct_time_left"] + row["pct_time_right"]
    row["pct_closed_arm"] = row["pct_time_top"] + row["pct_time_bottom"]
    oe = (row["left_entries"] + row["right_entries"]) / total * 100 if total else float("nan")
    row["pct_open_arm_entries"] = oe
    row["pct_closed_arm_entries"] = (row["top_entries"] + row["bottom_entries"]) / total * 100 if total else float("nan")
    row["anxiety_index_epm"] = (row["pct_open_arm"] + (oe if not np.isnan(oe) else 0)) / 2
    return row


# ── Yardimci: model yukle ────────────────────────────────────────────────────

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


def _train_from_csv() -> tuple[dict, object] | tuple[None, None]:
    """pkl yuklenemezse aninda egit (37 subject, <1s)."""
    data_csv = ROOT / "data" / "plus_maze_metrics_all.csv"
    if not data_csv.exists():
        return None, None
    df = pd.read_csv(data_csv)
    df["label"] = (df["cohort"] != "Control").astype(int)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return None, None
    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values
    imp = SimpleImputer(strategy="median")
    X_i = imp.fit_transform(X)
    sc  = StandardScaler()
    X_s = sc.fit_transform(X_i)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_s, y)
    bundle = {"imputer": imp, "scaler": sc, "features": FEATURE_COLS}
    return bundle, clf


@st.cache_resource
def load_model():
    scaler_path = MODEL_DIR / "scaler_epm.pkl"
    model_path  = MODEL_DIR / "lr_epm.pkl"
    if scaler_path.exists() and model_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                bundle = pickle.load(f)
            with open(model_path, "rb") as f:
                clf = pickle.load(f)
            return bundle, clf
        except Exception:
            pass  # numpy/sklearn surum uyumsuzlugu — aninda egit
    return _train_from_csv()


# ── Yardimci: tahmin ─────────────────────────────────────────────────────────

def predict(row: dict, bundle: dict, clf) -> dict:
    feats = bundle["features"]
    x = np.array([[row.get(c, np.nan) for c in feats]], dtype=float)
    x_i = bundle["imputer"].transform(x)
    x_s = bundle["scaler"].transform(x_i)
    pred  = int(clf.predict(x_s)[0])
    proba = clf.predict_proba(x_s)[0]

    coef    = clf.coef_.ravel()
    contrib = coef * x_s[0]
    order   = np.argsort(-np.abs(contrib))
    top = [
        {
            "feature": feats[i],
            "value":   float(x[0, i]) if not np.isnan(x[0, i]) else None,
            "z":       round(float(x_s[0, i]), 2),
            "push":    round(float(contrib[i]), 3),
            "toward":  "Treated" if contrib[i] > 0 else "Control",
        }
        for i in order[:5]
    ]
    return {
        "pred":           pred,
        "proba_control":  float(proba[0]),
        "proba_treated":  float(proba[1]),
        "top_contributors": top,
        "feature_cols":   feats,
    }


# ── Yardimci: genel bakis grafigi ────────────────────────────────────────────

def fig_overview(body_x, body_y, arms: dict, pred_info: dict) -> plt.Figure:
    color = "#E53935" if pred_info["pred"] == 1 else "#1E88E5"
    label = "Treated" if pred_info["pred"] == 1 else "Control"

    fig, ax = plt.subplots(figsize=(7, 7))

    # Kol dikdortgenleri
    for arm, (x0, x1, y0, y1) in arms.items():
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=2",
            linewidth=1.5, edgecolor=ZONE_COLORS[arm],
            facecolor=ZONE_COLORS[arm], alpha=0.18,
        )
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2,
                ARM_LABELS[arm], ha="center", va="center",
                fontsize=8, color=ZONE_COLORS[arm], fontweight="bold")

    # Yolu ciz
    valid = ~(np.isnan(body_x) | np.isnan(body_y))
    ax.plot(body_x[valid], body_y[valid],
            color=color, alpha=0.4, lw=0.8, label="body_center izi")
    if valid.any():
        ax.scatter(body_x[valid][0],  body_y[valid][0],
                   color="green", s=80, zorder=5, label="Baslangic")
        ax.scatter(body_x[valid][-1], body_y[valid][-1],
                   color="red",   s=80, zorder=5, marker="X", label="Bitis")

    ax.set_title(
        f"{label}  (P_treated={pred_info['proba_treated']:.2f})",
        fontsize=14, fontweight="bold",
        color="#E53935" if pred_info["pred"] == 1 else "#1E88E5",
    )
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ── Yardimci: subprocess ile gorsel uret ─────────────────────────────────────

def run_visuals(csv_path: Path, arms: dict, out_dir: Path, fps: float) -> dict:
    arm_args = []
    for key in ("bottom_arm", "left_arm", "right_arm", "top_arm"):
        arm_args += [f"--{key.replace('_', '-')}"] + [str(v) for v in arms[key]]

    common = [
        "--csv",        str(csv_path),
        "--out-dir",    str(out_dir),
        "--likelihood", "0.6",
        "--jump-thresh","60",
        "--smooth",     "5",
        "--fps",        str(fps),
    ] + arm_args

    images = {}
    for script, keys in [
        ("orbit_plot.py",      ["orbit", "bodyparts"]),
        ("activity_heatmap.py",["heatmap_kde", "heatmap_histogram"]),
    ]:
        try:
            subprocess.run(
                [sys.executable, str(PLUS_DIR / script)] + common,
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            pass

    stem = csv_path.stem
    for f in out_dir.iterdir():
        name = f.name
        if name.startswith(stem) and name.endswith(".png"):
            key = None
            if "orbit.png" in name:
                key = "orbit"
            elif "bodyparts.png" in name:
                key = "bodyparts"
            elif "heatmap_kde.png" in name:
                key = "heatmap_kde"
            elif "heatmap_histogram.png" in name:
                key = "heatmap_histogram"
            if key:
                images[key] = f.read_bytes()
    return images


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Ayarlar")
    fps = st.number_input("FPS", value=30.0, min_value=1.0, max_value=240.0, step=1.0)

    st.subheader("Kol Koordinatlari (xmin xmax ymin ymax)")
    arms: dict[str, tuple] = {}
    for arm_key, arm_label in ARM_LABELS.items():
        d = DEFAULT_ARMS.get(arm_key, (0, 100, 0, 100))
        st.markdown(f"**{arm_label}**")
        c1, c2, c3, c4 = st.columns(4)
        xmin = c1.number_input("xmin", value=int(d[0]), key=f"{arm_key}_x0", label_visibility="collapsed")
        xmax = c2.number_input("xmax", value=int(d[1]), key=f"{arm_key}_x1", label_visibility="collapsed")
        ymin = c3.number_input("ymin", value=int(d[2]), key=f"{arm_key}_y0", label_visibility="collapsed")
        ymax = c4.number_input("ymax", value=int(d[3]), key=f"{arm_key}_y1", label_visibility="collapsed")
        arms[arm_key] = (xmin, xmax, ymin, ymax)

    st.divider()
    run_visuals_flag = st.checkbox(
        "Trajectory + Heatmap gorselleri uret",
        value=True,
        help="Orbit plot ve KDE heatmap cizer. ~10-15 saniye ek sure.",
    )

# ── Dosya yukleme ─────────────────────────────────────────────────────────────

uploaded = st.file_uploader("DLC filtered pose CSV (tek subject)", type=["csv"])
if uploaded is None:
    st.info(
        "Sol panelden kol koordinatlarini ayarla, ardindan CSV yukle ve "
        "**Calistir** butonuna bas."
    )
    st.stop()

if not st.button("Calistir", type="primary"):
    st.stop()

# ── Model kontrol ─────────────────────────────────────────────────────────────

bundle, clf = load_model()
if bundle is None:
    st.error(
        "Model yuklenemedi ve egitim verisi de bulunamadi.\n\n"
        "`data/plus_maze_metrics_all.csv` dosyasinin var oldugunu kontrol et."
    )
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    csv_path = tmp_path / uploaded.name
    csv_path.write_bytes(uploaded.getvalue())

    progress = st.progress(0.0, "Basliyor...")

    try:
        # 1. EPM metrikleri
        progress.progress(0.2, "[1/4] EPM metrikleri hesaplaniyor...")
        zones = {k: tuple(v) for k, v in arms.items()}
        row = compute_metrics(str(csv_path), zones, fps=fps)
        row = add_derived(row)

        # 2. body_center oku (overview grafigi icin)
        progress.progress(0.4, "[2/4] Trajectory okunuyor...")
        body_x, body_y = load_body_center(
            str(csv_path), likelihood_thresh=0.6, jump_thresh=60.0, smooth=5
        )

        # 3. Tahmin
        progress.progress(0.6, "[3/4] Model tahmini...")
        pred_info = predict(row, bundle, clf)

        # 4. Gorseller
        images = {}
        if run_visuals_flag:
            progress.progress(0.75, "[4/4] Trajectory + heatmap gorselleri uretiliyor...")
            images = run_visuals(csv_path, zones, tmp_path, fps)

        # Genel bakis grafigini bellekte sakla
        fig_ov = fig_overview(body_x, body_y, zones, pred_info)
        import io as _io
        _buf = _io.BytesIO()
        fig_ov.savefig(_buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig_ov)
        overview_bytes = _buf.getvalue()

        progress.progress(1.0, "Bitti!")

    except Exception as e:
        st.error(f"Pipeline hatasi: {type(e).__name__}: {e}")
        st.exception(e)
        st.stop()


# ── Gosterim ─────────────────────────────────────────────────────────────────

pred_label  = "Treated" if pred_info["pred"] == 1 else "Control"
top_proba   = pred_info["proba_treated"] if pred_info["pred"] == 1 else pred_info["proba_control"]
subject     = Path(uploaded.name).stem

st.divider()

tab_pred, tab_metrics, tab_viz, tab_dl = st.tabs(
    ["EPM Tahmini", "EPM Metrikleri", "Gorsellestirmeler", "Indir"]
)

# ── Tab 1: EPM Tahmini ────────────────────────────────────────────────────────

with tab_pred:
    col_img, col_meta = st.columns([1.3, 1])

    with col_img:
        st.subheader("Genel Bakis (Trajectory + Kol Haritasi)")
        st.image(overview_bytes, use_container_width=True)

    with col_meta:
        st.subheader("Tahmin Sonucu")
        color_md = "red" if pred_info["pred"] == 1 else "blue"
        st.markdown(f"## :{color_md}[{pred_label}]")
        st.metric("Guven", f"%{top_proba * 100:.0f}")

        c1, c2 = st.columns(2)
        c1.metric("P(Control)", f"{pred_info['proba_control']:.3f}")
        c2.metric("P(Treated)", f"{pred_info['proba_treated']:.3f}")

        st.subheader("Temel EPM Metrikleri")
        st.markdown(
            f"- Acik kol suresi: **%{row['pct_open_arm']:.1f}**\n"
            f"- Anksiyete indeksi: **{row['anxiety_index_epm']:.2f}**\n"
            f"- Toplam giris: **{row['total_entries']}**\n"
            f"- Acik kol girisi: **%{row.get('pct_open_arm_entries', 0):.1f}**\n"
            f"- Alternasyon: **%{row.get('successive_alternation_pct', 0):.1f}**"
        )

        st.subheader("Karari En Cok Etkileyen Ozellikler")
        for i, c in enumerate(pred_info["top_contributors"][:3], 1):
            sign  = "+" if c["push"] >= 0 else "-"
            val_s = f"{c['value']:.2f}" if c["value"] is not None else "NaN"
            st.markdown(
                f"{i}. **{c['feature']}** = {val_s}  "
                f"(z={c['z']:+.2f}, push={sign}{abs(c['push']):.2f}) "
                f"→ {c['toward']}"
            )


# ── Tab 2: EPM Metrikleri ─────────────────────────────────────────────────────

with tab_metrics:
    st.subheader("Tum EPM Metrikleri")

    # Onemli metrikler vurgula
    highlight_cols = ["pct_open_arm", "anxiety_index_epm",
                      "pct_open_arm_entries", "total_entries",
                      "successive_alternation_pct", "perseveration_rate_pct",
                      "mean_speed_px_s", "arm_preference_index",
                      "pct_time_junction"]
    metric_rows = [
        {"Metrik": k.replace("_", " ").title(), "Deger": (
            f"{v:.2f}" if isinstance(v, float) else str(v)
        )}
        for k, v in row.items()
        if k in highlight_cols
    ]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Kol Bazli Zaman Dagilimi (%)")
    arm_data = {
        "Kol":  ["Sol (acik)", "Sag (acik)", "Alt (kapali)", "Ust (kapali)", "Kavsakta"],
        "Sure (%)": [
            row["pct_time_left"], row["pct_time_right"],
            row["pct_time_bottom"], row["pct_time_top"],
            row["pct_time_junction"],
        ],
        "Giris": [
            row["left_entries"], row["right_entries"],
            row["bottom_entries"], row["top_entries"], "-",
        ],
    }
    st.dataframe(pd.DataFrame(arm_data), use_container_width=True, hide_index=True)

    with st.expander("Giri dizisi"):
        seq = row.get("entry_sequence", "")
        st.code(seq if seq else "(giri yok)")


# ── Tab 3: Gorsellestirmeler ──────────────────────────────────────────────────

with tab_viz:
    if not images:
        st.info(
            "Gorsel uretimi devre disi. Sol panelden "
            "'Trajectory + Heatmap gorselleri uret' secenegini aktif et."
        )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            if "orbit" in images:
                st.markdown("### Kol-Renkli Trajectory")
                st.image(images["orbit"], use_container_width=True)
            if "heatmap_kde" in images:
                st.markdown("### KDE Aktivite Heatmap")
                st.image(images["heatmap_kde"], use_container_width=True)
        with col_b:
            if "bodyparts" in images:
                st.markdown("### Vucut Noktalari (10 keypoint)")
                st.image(images["bodyparts"], use_container_width=True)
            if "heatmap_histogram" in images:
                st.markdown("### Histogram Yogunluk")
                st.image(images["heatmap_histogram"], use_container_width=True)


# ── Tab 4: Indir ──────────────────────────────────────────────────────────────

with tab_dl:
    st.subheader("Dosyalari Indir")

    # Ozet CSV
    safe_row = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in row.items()
                if not k.startswith("_")}
    safe_row["pred_label"]   = pred_label
    safe_row["proba_control"] = round(pred_info["proba_control"], 4)
    safe_row["proba_treated"] = round(pred_info["proba_treated"], 4)
    csv_bytes = pd.DataFrame([safe_row]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    # JSON rapor
    json_payload = {
        "subject":          subject,
        "csv_file":         uploaded.name,
        "pred_label":       pred_label,
        "proba_control":    round(pred_info["proba_control"], 4),
        "proba_treated":    round(pred_info["proba_treated"], 4),
        "top_contributors": pred_info["top_contributors"],
        "features_used":    pred_info["feature_cols"],
        "metrics":          safe_row,
    }
    json_bytes = json.dumps(json_payload, indent=2, ensure_ascii=False).encode("utf-8")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Metrikleri CSV olarak indir",
            csv_bytes,
            file_name=f"{subject}_epm_metrics.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Tahmin Raporu (JSON)",
            json_bytes,
            file_name=f"{subject}_epm_report.json",
            mime="application/json",
        )
    with c3:
        st.download_button(
            "Genel Bakis (PNG)",
            overview_bytes,
            file_name=f"{subject}_epm_overview.png",
            mime="image/png",
        )

    if images:
        st.divider()
        st.subheader("Analiz Gorselleri")
        viz_map = [
            ("orbit",              "Trajectory"),
            ("bodyparts",          "Vucut Noktalari"),
            ("heatmap_kde",        "KDE Heatmap"),
            ("heatmap_histogram",  "Histogram"),
        ]
        cols = st.columns(4)
        for (key, label), col in zip(viz_map, cols):
            if key in images:
                col.download_button(
                    f"{label} (PNG)",
                    images[key],
                    file_name=f"{subject}_{key}.png",
                    mime="image/png",
                )
