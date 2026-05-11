"""
predict_anxiety_v3_gui.py
-------------------------
Streamlit sarmalayıcı: v2 anxiety pipeline'ını lokal web arayüzünden çalıştırır.
v2'nin saf fonksiyonlarını (predict_anxiety_v2) doğrudan import eder; CSV
geçici dizine yazılır, pipeline tamamen in-memory döner, çıktılar inline
gösterilir ve indirme butonları ile sunulur.

Kullanım:
    pip install streamlit
    streamlit run scripts/predict_anxiety_v3_gui.py

İstersen "Çıktıları reports/ altına da yaz" seçeneği ile v2 ile aynı yere
(`reports/anxiety_predictions_v2/`) yazdırabilirsin.
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predict_anxiety_v2 as v2  # noqa: E402


# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Anxiety Pipeline v3", page_icon="🐀", layout="wide")
st.title("🐀 Rat Anxiety Pipeline — v3 (GUI)")
st.caption(
    "DLC filtered pose CSV → rule-based bouts → OFT metrikleri → spatial rearing "
    "→ rear-only LogReg (AUC≈0.73 LOOCV) → Control / Treated tahmini. "
    "Pipeline detayı: `docs/anxiety_progress_2026-05-10.md`."
)


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Ayarlar")
    fps = st.number_input(
        "FPS", value=float(v2.DEFAULT_FPS), min_value=1.0, max_value=240.0, step=1.0,
    )
    tag = st.text_input(
        "Model tag", value="rearonly",
        help="`models/anxiety_classifier/{lr,scaler}_<tag>.pkl` dosyalarını yükler.",
    )
    save_to_reports = st.checkbox(
        f"Çıktıları `reports/anxiety_predictions_v2/` altına da yaz",
        value=False,
        help="Kapalıysa sadece bu oturumda indirme butonu üzerinden alabilirsin.",
    )

    st.divider()
    st.markdown(
        "**Not:** Model `Treated vs Control` ayırıyor. *Treated* = aspartam "
        "ve/veya grapefruit alan tüm hayvanlar; doğrudan 'anksiyeteli' değil. "
        "Yorum: *tedavi-kaynaklı davranışsal değişim sinyali*."
    )


# ── Input ───────────────────────────────────────────────────────────────────

uploaded = st.file_uploader("DLC filtered pose CSV", type=["csv"])

if uploaded is None:
    st.info("Sol panelden DLC CSV yükle, ardından **▶ Çalıştır** butonuna bas.")
    st.stop()

if not st.button("▶  Çalıştır", type="primary"):
    st.stop()


# ── Pipeline ────────────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    csv_path = tmp_path / uploaded.name
    csv_path.write_bytes(uploaded.getvalue())

    progress = st.progress(0.0, text="Başlıyor…")

    try:
        progress.progress(0.10, text="[1/4] DLC CSV → bouts (rearing / grooming)")
        bouts_df, n_frames = v2.detect_bouts(csv_path, fps=fps)

        progress.progress(0.35, text="[2/4] OFT metrikleri (locomotion / thigmotaxis / freeze / entropy)")
        oft, body_x, body_y = v2.compute_oft_metrics(csv_path, fps=fps)

        progress.progress(0.55, text="[3/4] Spatial rearing (center vs wall)")
        spatial = v2.compute_spatial_rearing(bouts_df, body_x, body_y)

        feat = v2.build_feature_dict(bouts_df, oft, spatial, oft["session_s"])

        progress.progress(0.75, text=f"[4/4] Model yükleme ve tahmin (tag={tag})")
        bundle, model = v2.load_model(tag)
        pred_info = v2.predict(feat, bundle, model)

        progress.progress(0.90, text="Çıktılar üretiliyor")

        subject = csv_path.stem
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Text report
        report_buf = StringIO()
        v2.render_summary(report_buf, csv_path, n_frames, feat, pred_info)
        report_text = report_buf.getvalue()
        txt_name = f"{subject}_anxiety_v2_report.txt"

        # PNG overview
        fig_path = out_dir / f"{subject}_anxiety_v2_overview.png"
        v2.plot_overview(csv_path, body_x, body_y, spatial, pred_info, fig_path)
        png_bytes = fig_path.read_bytes()
        png_name = fig_path.name

        # JSON payload (v2 main() ile aynı şema)
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
            (target / png_name).write_bytes(png_bytes)
            (target / txt_name).write_text(report_text, encoding="utf-8")
            (target / json_name).write_bytes(json_bytes)

        progress.progress(1.0, text="Bitti ✓")

    except FileNotFoundError as e:
        st.error("Model dosyaları bulunamadı.")
        st.code(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Pipeline hatası: {type(e).__name__}")
        st.exception(e)
        st.stop()


# ── Display ─────────────────────────────────────────────────────────────────

pred_label = v2.label_str(pred_info["pred"])
icon       = "🟧" if pred_info["pred"] == 1 else "🟦"
top_proba  = pred_info["proba_treated"] if pred_info["pred"] == 1 else pred_info["proba_control"]

st.divider()

col_img, col_meta = st.columns([1.4, 1])

with col_img:
    st.subheader("Overview")
    st.image(png_bytes, use_container_width=True)

with col_meta:
    st.subheader("Tahmin")
    st.markdown(f"### {icon}  {pred_label}")
    st.metric("Güven", f"%{top_proba * 100:.0f}")
    c1, c2 = st.columns(2)
    c1.metric("P(Control)", f"{pred_info['proba_control']:.3f}")
    c2.metric("P(Treated)", f"{pred_info['proba_treated']:.3f}")

    st.subheader("Anahtar metrikler")
    st.markdown(
        f"- Merkezde süre: **%{feat['pct_center']:.1f}**\n"
        f"- Duvar kenarında: **%{feat['pct_periphery']:.1f}**\n"
        f"- Donakalma: **%{feat['pct_freeze']:.1f}**\n"
        f"- Rearing toplam: **{int(feat['rear_count'])}** bout "
        f"({feat['rear_total_s']:.1f} s)\n"
        f"- Rearing — center: **{int(feat['rear_count_center'])}**, "
        f"wall: **{int(feat['rear_count_wall'])}**"
    )

st.subheader("Rapor (TR)")
st.code(report_text, language="text")

st.subheader("İndirme")
d1, d2, d3 = st.columns(3)
d1.download_button("📷  overview.png", png_bytes, file_name=png_name, mime="image/png")
d2.download_button("📝  report.txt", report_text, file_name=txt_name, mime="text/plain")
d3.download_button("📊  report.json", json_bytes, file_name=json_name, mime="application/json")

if save_to_reports:
    st.success(
        f"Çıktılar `{v2.DEFAULT_OUT.relative_to(ROOT)}/` altına da yazıldı."
    )
