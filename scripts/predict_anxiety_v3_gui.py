"""
predict_anxiety_v3_gui.py
-------------------------
Comprehensive Streamlit dashboard: integrates full open-field analysis (run_analysis.py)
with anxiety prediction pipeline (predict_anxiety_v2).

Pipeline outputs:
  1. Behavior detection (rearing/grooming bouts, timeline)
  2. Trajectory visualization (orbit plots, thigmotaxis)
  3. Activity heatmap (KDE heatmap — primary analysis)
  4. Per-bodypart heatmap grid
  5. Anxiety prediction (Treated/Control classification)

Kullanım:
    pip install streamlit
    streamlit run scripts/predict_anxiety_v3_gui.py

İstersen "Çıktıları reports/ altına da yaz" seçeneği ile tüm çıktılar
reports/anxiety_predictions_v2/ altına kaydedilir.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predict_anxiety_v2 as v2  # noqa: E402


# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Deney Hayvanı Davranış Analizi", page_icon="🐀", layout="wide")
st.title("🐀 Deney Hayvanı Davranış Analizi")
st.caption(
    "Complete pipeline: DLC CSV → behavior detection → open-field metrics → "
    "trajectory/heatmap visualization → anxiety prediction (Treated/Control). "
    "Details: `docs/anxiety_progress_2026-05-10.md`."
)


# ── Helper Functions ────────────────────────────────────────────────────────

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
    """Run open-field analysis pipeline (behavior + trajectories + heatmaps).
    
    Returns dict with paths to generated visualization files.
    Outputs are saved to the CSV's directory.
    """
    results = {
        "behavior_bouts": None,
        "behavior_frames": None,
        "behavior_timeline": None,
        "orbit_grid": None,
        "thigmotaxis": None,
        "heatmap_kde": None,
        "heatmap_histogram": None,
        "bodypart_heatmaps": None,
    }
    
    try:
        # Build common args for run_analysis.py
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

        if skip_behavior:
            common_args.append("--skip-behavior")
        if skip_orbit:
            common_args.append("--skip-orbit")
        if skip_heatmap:
            common_args.append("--skip-heatmap")
        if skip_bodypart:
            common_args.append("--skip-bodypart")
        
        cmd = [sys.executable] + common_args
        
        # Stream output for real-time progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT / "analysis" / "open_field"),
        )
        
        # Read output line by line
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.rstrip())
            if progress_container:
                # Update with current step
                if "[*]" in line or "[OK]" in line or "Step" in line:
                    progress_container.write(f"🔄 {line.rstrip()}")
        
        stdout, stderr = process.communicate(timeout=600)  # 10 min timeout
        
        if process.returncode != 0:
            raise RuntimeError(f"Analysis failed: {stderr}")
        
        # Scan CSV directory for generated files
        csv_dir = csv_path.parent
        stem = csv_path.stem
        for file in csv_dir.glob(f"{stem}*"):
            if "_behavior_bouts.csv" in file.name:
                results["behavior_bouts"] = file
            elif "_behavior_frames.csv" in file.name:
                results["behavior_frames"] = file
            elif "_behavior_timeline.png" in file.name:
                results["behavior_timeline"] = file
            elif "_orbit_grid.png" in file.name:
                results["orbit_grid"] = file
            elif "_thigmotaxis.png" in file.name:
                results["thigmotaxis"] = file
            elif "_heatmap_kde.png" in file.name:
                results["heatmap_kde"] = file
            elif "_heatmap_histogram.png" in file.name:
                results["heatmap_histogram"] = file
            elif "_bodypart_heatmaps.png" in file.name:
                results["bodypart_heatmaps"] = file
        
        return results
    except subprocess.TimeoutExpired:
        raise RuntimeError("Open-field analysis timed out (> 10 min)")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Analysis pipeline failed: {e.stderr}")


def load_image_bytes(path: Path) -> bytes:
    """Safely load image file."""
    if path and path.exists():
        return path.read_bytes()
    return None


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Ayarlar")
    
    st.subheader("Aspartam veya Greyfurt Suyu alıp almadığının Tahmini")
    fps = st.number_input(
        "FPS", value=float(v2.DEFAULT_FPS), min_value=1.0, max_value=240.0, step=1.0,
    )
    tag = st.text_input(
        "Model tag", value="rearonly",
        help="`models/anxiety_classifier/{lr,scaler}_<tag>.pkl` dosyalarını yükler.",
    )
    
    st.divider()
    
    st.subheader("Open-Field Analizi")
    run_analysis = st.checkbox(
        "Detaylı Analizi Çalıştır",
        value=False,
        help="Açılırsa, DLC CSV'ye dayalı kapsamlı bir açık alan analizi yapılır: davranış tespiti, yörünge görselleştirme, ısı haritaları vb. (Çok zaman alabilir!)",
    )
    
    if run_analysis:
        st.markdown("**Arena Sınırları**")
        
        col1, col2 = st.columns(2)
        with col1:
            arena_xmin = st.number_input("X min", value=397, step=1)
            arena_ymin = st.number_input("Y min", value=156, step=1)
        with col2:
            arena_xmax = st.number_input("X max", value=777, step=1)
            arena_ymax = st.number_input("Y max", value=535, step=1)
        
        st.markdown("**Hız Ayarları**")
        mode = st.radio(
            "Analiz modu",
            ["Davranış Analizi", "Davranış + Yörünge Analizi", "Detaylı Analiz (Tüm Adımlar)"],
            index=0,
            help="Minimal: sadece rearing/grooming | Fast: + trajectories | Full: + heatmaps"
        )
        
        if mode == "Minimal (~2 min)":
            skip_behavior, skip_orbit, fast_mode = False, True, True
            st.info("⚡ Minimal mod: sadece davranış tespiti çalışacak")
        elif mode == "Fast (~5 min)":
            skip_behavior, skip_orbit, fast_mode = False, False, True
            st.info("🚀 Fast mod: davranış + yörüngeler (ısı haritaları atlanacak)")
        else:  # Full
            skip_behavior, skip_orbit, fast_mode = False, False, False
            st.warning("⏳ Full mod: tüm analizler yapılacak (çok uzun sürebilir!)")

        auto_inner = st.checkbox("Auto-calculate inner zone (20% margin)", value=True)
        
        if not auto_inner:
            st.markdown("**Inner Zone** (manual)")
            col1, col2 = st.columns(2)
            with col1:
                inner_xmin = st.number_input("Inner X min", value=422, step=1)
                inner_ymin = st.number_input("Inner Y min", value=182, step=1)
            with col2:
                inner_xmax = st.number_input("Inner X max", value=748, step=1)
                inner_ymax = st.number_input("Inner Y max", value=506, step=1)
        else:
            inner_xmin = inner_xmax = inner_ymin = inner_ymax = None
    else:
        # Set defaults when run_analysis is False
        fast_mode = skip_behavior = skip_orbit = False
    
    st.divider()
    
    save_to_reports = st.checkbox(
        f"Çıktıları `reports/anxiety_predictions_v2/` altına da yaz",
        value=False,
        help="Kapalıysa sadece bu oturumda indirme butonu üzerinden alabilirsin.",
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
    
    # Storage for all results
    analysis_results = {}  # stores file paths
    analysis_images = {}   # stores file bytes (persists after temp dir cleanup)
    available_outputs = set()  # fast O(1) membership checks
    fig_images = {}

    try:
        # ── STEP 0: Open-Field Analysis (if requested) ──────────────────────
        if run_analysis:
            progress.progress(0.05, text="[0/5] Open-field analysis pipeline başlıyor...")
            
            if auto_inner:
                # Auto-calculate inner zone (20% margin)
                w = arena_xmax - arena_xmin
                h = arena_ymax - arena_ymin
                inner_xmin = arena_xmin + 0.20 * w
                inner_xmax = arena_xmax - 0.20 * w
                inner_ymin = arena_ymin + 0.20 * h
                inner_ymax = arena_ymax - 0.20 * h
            
            arena_bounds = (arena_xmin, arena_xmax, arena_ymin, arena_ymax)
            inner_zone_bounds = (inner_xmin, inner_xmax, inner_ymin, inner_ymax)
            
            # Create progress output container
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
            
            # ── READ ALL IMAGE FILES INTO MEMORY (before temp dir cleanup) ────
            for key in ("behavior_timeline", "orbit_grid", "thigmotaxis", "heatmap_histogram", "heatmap_kde", "bodypart_heatmaps"):
                if analysis_results.get(key) and analysis_results[key].exists():
                    try:
                        analysis_images[key] = analysis_results[key].read_bytes()
                        available_outputs.add(key)
                    except Exception:
                        pass
            
            progress.progress(0.20, text="[1/5] Open-field analysis complete ✓")
        
        # ── STEP 1: Behavior Detection (anxiety pipeline) ────────────────────
        progress.progress(0.30, text="[2/5] DLC CSV → bouts (rearing / grooming)")
        bouts_df, n_frames = v2.detect_bouts(csv_path, fps=fps)

        # ── STEP 2: OFT metrics ──────────────────────────────────────────────
        progress.progress(0.50, text="[3/5] OFT metrikleri (locomotion / thigmotaxis / freeze / entropy)")
        oft, body_x, body_y = v2.compute_oft_metrics(csv_path, fps=fps)

        # ── STEP 3: Spatial rearing ──────────────────────────────────────────
        progress.progress(0.65, text="[4/5] Spatial rearing (center vs wall)")
        spatial = v2.compute_spatial_rearing(bouts_df, body_x, body_y)

        feat = v2.build_feature_dict(bouts_df, oft, spatial, oft["session_s"])

        # ── STEP 4: Model prediction ─────────────────────────────────────────
        progress.progress(0.80, text=f"[5/5] Model yükleme ve tahmin (tag={tag})")
        bundle, model = v2.load_model(tag)
        pred_info = v2.predict(feat, bundle, model)

        progress.progress(0.90, text="Çıktılar üretiliyor")

        subject = csv_path.stem
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # ── Generate anxiety report ──────────────────────────────────────────
        report_buf = StringIO()
        v2.render_summary(report_buf, csv_path, n_frames, feat, pred_info)
        report_text = report_buf.getvalue()
        txt_name = f"{subject}_anxiety_v2_report.txt"

        # ── Generate anxiety overview plot ───────────────────────────────────
        fig_path = out_dir / f"{subject}_anxiety_v2_overview.png"
        v2.plot_overview(csv_path, body_x, body_y, spatial, pred_info, fig_path)
        fig_images["anxiety_overview"] = fig_path.read_bytes()

        # ── JSON payload ─────────────────────────────────────────────────────
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
            
            # Copy analysis results if available (from in-memory images)
            if analysis_images:
                for key, img_bytes in analysis_images.items():
                    # Dosya adını doğrudan 'key' kelimesini kullanarak oluşturuyoruz
                    (target / f"{subject}_{key}.png").write_bytes(img_bytes)

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
top_proba  = pred_info["proba_treated"] if pred_info["pred"] == 1 else pred_info["proba_control"]

st.divider()

# ── TAB 1: ANXIETY PREDICTION ────────────────────────────────────────────────
tab_anxiety, tab_analysis, tab_downloads = st.tabs(
    ["Anxiety Prediction", "Behavioral Analysis", "Downloads & Reports"]
)

with tab_anxiety:
    col_img, col_meta = st.columns([1.4, 1])
    
    with col_img:
        st.subheader("Anxiety Prediction Overview")
        st.image(fig_images["anxiety_overview"], use_container_width=True)
    
    with col_meta:
        st.subheader("Tahmin Sonucu")
        st.markdown(f"## {pred_label}")
        st.metric("Güven", f"%{top_proba * 100:.0f}")
        
        c1, c2 = st.columns(2)
        c1.metric("P(Control)", f"{pred_info['proba_control']:.3f}")
        c2.metric("P(Treated)", f"{pred_info['proba_treated']:.3f}")
        
        st.subheader("Temel Metrikler")
        st.markdown(
            f"- Merkezde süre: **%{feat['pct_center']:.1f}**\n"
            f"- Duvar kenarında: **%{feat['pct_periphery']:.1f}**\n"
            f"- Donakalma: **%{feat['pct_freeze']:.1f}**\n"
            f"- Rearing toplam: **{int(feat['rear_count'])}** bout ({feat['rear_total_s']:.1f} s)\n"
            f"- Rearing center: **{int(feat['rear_count_center'])}** | wall: **{int(feat['rear_count_wall'])}**"
        )
        
        st.subheader("En Önemli Özellikler")
        for i, c in enumerate(pred_info["top_contributors"][:3], 1):
            st.markdown(
                f"{i}. **{c['feature']}** (z={c['z']:+.2f}) → {c['toward']}"
            )
    
    st.divider()
    st.subheader("Detaylı Rapor")
    st.code(report_text, language="text")

# ── TAB 2: BEHAVIORAL ANALYSIS ───────────────────────────────────────────────
with tab_analysis:
    if available_outputs:
        st.subheader("Open-Field Behavioral Analysis")
        
        # ── Behavior Detection ───────────────────────────────────────────────
        if "behavior_timeline" in available_outputs:
            st.markdown("### 1. Behavior Detection (Rearing & Grooming Timeline)")
            st.image(analysis_images["behavior_timeline"], use_container_width=True)
            st.caption("Bout detection timeline across session")
        
        # ── Trajectories ─────────────────────────────────────────────────────
        col_traj1, col_traj2 = st.columns(2)
        
        with col_traj1:
            if "orbit_grid" in available_outputs:
                st.markdown("### 2a. Trajectory Grid (Per-Bodypart)")
                st.image(analysis_images["orbit_grid"], use_container_width=True)
        
        with col_traj2:
            if "thigmotaxis" in available_outputs:
                st.markdown("### 2b. Thigmotaxis (Wall Hugging)")
                st.image(analysis_images["thigmotaxis"], use_container_width=True)
        
        # ── Heatmaps ─────────────────────────────────────────────────────────
        st.markdown("### 3. Activity Heatmaps (PRIMARY ANALYSIS)")
        
        col_heat1, col_heat2 = st.columns(2)
        
        with col_heat1:
            if "heatmap_histogram" in available_outputs:
                st.markdown("**Histogram Density**")
                st.image(analysis_images["heatmap_histogram"], use_container_width=True)
        
        with col_heat2:
            if "heatmap_kde" in available_outputs:
                st.markdown("**KDE Heatmap (Smooth)**")
                st.image(analysis_images["heatmap_kde"], use_container_width=True)
        
        # ── Per-bodypart grid ────────────────────────────────────────────────
        if "bodypart_heatmaps" in available_outputs:
            st.markdown("### 4. Per-Bodypart Heatmap Grid (4×3)")
            st.image(analysis_images["bodypart_heatmaps"], use_container_width=True)
            st.caption("Individual bodypart activity distribution")
        
        # ── Behavior data summary ────────────────────────────────────────────
        if analysis_results.get("behavior_bouts") and analysis_results["behavior_bouts"].exists():
            st.markdown("### 5. Behavior Bout Summary")
            try:
                bouts_data = pd.read_csv(analysis_results["behavior_bouts"])
                st.dataframe(bouts_data.head(10), use_container_width=True)
                st.caption(f"Total bouts: {len(bouts_data)}")
            except Exception as e:
                st.warning(f"Could not load bout summary: {e}")
    else:
        st.info("Open-field analysis not selected or no results. Enable in sidebar and check 'Detaylı Analizi Çalıştır' to view comprehensive behavioral results.")

# ── TAB 3: DOWNLOADS ─────────────────────────────────────────────────────────
with tab_downloads:
    st.subheader("Report Files")
    
    d1, d2, d3 = st.columns(3)
    
    with d1:
        st.download_button(
            "📝 Anxiety Report (TXT)",
            report_text,
            file_name=txt_name,
            mime="text/plain"
        )
    
    with d2:
        st.download_button(
            "📊 Anxiety Report (JSON)",
            json_bytes,
            file_name=json_name,
            mime="application/json"
        )
    
    with d3:
        st.download_button(
            "📷 Anxiety Overview (PNG)",
            fig_images["anxiety_overview"],
            file_name=f"{subject}_anxiety_v2_overview.png",
            mime="image/png"
        )
    
    if available_outputs:
        st.divider()
        st.subheader("Analysis Visualizations")
        
        cols = st.columns(3)
        col_idx = 0
        
        viz_items = [
            ("behavior_timeline", "Timeline"),
            ("orbit_grid", "Trajectories"),
            ("thigmotaxis", "Thigmotaxis"),
            ("heatmap_histogram", "Histogram"),
            ("heatmap_kde", "KDE Heatmap"),
            ("bodypart_heatmaps", "Bodypart Grid"),
        ]
        
        for key, label in viz_items:
            if key in available_outputs:
                img_bytes = analysis_images[key]
                fpath = analysis_results.get(key)
                with cols[col_idx % 3]:
                    st.download_button(
                        f"📸 {label}",
                        img_bytes,
                        file_name=fpath.name if fpath else f"{label}.png",
                        mime="image/png"
                    )
                col_idx += 1
    
    if save_to_reports:
        st.divider()
        st.success(f"✓ Tüm çıktılar `{v2.DEFAULT_OUT.relative_to(ROOT)}/` altına yazıldı.")
