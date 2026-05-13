"""
predict_anxiety_v2.py
---------------------
End-to-end Treated-vs-Control tahmini için bir DLC pose CSV'sinden başlar:

  DLC CSV
    └─ behavior_detection (rearing/grooming bout'ları)
    └─ oft_metrics       (locomotion / thigmotaxis / freeze / entropy)
    └─ spatial_rearing   (her bout center / wall)
    └─ rear-only feature vector (18 sütun, scaler_rearonly.pkl ile uyumlu)
    └─ models/anxiety_classifier/lr_rearonly.pkl tahmini

Çıktılar (--output altına):
    <subject>_anxiety_v2_report.txt    — okunabilir özet
    <subject>_anxiety_v2_report.json   — yapılandırılmış features + prediction
    <subject>_anxiety_v2_overview.png  — arena + rearing konum overlay'i

Kullanım
--------
  python scripts/predict_anxiety_v2.py path/to/<subject>.csv
  python scripts/predict_anxiety_v2.py path/to/<subject>.csv -o reports/preds_v2

Önemli
------
Model "Treated vs Control" sınıflandırması yapar. Treated = aspartam ve/veya
grapefruit alan tüm hayvanlar; doğrudan "anksiyeteli" değildir. Yorumlama
şekli: "tedavi-kaynaklı davranışsal değişim sinyali". Detay ve kalibrasyon
durumu için `docs/anxiety_progress_2026-05-10.md`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse existing modules
from src.behavior_detection import (
    LIKELIHOOD_THRESH,
    INTER_BOUT_GAP,
    MIN_BOUT_FRAMES,
    DEFAULT_FPS,
    bouts_to_dataframe,
    classify_frames,
    compute_features as bd_compute_features,
    frames_to_bouts,
    load_dlc_csv,
    mask_low_likelihood,
)
from analysis.open_field.oft_metrics import (
    freezing_metrics,
    load_body_center,
    locomotion_metrics,
    spatial_entropy,
    thigmotaxis_metrics,
)
from src.anxiety.config import ARENA, INNER_ZONE
from src.anxiety.spatial_rearing import classify_bout

EARLY_S    = 90.0   # erken faz cutoff (profile.py ile uyumlu)
MODEL_DIR  = ROOT / "models" / "anxiety_classifier"
DEFAULT_OUT = ROOT / "reports" / "anxiety_predictions_v2"

LINE = "─" * 60


# ── Pipeline -----------------------------------------------------------------

def detect_bouts(csv_path: Path, fps: float) -> tuple[pd.DataFrame, int]:
    """DLC CSV → rearing/grooming bout dataframe (start_frame/end_frame/duration)."""
    raw = load_dlc_csv(str(csv_path))
    masked = mask_low_likelihood(raw, LIKELIHOOD_THRESH)
    feat = bd_compute_features(raw, masked, fps=fps)
    labels = classify_frames(feat)

    rear_idx  = np.where(labels == "rearing")[0]
    groom_idx = np.where(labels == "grooming")[0]
    rear_b  = frames_to_bouts(rear_idx,  gap=INTER_BOUT_GAP, min_dur=MIN_BOUT_FRAMES)
    groom_b = frames_to_bouts(groom_idx, gap=INTER_BOUT_GAP, min_dur=MIN_BOUT_FRAMES)

    parts = [bouts_to_dataframe(rear_b, "rearing", fps),
             bouts_to_dataframe(groom_b, "grooming", fps)]
    parts = [p for p in parts if not p.empty]
    bout_df = (pd.concat(parts).sort_values("start_frame").reset_index(drop=True)
               if parts else pd.DataFrame(
                   columns=["behaviour", "bout", "start_frame", "end_frame",
                            "start_s", "end_s", "duration_s"]))
    return bout_df, len(raw)


def compute_oft_metrics(csv_path: Path, fps: float) -> dict:
    """oft_metrics.py'deki helper'ları doğrudan çağır."""
    x, y = load_body_center(str(csv_path), likelihood_thresh=0.6,
                            jump_thresh=60.0, smooth=5)
    out: dict = {"n_frames": len(x), "session_s": len(x) / fps}
    out.update(locomotion_metrics(x, y, fps))
    out.update(thigmotaxis_metrics(x, y, INNER_ZONE))
    out.update(freezing_metrics(x, y, fps))
    out["spatial_entropy_norm"] = spatial_entropy(x, y, ARENA)
    return out, x, y


def compute_spatial_rearing(bouts_df: pd.DataFrame,
                             body_x: np.ndarray, body_y: np.ndarray) -> dict:
    """Her rearing bout için body_center medyanı → center / wall."""
    rears = bouts_df[bouts_df["behaviour"] == "rearing"]
    n_total = len(rears)
    n_center = n_wall = n_unk = 0
    s_center = s_wall = 0.0
    bout_zones: list[dict] = []

    for _, r in rears.iterrows():
        s = int(r["start_frame"]); e = int(r["end_frame"])
        zone, mx, my = classify_bout(body_x, body_y, s, e, INNER_ZONE)
        bout_zones.append({"bout": int(r["bout"]), "start_s": float(r["start_s"]),
                           "duration_s": float(r["duration_s"]),
                           "median_x": mx, "median_y": my, "zone": zone})
        if zone == "center":
            n_center += 1; s_center += float(r["duration_s"])
        elif zone == "wall":
            n_wall += 1; s_wall += float(r["duration_s"])
        else:
            n_unk += 1

    classified = n_center + n_wall
    return {
        "rear_count_center":      n_center,
        "rear_count_wall":        n_wall,
        "rear_count_unknown":     n_unk,
        "rear_center_frac":       (n_center / classified) if classified else float("nan"),
        "rear_total_s_center":    round(s_center, 2),
        "rear_total_s_wall":      round(s_wall, 2),
        "rear_center_minus_wall": (n_center - n_wall),
        "_bout_zones":            bout_zones,
    }


def build_feature_dict(bouts_df: pd.DataFrame, oft: dict, spatial: dict,
                       session_s: float) -> dict:
    """profile.py'deki feature isimleriyle birebir aynı sözlük döner."""
    rears  = bouts_df[bouts_df["behaviour"] == "rearing"]
    grooms = bouts_df[bouts_df["behaviour"] == "grooming"]

    r_count = len(rears); r_total = float(rears["duration_s"].sum()) if r_count else 0.0
    g_count = len(grooms); g_total = float(grooms["duration_s"].sum()) if g_count else 0.0

    def _early_frac(sub):
        if len(sub) == 0:
            return float("nan")
        early = sub[sub["start_s"] < EARLY_S]["duration_s"].sum()
        total = sub["duration_s"].sum()
        return float(early / total) if total > 0 else float("nan")

    def _bout_cv(sub):
        if len(sub) == 0:
            return float("nan")
        d = sub["duration_s"].to_numpy()
        return float(d.std() / d.mean()) if d.mean() > 0 else float("nan")

    feat = {
        "pct_periphery":     oft["pct_time_periphery"],
        "pct_center":        oft["pct_time_center"],
        "center_entries":    oft["center_zone_entries"],
        "total_distance_px": oft["total_distance_px"],
        "mean_speed_px_s":   oft["mean_speed_px_s"],
        "spatial_entropy":   oft["spatial_entropy_norm"],
        "pct_freeze":        oft["pct_time_freeze"],
        "freeze_bout_count": oft["freeze_bout_count"],
        "rear_count":        float(r_count),
        "rear_total_s":      r_total,
        "rear_pct":          (100.0 * r_total / session_s) if session_s > 0 else float("nan"),
        "rear_mean_bout_s":  (r_total / r_count) if r_count else 0.0,
        "rear_rate_per_min": (60.0 * r_count / session_s) if session_s > 0 else float("nan"),
        "rear_early_frac":   _early_frac(rears),
        "groom_count":       float(g_count),
        "groom_total_s":     g_total,
        "groom_pct":         (100.0 * g_total / session_s) if session_s > 0 else float("nan"),
        "groom_mean_bout_s": (g_total / g_count) if g_count else 0.0,
        "groom_bout_cv":     _bout_cv(grooms),
        "groom_early_frac":  _early_frac(grooms),
        "rear_groom_ratio":  (r_total / g_total) if g_total > 0 else float("nan"),
        "rear_per_100px":    (100.0 * r_count / oft["total_distance_px"])
                              if oft["total_distance_px"] > 0 else float("nan"),
        "comfort_ratio":     (g_total / oft["pct_time_center"])
                              if oft["pct_time_center"] > 0 else float("nan"),
    }
    # spatial rearing keys
    for k in ("rear_count_center", "rear_count_wall", "rear_center_frac",
              "rear_total_s_center", "rear_total_s_wall", "rear_center_minus_wall"):
        feat[k] = spatial[k]
    return feat


# ── Model ı/o ----------------------------------------------------------------

def load_model(tag: str = "rearonly") -> tuple[dict, object]:
    sfx = f"_{tag}" if tag else ""
    scaler_path = MODEL_DIR / f"scaler{sfx}.pkl"
    model_path  = MODEL_DIR / f"lr{sfx}.pkl"
    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Model dosyaları yok: {scaler_path.name}, {model_path.name}\n"
            f"Önce: python -m src.anxiety.classifier --csv data/anxiety_features_extended.csv "
            f"--features 'rear|pct_periphery|pct_freeze|spatial_entropy|comfort' --tag rearonly"
        )
    with open(scaler_path, "rb") as f:
        bundle = pickle.load(f)   # {"imputer", "scaler", "features"}
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return bundle, model


def predict(feat: dict, bundle: dict, model) -> dict:
    cols    = bundle["features"]
    imp     = bundle["imputer"]
    sc      = bundle["scaler"]

    x = np.array([[feat.get(c, np.nan) for c in cols]], dtype=float)
    x_i = imp.transform(x)
    x_s = sc.transform(x_i)

    # Fix for scikit-learn version compatibility (multi_class attribute)
    if not hasattr(model, 'multi_class'):
        model.multi_class = 'auto'

    pred  = int(model.predict(x_s)[0])
    proba = model.predict_proba(x_s)[0]

    # En çok pushlayan featureları çıkar (LR: coef × standardized value)
    coef = model.coef_.ravel()
    contrib = coef * x_s[0]   # +: Treated yönüne push, -: Control yönüne
    order = np.argsort(-np.abs(contrib))
    top = [{"feature":  cols[i],
            "value":    float(x[0, i]) if not np.isnan(x[0, i]) else None,
            "z":        round(float(x_s[0, i]), 2),
            "coef":     round(float(coef[i]), 3),
            "push":     round(float(contrib[i]), 3),
            "toward":   "Treated" if contrib[i] > 0 else "Control"}
           for i in order[:5]]

    return {"pred": pred, "proba_control": float(proba[0]),
            "proba_treated": float(proba[1]),
            "top_contributors": top, "feature_cols": cols}


# ── Render -------------------------------------------------------------------

def label_str(pred: int) -> str:
    return "Treated-benzeri" if pred == 1 else "Control-benzeri"


def render_summary(buf, csv_path: Path, n_frames: int, feat: dict,
                   pred_info: dict) -> None:
    proba = pred_info["proba_treated"] if pred_info["pred"] == 1 else pred_info["proba_control"]
    print(f"\n🐀  DLC CSV: {csv_path.name}", file=buf)
    print(LINE, file=buf)
    print(f"✓ Pose verisi okundu        ({n_frames:,} frame)", file=buf)
    print(f"✓ Davranış bout'ları çıkarıldı  "
          f"(rearing={int(feat['rear_count'])}, grooming={int(feat['groom_count'])})", file=buf)
    print(f"✓ OFT metrikleri ve mekansal rearing hesaplandı", file=buf)
    print(f"✓ Rear-only model yüklendi  (LogReg, AUC=0.73 LOOCV)", file=buf)
    print(LINE, file=buf)
    print(f"\n  Anahtar metrikler:", file=buf)
    print(f"    Merkezde süre        : %{feat['pct_center']:5.1f}", file=buf)
    print(f"    Duvar kenarında süre : %{feat['pct_periphery']:5.1f}", file=buf)
    print(f"    Donakalma süresi     : %{feat['pct_freeze']:5.1f}", file=buf)
    print(f"    Rearing toplam       : {int(feat['rear_count'])} bout, "
          f"{feat['rear_total_s']:.1f} s", file=buf)
    print(f"    Rearing — center     : {int(feat['rear_count_center'])} bout, "
          f"{feat['rear_total_s_center']:.1f} s", file=buf)
    print(f"    Rearing — wall       : {int(feat['rear_count_wall'])} bout, "
          f"{feat['rear_total_s_wall']:.1f} s", file=buf)
    rcf = feat['rear_center_frac']
    if rcf == rcf:  # not NaN
        print(f"    rear_center_frac     : {rcf:.3f}  "
              f"(eğitim datası grup medianları için: "
              f"reports/spatial_rearing_stats.csv)", file=buf)
    print(LINE, file=buf)

    icon = "🟧" if pred_info["pred"] == 1 else "🟦"
    print(f"\n{icon} TAHMİN: {label_str(pred_info['pred'])}", file=buf)
    print(f"    P(Treated)={pred_info['proba_treated']:.3f}    "
          f"P(Control)={pred_info['proba_control']:.3f}", file=buf)
    print(f"    Güven      : %{proba*100:.0f}", file=buf)
    print(f"\n  Kararı en çok etkileyen 5 özellik (LR coef × z-skor):", file=buf)
    for c in pred_info["top_contributors"]:
        sign = "+" if c["push"] >= 0 else "−"
        print(f"    {c['feature']:<26s}  z={c['z']:>+5.2f}  "
              f"push={sign}{abs(c['push']):.2f} → {c['toward']}", file=buf)
    print("\n  Not: model 'Treated vs Control' ayırıyor — doğrudan 'anksiyeteli' değil.", file=buf)
    print(file=buf)


def plot_overview(csv_path: Path, body_x, body_y, spatial: dict,
                  pred_info: dict, out: Path,
                  show_title: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(9, 8.4))
    # arena
    ax.plot([ARENA[0], ARENA[1], ARENA[1], ARENA[0], ARENA[0]],
            [ARENA[2], ARENA[2], ARENA[3], ARENA[3], ARENA[2]],
            "k-", lw=2)
    ax.plot([INNER_ZONE[0], INNER_ZONE[1], INNER_ZONE[1], INNER_ZONE[0], INNER_ZONE[0]],
            [INNER_ZONE[2], INNER_ZONE[2], INNER_ZONE[3], INNER_ZONE[3], INNER_ZONE[2]],
            "k--", lw=1, alpha=0.6, label="İç bölge (manual)")

    # path
    color = "tab:red" if pred_info["pred"] == 1 else "tab:blue"
    ax.plot(body_x, body_y, color=color, alpha=0.35, lw=0.7, label="body_center izi")

    # rear bouts
    centers = [b for b in spatial["_bout_zones"] if b["zone"] == "center"]
    walls   = [b for b in spatial["_bout_zones"] if b["zone"] == "wall"]
    if walls:
        ax.scatter([b["median_x"] for b in walls], [b["median_y"] for b in walls],
                   color="orange", edgecolor="black", s=80,
                   label=f"wall rear (n={len(walls)})", zorder=4)
    if centers:
        ax.scatter([b["median_x"] for b in centers], [b["median_y"] for b in centers],
                   color="lime", edgecolor="black", s=110, marker="*",
                   label=f"center rear (n={len(centers)})", zorder=5)

    
    if show_title:
        ax.set_title(f"{label_str(pred_info['pred'])}  "
                     f"(P_treated={pred_info['proba_treated']:.2f})",
                     fontsize=14, pad=12)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── CLI ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="DLC CSV'den rear-only Treated/Control tahmini.")
    p.add_argument("csv", type=Path, help="DLC filtered pose CSV")
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT,
                   help=f"Çıktı klasörü (default: {DEFAULT_OUT})")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help=f"Video FPS (default {DEFAULT_FPS})")
    p.add_argument("--tag", default="rearonly",
                   help="Hangi modeli yükleyeceğini belirtir (default 'rearonly').")
    args = p.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.exists():
        print(f"❌ CSV bulunamadı: {csv_path}")
        sys.exit(1)

    print(f"[1/4] DLC CSV → bouts ({csv_path.name})")
    bouts_df, n_frames = detect_bouts(csv_path, fps=args.fps)
    print(f"      rearing={int((bouts_df['behaviour']=='rearing').sum())}  "
          f"grooming={int((bouts_df['behaviour']=='grooming').sum())}")

    print("[2/4] OFT metrikleri (locomotion / thigmotaxis / freeze / entropy)")
    oft, body_x, body_y = compute_oft_metrics(csv_path, fps=args.fps)

    print("[3/4] Spatial rearing (center vs wall)")
    spatial = compute_spatial_rearing(bouts_df, body_x, body_y)
    print(f"      center={spatial['rear_count_center']}  "
          f"wall={spatial['rear_count_wall']}  "
          f"center_frac={spatial['rear_center_frac']:.3f}"
          if spatial["rear_count_center"] + spatial["rear_count_wall"] > 0
          else "      hiç sınıflanabilir rearing bout yok")

    feat = build_feature_dict(bouts_df, oft, spatial, oft["session_s"])

    print(f"[4/4] Model yükleme ve tahmin (tag={args.tag})")
    bundle, model = load_model(args.tag)
    pred_info = predict(feat, bundle, model)

    # ── outputs ─────────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    subject = csv_path.stem

    render_summary(sys.stdout, csv_path, n_frames, feat, pred_info)

    txt_path = args.output / f"{subject}_anxiety_v2_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        render_summary(f, csv_path, n_frames, feat, pred_info)

    json_payload = {
        "subject":         subject,
        "csv_path":        str(csv_path),
        "n_frames":        n_frames,
        "session_s":       oft["session_s"],
        "model_tag":       args.tag,
        "pred":            pred_info["pred"],
        "label":           label_str(pred_info["pred"]),
        "proba_control":   round(pred_info["proba_control"], 4),
        "proba_treated":   round(pred_info["proba_treated"], 4),
        "top_contributors": pred_info["top_contributors"],
        "features":        {k: (None if isinstance(v, float) and np.isnan(v) else v)
                             for k, v in feat.items()},
        "feature_cols_used": pred_info["feature_cols"],
        "n_rearing_bouts":  int((bouts_df["behaviour"]=="rearing").sum()),
        "n_grooming_bouts": int((bouts_df["behaviour"]=="grooming").sum()),
        "spatial_rearing": {
            "rear_count_center":   spatial["rear_count_center"],
            "rear_count_wall":     spatial["rear_count_wall"],
            "rear_count_unknown":  spatial["rear_count_unknown"],
            "rear_center_frac":    (None if spatial["rear_center_frac"] != spatial["rear_center_frac"]
                                    else round(spatial["rear_center_frac"], 4)),
        },
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
    }
    json_path = args.output / f"{subject}_anxiety_v2_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    fig_path = args.output / f"{subject}_anxiety_v2_overview.png"
    plot_overview(csv_path, body_x, body_y, spatial, pred_info, fig_path)

    # ham bouts CSV — debugging için
    bouts_path = args.output / f"{subject}_behavior_bouts.csv"
    bouts_df.to_csv(bouts_path, index=False)

    print(f"\n📁 Çıktılar:")
    for p in (txt_path, json_path, fig_path, bouts_path):
        print(f"    {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")


if __name__ == "__main__":
    main()
