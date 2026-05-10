"""
Predict whether a rat appears anxious from a DLC pose CSV.

Usage:
    python scripts/predict_anxiety.py path/to/<subject>.csv
    python scripts/predict_anxiety.py path/to/<subject>.csv --output reports/anxiety_predictions

Writes three files per run into the output directory:
    <subject>_anxiety_report.txt       — the same summary printed to stdout
    <subject>_anxiety_report.json      — structured features + prediction
    <subject>_anxiety_overview.png     — trajectory plot with arena/center zone
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "anxiety_demo"
DEFAULT_OUTPUT = ROOT / "reports" / "anxiety_predictions"

sys.path.insert(0, str(ROOT))
from src._metrics_minimal import ARENA, PERIPHERY_MARGIN, compute_anxiety_features, load_dlc

LINE = "─" * 56


def explain(feats: dict, prediction: int) -> list[str]:
    notes: list[str] = []
    if feats["pct_time_periphery"] > 70:
        notes.append(f"• Vaktinin %{feats['pct_time_periphery']:.0f}'ini "
                     f"duvar kenarında geçirdi (kaygı sinyali)")
    if feats["pct_time_center"] < 20:
        notes.append(f"• Merkezde sadece %{feats['pct_time_center']:.0f} "
                     f"vakit geçirdi (kaygı sinyali)")
    if feats["pct_time_freeze"] > 5:
        notes.append(f"• Donakalma süresi %{feats['pct_time_freeze']:.1f} "
                     f"(kaygı sinyali)")
    if feats["center_zone_entries"] < 10:
        notes.append(f"• Merkeze sadece {int(feats['center_zone_entries'])} "
                     f"kez girdi (keşif düşük)")
    if prediction == 0:
        if feats["pct_time_center"] >= 25:
            notes.append(f"• Merkezde %{feats['pct_time_center']:.0f} vakit "
                         f"geçirdi (rahatlık sinyali)")
        if feats["center_zone_entries"] >= 15:
            notes.append(f"• Merkeze {int(feats['center_zone_entries'])} kez "
                         f"girdi (yüksek keşif)")
    return notes


def render_summary(buf, csv_path, n_frames, feats, pred, proba, notes):
    print(f"\n🐀  Fareyi analiz ediyorum: {csv_path.name}", file=buf)
    print(LINE, file=buf)
    print(f"✓ Pose verisi okundu  ({n_frames:,} frame)", file=buf)
    print(f"✓ Davranış metrikleri hesaplandı:", file=buf)
    print(f"    Merkezde geçen süre   : %{feats['pct_time_center']:5.1f}", file=buf)
    print(f"    Duvar kenarında süre  : %{feats['pct_time_periphery']:5.1f}", file=buf)
    print(f"    Merkeze giriş sayısı  : {int(feats['center_zone_entries']):>5}", file=buf)
    print(f"    Donakalma süresi      : %{feats['pct_time_freeze']:5.1f}", file=buf)
    print("✓ Eğitilmiş model yüklendi", file=buf)
    print(LINE, file=buf)

    if pred == 1:
        confidence = proba[1] * 100
        print(f"\n🚨  SONUÇ:  Bu fare KAYGILI görünüyor", file=buf)
        print(f"    Güven   : %{confidence:.0f}", file=buf)
    else:
        confidence = proba[0] * 100
        print(f"\n✅  SONUÇ:  Bu fare SAKİN görünüyor", file=buf)
        print(f"    Güven   : %{confidence:.0f}", file=buf)

    if notes:
        print(f"\n  Neden böyle düşündü?", file=buf)
        for n in notes:
            print(f"    {n}", file=buf)
    print(file=buf)


def plot_overview(dlc, feats, pred, proba, out_path: Path) -> None:
    bc_x = dlc["body_center_x"].copy()
    bc_y = dlc["body_center_y"].copy()
    bc_lik = dlc["body_center_likelihood"]
    bc_x[bc_lik < 0.6] = np.nan
    bc_y[bc_lik < 0.6] = np.nan

    fig, ax = plt.subplots(figsize=(9, 8.5))

    A = ARENA
    ax.add_patch(plt.Rectangle(
        (A["x_left"], A["y_top"]),
        A["x_right"] - A["x_left"], A["y_bottom"] - A["y_top"],
        fill=False, edgecolor="black", linewidth=2.5))

    cx = (A["x_left"] + A["x_right"]) / 2
    cy = (A["y_top"] + A["y_bottom"]) / 2
    half_w = (A["x_right"] - A["x_left"]) / 2 * (1 - PERIPHERY_MARGIN)
    half_h = (A["y_bottom"] - A["y_top"]) / 2 * (1 - PERIPHERY_MARGIN)
    ax.add_patch(plt.Rectangle(
        (cx - half_w, cy - half_h), 2 * half_w, 2 * half_h,
        fill=True, facecolor="lightgreen", alpha=0.25,
        edgecolor="green", linewidth=1.5, linestyle="--",
        label="Merkez bölge"))

    color = "tab:red" if pred == 1 else "tab:blue"
    ax.plot(bc_x.values, bc_y.values, color=color, alpha=0.5, linewidth=0.8,
            label="Hareket izi")
    ax.scatter(bc_x.values[0:1], bc_y.values[0:1], color="green", s=80,
               zorder=5, label="Başlangıç")
    valid = bc_x.dropna()
    if len(valid):
        last = valid.index[-1]
        ax.scatter(bc_x.values[last:last + 1], bc_y.values[last:last + 1],
                   color="black", s=80, marker="X", zorder=5, label="Bitiş")

    label = "KAYGILI" if pred == 1 else "SAKİN"
    confidence = proba[pred] * 100
    icon = "🚨" if pred == 1 else "✅"
    ax.set_title(f"{icon}  {label}   (güven  %{confidence:.0f})", fontsize=18, pad=14)

    info = (f"Merkezde: %{feats['pct_time_center']:.0f}    "
            f"Duvarda: %{feats['pct_time_periphery']:.0f}    "
            f"Merkeze giriş: {int(feats['center_zone_entries'])}    "
            f"Donakalma: %{feats['pct_time_freeze']:.1f}")
    ax.text(0.5, -0.08, info, transform=ax.transAxes,
            ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"))

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("X (piksel)")
    ax.set_ylabel("Y (piksel)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Pose CSV'den kaygı tahmini.")
    p.add_argument("csv", type=Path, help="DLC filtered pose CSV")
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ Dosya bulunamadi: {args.csv}")
        sys.exit(1)

    if not (MODEL_DIR / "anxiety_classifier.pkl").exists():
        print(f"❌ Model henüz eğitilmemiş. Önce şunu çalıştır:")
        print(f"   python -m src.train_anxiety_demo")
        sys.exit(1)

    dlc = load_dlc(args.csv)
    feats = compute_anxiety_features(dlc)

    with open(MODEL_DIR / "anxiety_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols: list[str] = json.load(f)

    X = np.array([[feats[c] for c in feature_cols]])
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = model.predict_proba(Xs)[0]
    notes = explain(feats, pred)

    # 1) print summary to stdout
    render_summary(sys.stdout, args.csv, len(dlc), feats, pred, proba, notes)

    # 2) build outputs
    args.output.mkdir(parents=True, exist_ok=True)
    subject = args.csv.stem

    txt_path = args.output / f"{subject}_anxiety_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        render_summary(f, args.csv, len(dlc), feats, pred, proba, notes)

    json_path = args.output / f"{subject}_anxiety_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "subject": subject,
            "csv_path": str(args.csv),
            "n_frames": int(len(dlc)),
            "features": feats,
            "prediction": "anxious" if pred == 1 else "calm",
            "prediction_label_tr": "KAYGILI" if pred == 1 else "SAKİN",
            "confidence_pct": round(float(proba[pred] * 100), 2),
            "proba_calm": round(float(proba[0]), 4),
            "proba_anxious": round(float(proba[1]), 4),
            "explanation": notes,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2, ensure_ascii=False)

    fig_path = args.output / f"{subject}_anxiety_overview.png"
    plot_overview(dlc, feats, pred, proba, fig_path)

    print(f"📁 Raporlar yazıldı:")
    print(f"    {txt_path}")
    print(f"    {json_path}")
    print(f"    {fig_path}")


if __name__ == "__main__":
    main()
