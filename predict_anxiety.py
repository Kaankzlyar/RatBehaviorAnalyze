"""
Demo: predict whether a rat appears anxious from a DLC pose CSV.

Usage:
    python predict_anxiety.py path/to/<subject>.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "anxiety_demo"

sys.path.insert(0, str(ROOT))
from src._metrics_minimal import compute_anxiety_features, load_dlc

LINE = "─" * 56


def explain(feats: dict, prediction: int) -> list[str]:
    """Build the bullet-point reasoning shown to the kids."""
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
    if prediction == 0:  # calm
        if feats["pct_time_center"] >= 25:
            notes.append(f"• Merkezde %{feats['pct_time_center']:.0f} vakit "
                         f"geçirdi (rahatlık sinyali)")
        if feats["center_zone_entries"] >= 15:
            notes.append(f"• Merkeze {int(feats['center_zone_entries'])} kez "
                         f"girdi (yüksek keşif)")
    return notes


def main() -> None:
    p = argparse.ArgumentParser(description="Pose CSV'den kaygı tahmini.")
    p.add_argument("csv", type=Path, help="DLC filtered pose CSV")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ Dosya bulunamadi: {args.csv}")
        sys.exit(1)

    if not (MODEL_DIR / "anxiety_classifier.pkl").exists():
        print(f"❌ Model henüz eğitilmemiş. Önce şunu çalıştır:")
        print(f"   python -m src.train_anxiety_demo")
        sys.exit(1)

    print(f"\n🐀  Fareyi analiz ediyorum: {args.csv.name}")
    print(LINE)

    dlc = load_dlc(args.csv)
    print(f"✓ Pose verisi okundu  ({len(dlc):,} frame)")

    feats = compute_anxiety_features(dlc)
    print(f"✓ Davranış metrikleri hesaplandı:")
    print(f"    Merkezde geçen süre   : %{feats['pct_time_center']:5.1f}")
    print(f"    Duvar kenarında süre  : %{feats['pct_time_periphery']:5.1f}")
    print(f"    Merkeze giriş sayısı  : {int(feats['center_zone_entries']):>5}")
    print(f"    Donakalma süresi      : %{feats['pct_time_freeze']:5.1f}")

    # load model
    with open(MODEL_DIR / "anxiety_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols: list[str] = json.load(f)

    print("✓ Eğitilmiş model yüklendi")
    print(LINE)

    X = np.array([[feats[c] for c in feature_cols]])
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = model.predict_proba(Xs)[0]

    if pred == 1:
        confidence = proba[1] * 100
        print(f"\n🚨  SONUÇ:  Bu fare KAYGILI görünüyor")
        print(f"    Güven   : %{confidence:.0f}")
    else:
        confidence = proba[0] * 100
        print(f"\n✅  SONUÇ:  Bu fare SAKİN görünüyor")
        print(f"    Güven   : %{confidence:.0f}")

    notes = explain(feats, pred)
    if notes:
        print(f"\n  Neden böyle düşündü?")
        for n in notes:
            print(f"    {n}")

    print()


if __name__ == "__main__":
    main()
