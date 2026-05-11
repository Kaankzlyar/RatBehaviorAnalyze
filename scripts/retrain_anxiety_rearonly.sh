#!/usr/bin/env bash
# retrain_anxiety_rearonly.sh
# ---------------------------
# Mevcut INNER_ZONE (src/anxiety/config.py) ile spatial_rearing.csv'yi yeniden
# üretir, anxiety_features_extended.csv'yi yeniden inşa eder ve rear-only
# LOOCV modelini yeniden eğitir.
#
# Bunu HER ZAMAN çalıştır:
#   - INNER_ZONE / ARENA / FPS değiştiyse
#   - Yeni subject eklendiyse
#   - spatial_rearing veya OFT metrik kodu değiştiyse
#
# 2026-05-11 incident: dd5943b commit'i INNER_ZONE'u %20 margin'a düşürünce
# (önceden ~%7) training datasıyla inference arasında feature shift oluştu.
# Training rear_center_frac değerleri 0.85-1.0 (geniş center), inference
# değerleri 0.1-0.2 (dar center) → model her şeyi Control sandı. Bu script
# tam olarak bu durumu engellemek için var.
#
# Kullanım:
#   bash scripts/retrain_anxiety_rearonly.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════"
echo " Anxiety rear-only model — yeniden eğitim pipeline"
echo "════════════════════════════════════════════════════════════"
echo

echo "[1/2] spatial_rearing — current INNER_ZONE ile yeniden üret"
echo "      (data/spatial_rearing.csv + data/anxiety_features_extended.csv)"
python -m src.anxiety.spatial_rearing

echo
echo "[2/2] classifier — rear-only LOOCV (tag=rearonly)"
echo "      (models/anxiety_classifier/{lr,rf,svm}_rearonly.pkl + scaler_rearonly.pkl)"
python -m src.anxiety.classifier \
    --csv data/anxiety_features_extended.csv \
    --features 'rear|pct_periphery|pct_freeze|spatial_entropy|comfort' \
    --tag rearonly

echo
echo "════════════════════════════════════════════════════════════"
echo " ✓ Bitti. Sanity check için bilinen bir denekle dene:"
echo "   python scripts/predict_anxiety_v2.py \\"
echo "       data/DLCfiltered/<COHORT>/OpenField<ID>/OpenField<ID>.csv"
echo "════════════════════════════════════════════════════════════"
