# Repo Haritası

`develop` branch, PR #6 (feature/anxiety-analysis) sonrası mevcut yapı. Aktif olarak çalışılan ve görmezden gelinebilecek kısımları ayırmak için.

> Bu harita 2026-05-11'de yazıldı. Yapı değiştikçe güncelle. Detaylı bağlam: `docs/anxiety_progress_2026-05-10.md`.

## 🟢 Anxiety hot path

Anksiyete üzerine çalışırken pratikte sadece bu yollara bakmak yetiyor.

### Giriş noktaları (CLI / GUI)
| Yol | İşlev | Çıktı |
|---|---|---|
| `scripts/predict_anxiety_v2.py` | Uçtan uca çıkarım (CLI): DLC CSV → rearing tespiti → OFT metrikleri → spatial rearing → 18-feature vektör → Treated/Control tahmini | `reports/anxiety_predictions_v2/` |
| `scripts/predict_anxiety_v3_gui.py` | v2'nin Streamlit GUI sarmalı: `streamlit run` ile tarayıcıdan CSV yükle → overview PNG + TR rapor + indirme butonları | `reports/anxiety_predictions_v2/` (opsiyonel) |
| `src/anxiety/classifier.py` | LOOCV ile binary (Control vs Treated) eğitici (LR / RF / SVM) | `models/anxiety_classifier/` |

### `src/anxiety/` modülü (hepsi aktif)
| Dosya | İşlev |
|---|---|
| `profile.py` | Özellik matrisi inşa edici (OFT metrikleri + behavior bouts → 20+ metrik) → `data/anxiety_features.csv` |
| `classifier.py` | LR / RF / SVM eğitimi, feature filtering opsiyonları |
| `spatial_rearing.py` | Rearing bout'larını center vs wall olarak sınıflıyor |
| `composite_index.py` | Bileşik anksiyete skoru (baseline'a göre iyileşme bulunamadı) |
| `regression.py` | PC1 ekseninde regresyon (age/sex covariates) |
| `config.py` | ARENA / INNER_ZONE / FPS tek-kaynak sabitleri |

### Anxiety pipeline'ın import ettiği dış modüller
- `src/behavior_detection.py` — kural tabanlı rearing/grooming detektörü
- `analysis/open_field/oft_metrics.py` — locomotion / thigmotaxis / freeze / spatial entropy

### Aktif çıktı klasörleri
- `models/anxiety_classifier/` — `{lr,rf,svm}_rearonly.pkl` + scaler
- `models/anxiety_regression/`
- `reports/anxiety_predictions_v2/` — subject başına v2 tahmin

### Dokümanlar
- `docs/anxiety_progress_2026-05-10.md` — en güncel ilerleme logu
- `docs/anxiety_findings_report.md` — pipeline contribution + pilot bulgular
- `README.md` 36–47 — özet durum

---

## 🟡 Planlanmış / paralel (anxiety'ye henüz entegre değil)

| Yol | Durum |
|---|---|
| `src/window_classifier/` + `models/window_classifier/` | 1-saniyelik pose-window classifier (Stage 06.2). Eğitim kodu var ama anxiety pipeline hâlâ kural tabanlı `behavior_detection.py` kullanıyor. Plan: `docs/window_classifier_plan.md`. |
| `analysis/tmaze/` | T-maze görevi (farklı keypoint profili, 5 nokta). DLC datası henüz toplanmadı. Plan: `docs/tmaze_keypoints_and_layout.md`. |

---

## ⚪ Yardımcı / one-off (anxiety hot path'inde değil ama duruyor)

OFT için keşif ve görselleştirme; anxiety eğitim döngüsünde değiller.

### `analysis/open_field/`
- `behavior_analysis.py` — bout aggregate'leri → grup istatistikleri
- `kutu_validation.py` — MATLAB pipeline'ı ile çapraz doğrulama
- `label_bouts.py` — interaktif bout etiketleme
- `bodypart_heatmaps.py`, `activity_heatmap.py`, `orbit_plot.py` — görselleştirme
- `run_kare_batch.py` — tüm OFT görsellerini batch üreten runner
- `cohort_stats.py`, `run_analysis.py`, `show_frame_coords.py` — yardımcılar

### `analysis/common/`
- `speed_analysis.py` — speed pipeline + cohort özeti (anxiety'den bağımsız)

### `src/` kökünde dataset hazırlığı
- `src/dlc/` — DLC proje setup / eğitim / inference (her dataset'te bir kez)
- `src/video_preprocessing.py` — video QC (Stage 01)
- `src/synthesize_keypoints.py` — keypoint sentezi / interpolasyon

### `tools/`
- Yardımcı CLI'lar; anxiety pipeline'a girmiyor.

---

## 🔴 Arşivlendi — `archive/`

Aktif kodda kullanılmıyor, görmezden gel. Detay: `archive/README.md`.

| Eski yol | Yeni yol | Neden |
|---|---|---|
| `scripts/predict_anxiety.py` | `archive/scripts/` | v2 ile değiştirildi |
| `src/train_baseline.py` | `archive/src/` | Eski 4-hedefli baseline; anxiety classifier'a geçildi |
| `src/features.py` | `archive/src/` | `src/anxiety/profile.py` yerini aldı |
| `src/_metrics_minimal.py` | `archive/src/` | `src/anxiety/config.py` ile çakışıyordu |
| `src/visualize_reports.py` | `archive/src/` | Eski modellerin SHAP plot'ları |
| `src/train_anxiety_demo.py` | `archive/src/` | Hiçbir yerden import edilmiyor |
| `models/classifier/` (25 .pkl) | `archive/models/classifier/` | `models/anxiety_classifier/` yerini aldı |

---

## Hızlı referans

Anxiety üzerinde çalışırken bakılacak yerler (tam liste):
```
scripts/predict_anxiety_v2.py           # CLI
scripts/predict_anxiety_v3_gui.py       # Streamlit GUI sarmalayıcı
src/anxiety/
src/behavior_detection.py
analysis/open_field/oft_metrics.py

# Çıktılar
models/anxiety_classifier/
models/anxiety_regression/
reports/anxiety_predictions_v2/

# Dokümanlar
docs/anxiety_progress_2026-05-10.md
docs/anxiety_findings_report.md
```
