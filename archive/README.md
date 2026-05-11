# archive/

Aktif geliştirmenin dışında bırakılmış eski kod ve modeller. Silmek yerine arşivde tutuldu — git history korunsun ve eski sonuçlar tekrar üretilebilsin diye.

Aktif anxiety pipeline'ı `src/anxiety/`, `scripts/predict_anxiety_v2.py` ve `analysis/open_field/oft_metrics.py` üzerinde duruyor. Yapının tamamı için: `docs/REPO_MAP.md`.

## İçindekiler

### `archive/scripts/`
- **`predict_anxiety.py`** — v1 çıkarım scripti. `predict_anxiety_v2.py` (`src/anxiety/` modülü ile entegre, 18 özellikli) yerini aldı.

### `archive/src/`
- **`train_baseline.py`** — eski 4-hedefli (group / anxiety_level / rearing_profile / grooming_profile) LOOCV eğitici. `models/classifier/` modellerini üretirdi. Anxiety classifier (Control vs Treated, n=29) ile değiştirildi.
- **`features.py`** — `train_baseline.py` için 24 özellikli hand-crafted feature engineering. Modern pipeline `src/anxiety/profile.py` kullanıyor.
- **`_metrics_minimal.py`** — `ARENA` / `PERIPHERY_MARGIN` sabitleri ve minimal anxiety feature hesabı. `src/anxiety/config.py` ile çakışıyordu; yalnızca v1 `predict_anxiety.py` kullanıyordu.
- **`visualize_reports.py`** — eski 4-hedefli modeller için SHAP + confusion matrix plot'ları.
- **`train_anxiety_demo.py`** — keşif amaçlı standalone demo. Hiçbir yerden import edilmiyor.

### `archive/models/`
- **`classifier/`** — eski 4-hedefli baseline (`*_group`, `*_anxiety_level`, `*_rearing_profile`, `*_grooming_profile` × LR / RF / SVM / XGB / LightGBM / L1-MI8) toplam 25 .pkl. n=12 ile eğitildi, F1 düşük. Anxiety pivotunda `models/anxiety_classifier/` ile değiştirildi.

## Aktif modüllere geri dön
- Anxiety: `src/anxiety/`, `scripts/predict_anxiety_v2.py`
- OFT metrikleri: `analysis/open_field/oft_metrics.py`
- Davranış tespiti: `src/behavior_detection.py`
