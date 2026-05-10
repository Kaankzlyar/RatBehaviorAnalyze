# Yapılacaklar

**Son güncelleme:** 2026-05-10  
**Öncelik:** Yukarıdan aşağı — 1. en acil

---

## ✅ 1. Plus Maze toplu metrik tablosu — TAMAMLANDI

**Ne:** 12 sıçanın ayrı `*_plus_maze_metrics.csv` dosyalarını tek `data/plus_maze_metrics_all.csv` tablosuna birleştir.  
**Çıktı:** `data/plus_maze_metrics_all.csv` (12 satır × 35 sütun)  
**Script:** `analysis/tmaze/batch_metrics.py`

---

## ✅ 2. EPM anksiyete istatistiği — TAMAMLANDI

**Ne:** Plus Maze açık kol % zamanı ve girişi üzerinde Kruskal-Wallis + Dunn post-hoc + permütasyon testi.  
**Çıktı:** `reports/cohort_epm_kw.csv`, `reports/cohort_epm_dunn.csv`, `reports/cohort_epm_permanova.csv`  
**Script:** `analysis/tmaze/cohort_stats_epm.py`  
**Bulgular:** pct_open_arm_entries p=0.0488 ε²=0.493 (medium); PERMANOVA R²=0.368 p=0.150

---

## ✅ 3. Açık kol % grafiği — TAMAMLANDI

**Ne:** Kohort bazında açık-kol süresi yüzdesini gösteren box/strip + stacked bar + scatter.  
**Çıktı:** `reports/figures/epm_open_arm_by_cohort.png`, `epm_arm_distribution.png`, `epm_locomotor_covariate.png`  
**Script:** `analysis/tmaze/plot_open_arm.py`

---

## ✅ 4. Label leakage düzeltmesi — TAMAMLANDI

**Ne:** `src/train_baseline.py`'da `anxiety_level` hedefini tahmin etmek için kullanılan feature'lardan etiket bileşenlerini çıkar.  
**Çıktı:** `reports/model_comparison_no_leakage.csv`  
**Sonuç:** anxiety_level LOOCV F1 = 0.65 → 0.17-0.26 (dürüst baseline); LEAKAGE_MAP ile 6 feature temizlendi.

---

## ✅ 5. OFT + Plus Maze feature birleştirmesi — TAMAMLANDI

**Ne:** `src/features.py`'ı güncelle — plus maze metriklerini OFT metriklerine ekle (16 `pm_` prefix özellik).  
**Çıktı:** `data/features/features_combined.csv` (12 satır × 42 özellik), `scaler_combined.pkl`  
**Not:** Eşleştirme: `PlusMazeMA1_1` → `MA1_1` (prefix strip)

---

## 6. Window classifier için ground-truth etiketleme (Manuel İş)

**Ne:** 8+ sıçan için rearing/grooming/locomotion/immobile bout'larını elle etiketle.  
**Neden:** `src/train_window_classifier.py` hazır ama `data/behavior_ground_truth.csv` yok — eğitim başlatılamıyor.  
**Araç (hazır):** `analysis/open_field/label_bouts.py`  
**Çalıştırma:**
```bash
python analysis/open_field/label_bouts.py \
    --subject OpenFieldMA1_1 \
    --video /path/to/OpenFieldMA1_1.avi
```
**Tuş kılavuzu:** SPACE=oynat | r/g/l/i=etiket | b=başlat | e=bitir | A=tüm zayıf etiketleri kabul | s=kaydet | q=çık  
**Tahmini süre:** 6-8 saat (8 sıçan × 45 dk)  
**Beklenen çıktı:** `data/behavior_ground_truth.csv`, ardından `src/train_window_classifier.py` çalıştırılabilir

---

## Notlar

- **Madde 1-5** tamamlandı — tez savunması için minimum viable set hazır.  
- **Madde 6** zaman kalırsa; kalmazsa sonraki çalışma önerisi olarak yazılabilir.
- **Label leakage:** Savunmada sorulursa — "bulduk, LEAKAGE_MAP ile temizledik, gerçek F1=%17-26 olarak raporladık" — yeterli.
