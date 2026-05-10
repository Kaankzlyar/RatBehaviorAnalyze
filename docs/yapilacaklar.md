# Yapılacaklar

**Son güncelleme:** 2026-05-10  
**Öncelik:** Yukarıdan aşağı — 1. en acil

---

## 1. Plus Maze toplu metrik tablosu (acil)

**Ne:** 12 sıçanın ayrı `*_plus_maze_metrics.csv` dosyalarını tek `data/plus_maze_metrics_all.csv` tablosuna birleştir.  
**Neden:** Cohort istatistiği ve grafik için ortak tablo yok; her sıçan ayrı klasörde duruyor.  
**Çözdüğü sorun:** Grup karşılaştırması yapılamıyor.  
**Nasıl:** `analysis/tmaze/run_analysis.py`'a `--batch-dir` modu ekle veya pandas `concat` ile tek seferlik script yaz.  
**Beklenen çıktı:** `data/plus_maze_metrics_all.csv` (12 satır × 30 sütun)

---

## 2. EPM anksiyete istatistiği (acil)

**Ne:** Plus Maze açık kol % zamanı ve girişi üzerinde Kruskal-Wallis + Dunn post-hoc + permütasyon testi.  
**Neden:** OFT için `analysis/cohort_stats.py` var ama Plus Maze için yok; tez için anlamlılık testi şart.  
**Çözdüğü sorun:** "Hangi kohort daha az kaygılı?" sorusunun istatistiksel kanıtı yok.  
**Hedef metrikler:** `pct_open_arm` (left+right), `pct_open_arm_entries`, `total_entries` (lokomotor kovariat).  
**Nasıl:** `analysis/cohort_stats.py`'ı kopyala, Plus Maze feature'larına adapte et → `analysis/tmaze/cohort_stats_epm.py`.  
**Beklenen çıktı:** `reports/cohort_epm_kw.csv`, `reports/cohort_epm_dunn.csv`

---

## 3. Açık kol % grafiği (acil)

**Ne:** Kohort bazında açık-kol süresi yüzdesini gösteren box/violin plot.  
**Neden:** Tezin en önemli görsellerinden biri — hangi maddenin anksiyolitik benzeri etki yaptığını özetliyor.  
**Çözdüğü sorun:** Sonuç sayısal olarak var ama görselleştirilmemiş, teze alınamıyor.  
**Beklenen çıktı:** `reports/figures/epm_open_arm_by_cohort.png`

---

## 4. Label leakage düzeltmesi (önemli)

**Ne:** `src/train_baseline.py`'da `anxiety_level` hedefini tahmin etmek için kullanılan feature'lar, etiketin kendisini oluşturan formülle örtüşüyor (`pct_time_periphery`, `pct_time_freeze`, `center_zone_entries`).  
**Neden:** Tezde "model %62 F1 elde etti" demek yanıltıcı — model etiketi ezberliyor.  
**Çözdüğü sorun:** Savunmada sorgulanacak en kritik metodolojik açık.  
**Nasıl:** Bu 3 feature'ı X'ten çıkar, LOOCV'yi yeniden çalıştır. Gerçek F1 raporla (muhtemelen 0.10-0.15 düşer ama dürüst olur). Eski ve yeni sonuçları yan yana tablo yap.  
**Beklenen çıktı:** `reports/model_comparison_no_leakage.csv`

---

## 5. OFT + Plus Maze feature birleştirmesi

**Ne:** `src/features.py`'ı güncelle — plus maze metriklerini OFT metriklerine ekle (16 yeni özellik, `pm_` prefix).  
**Neden:** Şu anki ML modeli sadece OFT verisini görüyor; plus maze ile birlikte kohort ayrımı daha güçlü olabilir.  
**Çözdüğü sorun:** Tek arena analizi tezin kapsamını daraltıyor; iki arena birlikte daha zengin profil sunuyor.  
**Önkoşul:** Madde 1 tamamlanmış olmalı.  
**Beklenen çıktı:** `data/features/features_combined.csv` (12 satır × 42 özellik), yeniden eğitilmiş modeller

---

## 6. Window classifier için ground-truth etiketleme

**Ne:** 8+ sıçan için rearing/grooming/locomotion/immobile bout'larını elle etiketle.  
**Neden:** `src/train_window_classifier.py` hazır ama `data/windows_labeled.parquet` yok — eğitim başlatılamıyor.  
**Çözdüğü sorun:** Rule-based detector sadece 2 sıçan üzerinde doğrulandı; yeni sıçanlarda hata oranı bilinmiyor.  
**Araç:** `analysis/open_field/label_bouts.py`  
**Tahmini süre:** 6-8 saat (8 sıçan × 45 dk)  
**Beklenen çıktı:** `data/behavior_ground_truth.csv`, ardından `src/train_window_classifier.py` çalıştırılabilir

---

## Notlar

- **Madde 1-3** tez savunması için minimum viable set — bunlar bitmeden tez eksik.  
- **Madde 4** savunmada sorulacak; "bulduk ve raporladık" demek yeterli, düzeltme zorunda değilsin.  
- **Madde 5-6** zaman kalırsa; kalmazsa sonraki çalışma önerisi olarak yazılabilir.
