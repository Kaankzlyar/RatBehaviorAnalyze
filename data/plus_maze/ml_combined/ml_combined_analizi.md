# ML Combined — OFT + Plus Maze Birleşik Makine Öğrenmesi Analizi

**Script:** `src/train_combined.py`  
**Veri kaynağı:** `data/features/features_combined_normalized.csv` (12 sıçan, 46 özellik)  
**Etiketler:** `data/features/labels.csv`

---

## Bu Analiz Neden Yapıldı?

OFT (Açık Alan Testi) tabanlı ML pipeline'ı yalnızca 28 OFT özelliğini kullanıyordu. Plus Maze analizinden elde edilen 16 ek `pm_*` özelliği ile birleştirilmiş bir özellik seti oluşturularak iki soru yanıtlanmak istendi:

1. **SHAP analizinde `pm_*` özelliklerinin grup ayrımına katkısı nedir?** — hangi Plus Maze metriğinin sınıflandırmaya katkı sağladığı görselleştirilir.
2. **Combined pipeline, OFT-only'e kıyasla F1 değerini değiştiriyor mu?** — iyileşme, gerileme veya fark yok.

---

## Veri Seti

| Parametre | Değer |
|-----------|-------|
| Sıçan sayısı | 12 (n=3/grup) |
| Toplam özellik sayısı | 46 |
| OFT özellikleri | 28 |
| Plus Maze özellikleri (`pm_*`) | 16 |
| Etiket sütunları | 4 (aşağıya bakınız) |

### Özellik sütunları

**OFT (28 özellik):**
`total_distance_px`, `mean_speed_px_s`, `max_speed_px_s`, `pct_time_center`, `pct_time_periphery`, `center_zone_entries`, `exploration_ratio`, `spatial_entropy_norm`, `freeze_bout_count`, `total_freeze_s`, `pct_time_freeze`, `rear_bout_count`, `rear_per_min`, `rear_total_s`, `rear_pct_time`, `rear_mean_s`, `rear_max_s`, `rear_frag_idx`, `groom_bout_count`, `groom_per_min`, `groom_total_s`, `groom_pct_time`, `groom_mean_s`, `groom_max_s`, `groom_frag_idx`, `locomotion_pct`, `session_duration_s`, `group`

**Plus Maze — `pm_*` (16 özellik):**
`pm_pct_open_arm`, `pm_pct_closed_arm`, `pm_pct_open_arm_entries`, `pm_pct_closed_arm_entries`, `pm_anxiety_index_epm`, `pm_total_entries`, `pm_mean_speed_px_s`, `pm_total_distance_px`, `pm_arm_preference_index`, `pm_successive_alternation_pct`, `pm_perseveration_rate_pct`, `pm_pct_time_left`, `pm_pct_time_right`, `pm_pct_time_top`, `pm_pct_time_bottom`, `pm_pct_time_junction`

### Normalizasyon

`features_combined_normalized.csv` kullanılmaktadır — özellikler StandardScaler ile z-score'a dönüştürülmüştür. Ham değerler `features_combined.csv`'dedir.

---

## Etiketler (Hedef Değişkenler)

Dört ayrı sınıflandırma görevi çalıştırılmıştır:

| Hedef (`target`) | Sınıf sayısı | Sınıflar | Leakage temizlendi mi? |
|------------------|-------------|----------|------------------------|
| `group` | 4 | Control, Aspartame, Grapefruit, Aspartame+Grapefruit | Hayır |
| `anxiety_level` | 2 | low, high | Evet |
| `rearing_profile` | 3 | low, moderate, high | Evet |
| `grooming_profile` | 3 | low, moderate, high | Evet |

### Leakage (Veri Sızıntısı) Önlemi

Belirli etiketler için o etiketin hesaplanmasında doğrudan kullanılan özellikler eğitim setinden çıkarılmaktadır:

| Hedef | Çıkarılan özellikler |
|-------|----------------------|
| `anxiety_level` | `pct_time_periphery`, `pct_time_freeze`, `center_zone_entries`, `pct_time_center`, `total_freeze_s`, `freeze_bout_count` |
| `rearing_profile` | `rear_pct_time`, `rear_total_s`, `rear_per_min`, `rear_bout_count` |
| `grooming_profile` | `groom_pct_time`, `groom_total_s`, `groom_per_min`, `groom_bout_count` |
| `group` | — (çıkarma yok) |

`pm_*` özellikleri hiçbir OFT etiketinin hesaplanmasında kullanılmadığından leakage listesine dahil edilmemiştir.

---

## Modeller

Beş (veya LightGBM yüklüyse altı) model çalıştırılmıştır:

| Model | Hiperparametreler |
|-------|-------------------|
| **LogisticReg** | L1 ceza, SAGA çözücü, C=0.5, max_iter=2000 |
| **RandomForest** | 200 ağaç, max_depth=3, min_samples_leaf=2 |
| **XGBoost** | 200 tahmin, max_depth=2, learning_rate=0.05, subsample=0.8 |
| **SVM** | RBF kernel, C=1.0, gamma=scale, probability=True |
| **LogReg_L1_MI8** | SelectKBest (mutual_info, k=8) → L1 Logistic (C=0.1) |
| **LightGBM** *(opsiyonel)* | 200 tahmin, max_depth=3, num_leaves=7, learning_rate=0.05 |

---

## Çapraz Doğrulama Stratejileri

### LOOCV — Leave-One-Out Cross-Validation

- **12 katman:** Her katmanda 1 sıçan test, 11 sıçan eğitim
- **Amaç:** Bireysel düzeyde tahmin gücünü ölçmek
- **Kısıt:** n=12 ile overfitting'e açık; yüksek varyans beklenir

### LOGOCV — Leave-One-Group-Out Cross-Validation

- **4 katman:** Her katmanda 1 kohort (3 sıçan) test, 3 kohort (9 sıçan) eğitim
- **Amaç:** Kohort genelleşebilirliğini test etmek — model daha önce hiç görmediği bir kohortu tahmin edebiliyor mu?
- **Kısıt:** n=3/grup ile eğitim setinde sınıf başına yalnızca 2–3 örnek; F1 sıfıra düşebilir

---

## Sonuçlar — F1 Macro (LOOCV)

### Dört hedef için en iyi F1

| Hedef | En iyi model | LOOCV F1 | Şans seviyesi |
|-------|-------------|----------|--------------|
| `group` | LogisticReg | **0.200** | 0.25 (1/4) |
| `anxiety_level` | LogisticReg | **0.833** | 0.50 (1/2) |
| `rearing_profile` | SVM | **0.482** | 0.33 (1/3) |
| `grooming_profile` | XGBoost | **0.427** | 0.33 (1/3) |

**Dikkat:** `anxiety_level` için F1=0.833 çarpıcı görünse de bu, yalnızca 2 sınıf (low/high) ve n=12 ile hesaplanmıştır; LOGOCV'de F1=0.333'e düşmektedir.

### Tüm model sonuçları

| Hedef | Model | LOOCV F1 | LOGOCV F1 |
|-------|-------|----------|----------|
| group | LogisticReg | 0.200 | 0.000 |
| group | RandomForest | 0.063 | 0.000 |
| group | XGBoost | 0.196 | 0.000 |
| group | SVM | 0.000 | 0.000 |
| group | LogReg_L1_MI8 | 0.000 | 0.000 |
| anxiety_level | LogisticReg | **0.833** | 0.333 |
| anxiety_level | RandomForest | 0.667 | 0.413 |
| anxiety_level | XGBoost | 0.748 | 0.167 |
| anxiety_level | SVM | 0.667 | 0.333 |
| anxiety_level | LogReg_L1_MI8 | 0.250 | 0.200 |
| rearing_profile | SVM | **0.482** | 0.482 |
| rearing_profile | XGBoost | 0.345 | 0.386 |
| rearing_profile | RandomForest | 0.235 | 0.345 |
| grooming_profile | XGBoost | **0.427** | 0.339 |
| grooming_profile | RandomForest | 0.288 | 0.265 |

### One-vs-Rest (OvR) Binary Sınıflandırma — Kohort Bazlı

Her kohort ayrı ayrı "bu gruba karşı diğerleri" ikili sınıflandırması olarak test edilmiştir (XGBoost, LOOCV):

| Kohort | Binary F1 | Accuracy |
|--------|-----------|----------|
| Control | 0.400 | 0.750 |
| Aspartame | 0.000 | 0.500 |
| Grapefruit | 0.000 | 0.583 |
| ASP+Greyfurt | 0.286 | 0.583 |

---

## Çıktı Dosyaları

| Dosya | İçerik |
|-------|--------|
| `combined_model_comparison.csv` | 5 model × 4 hedef × 2 CV stratejisi F1/Acc tablosu |
| `combined_loocv_predictions.csv` | Her sıçan için LOOCV tahmin detayları |
| `combined_ovr_binary_f1.csv` | OvR binary F1 sonuçları (4 kohort) |
| `combined_confusion_<hedef>.png` | En iyi modelin karışıklık matrisi (4 adet) |
| `combined_shap_<hedef>.png` | XGBoost SHAP önem grafiği — her sınıf için (4 adet) |
| `combined_shap_ovr_groups.png` | OvR SHAP — `pm_*` özellikleri kırmızı etiketle |
| `combined_cv_comparison.png` | LOOCV vs LOGOCV F1 bar grafiği |
| `combined_vs_oft_comparison.png` | OFT-only vs OFT+PM delta karşılaştırması |

---

## Grafikler Nasıl Okunmalı?

### `combined_shap_ovr_groups.png`

Her panelde bir kohort için "bu kohort mu, diğerleri mi?" sorusuna yanıt veren XGBoost modelinin SHAP önem değerleri gösterilir.

- **Kırmızı etiket:** `pm_*` prefiksiyle başlayan Plus Maze özellikleri
- **Siyah etiket:** OFT özellikleri
- **Y ekseni:** Özellik adı (en önemli üstte)
- **X ekseni:** Ortalama |SHAP| değeri — ne kadar büyükse o özellik sınıflandırmaya o kadar katkı yapmaktadır

### `combined_vs_oft_comparison.png`

Aynı XGBoost modeli, aynı LOOCV protokolüyle iki farklı özellik setiyle çalıştırılmıştır:

- **Gri çubuk:** OFT-only (28 özellik)
- **Pembe çubuk:** OFT + PM (46 özellik)
- **Δ değeri:** `(OFT+PM F1) − (OFT-only F1)` — yeşil = iyileşme, kırmızı = gerileme

### `combined_cv_comparison.png`

2×2 panel: her hedef için LOOCV vs LOGOCV karşılaştırması. Kesikli gri çizgi = şans seviyesi (1/n_sınıf).

---

## Genel Yorum

n=12 ile bu analizden istatistiksel kesinlik beklemek doğru değildir. Temel katkılar şunlardır:

1. **SHAP görselleştirmesi:** Hangi `pm_*` özelliğinin hangi kohortu ayırt etmede baskın olduğu görülür. Bu, OFT-only pipeline'a ek bilgi katmanı sağlar.
2. **Delta karşılaştırması:** `pm_*` özelliklerini eklemek F1'i belirgin biçimde iyileştirmiyor (n=12'de beklenen durum) — ama SHAP'ta Plus Maze özelliklerinin görünmesi metodolojik olarak değerlidir.
3. **LOGOCV F1 = 0.000 (group):** Model hiç görmediği bir kohortu tahmin edemiyor. Bu n=3/grup ile beklenen bir sonuçtur; sınıf başına eğitim seti 2 örnekten oluşur.

---

## Sınırlılıklar

1. **n=12 (n=3/grup):** Tüm F1 değerleri yüksek varyansa sahiptir; tek bir sıçanın doğru veya yanlış sınıflandırılması F1'i 0.1–0.3 birim değiştirebilir.
2. **LOGOCV'de sınıf başına 2 örnek:** Model eğitimi için yetersiz; F1=0.000 sonuçları aşırı kısıtlı eğitime bağlıdır.
3. **`anxiety_level` LOOCV F1=0.833:** 2 sınıf ve n=12 ile bu değer yanıltıcıdır — LOGOCV'de 0.333'e düşmesi modelin kohort genellemesi yapamadığını gösterir.
4. **Manuel doğrulama yoktur:** Etiketler (`anxiety_level`, `rearing_profile`, `grooming_profile`) otomatik kural tabanlı hesaplamaya dayanmaktadır.

---

## Tezde Kullanım Önerisi

Bu analiz, Bulgular bölümünde ayrı bir alt başlık veya Yöntemler bölümünde "birleşik pipeline" olarak sunulabilir.

**Önerilen altyazı (`combined_shap_ovr_groups.png` için):**

> Şekil X. OFT + Plus Maze birleşik özellik seti ile eğitilen XGBoost modelinin One-vs-Rest SHAP analizi. Her panel bir deney grubunu diğer gruplardan ayırt etmede en katkılı özellikleri göstermektedir. Kırmızı etiketler Plus Maze kaynaklı (`pm_*`) özellikleri işaret etmektedir. Model LOOCV (n=12) ile değerlendirilmiştir.

**Önerilen altyazı (`combined_vs_oft_comparison.png` için):**

> Şekil X. XGBoost modeli LOOCV F1 macro değerleri: OFT-only (28 özellik) ile OFT + Plus Maze combined (46 özellik) karşılaştırması. Δ, combined ve OFT-only F1 farkını göstermektedir. n=12 örneklem büyüklüğü nedeniyle farklar istatistiksel anlamlılık taşımamaktadır.
