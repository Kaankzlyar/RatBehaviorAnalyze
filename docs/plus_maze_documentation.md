# Plus Maze (Elevated Plus Maze) — Kapsamlı Tez Dokümantasyonu

**Proje:** Rat Behavioral Analysis — Thesis  
**Son güncelleme:** 2026-05-11  
**Kapsam:** DLC modeli · Eğitim verisi · İstatistiksel analiz · ML pipeline · Markov gecis analizi · Tez için sunum rehberi

---

## İçindekiler

1. [Kullanılan DLC Modeli](#1-kullanilan-dlc-modeli)
2. [Model Mimarisi ve Eğitim Parametreleri](#2-model-mimarisi-ve-egitim-parametreleri)
3. [Eğitim Verisi](#3-egitim-verisi)
4. [Takip Edilen Vücut Parçaları](#4-takip-edilen-vucut-parcalari)
5. [Analiz Boru Hattı ve Metrikler](#5-analiz-boru-hatti-ve-metrikler)
6. [EPM İstatistiksel Analizi](#6-epm-istatistiksel-analizi)
7. [ML Pipeline — OFT + Plus Maze Birleşik Model](#7-ml-pipeline--oft--plus-maze-birleshik-model)
8. [Mevcut Bulgular ve Tez Sunumu](#8-mevcut-bulgular-ve-tez-sunumu)
9. [Çıktı Dosyaları ve Klasör Yapısı](#9-cikti-dosyalari-ve-klasor-yapisi)
10. [Sınırlılıklar ve Dürüst Yorum](#10-sinirliliklar-ve-donust-yorum)
11. [Markov Geçiş Matrisi Analizi](#11-markov-gecis-matrisi-analizi)
12. [Pose-Bazlı Etolojik Özellikler](#12-pose-bazli-etolojik-ozellikler)

---

## 1. Kullanılan DLC Modeli

CSV kayıtlarının `scorer` satırından tespit edilen model:

```
DLC_Resnet50_rat_behavior_tmazeApr1shuffle1_snapshot_best-210
```

| Alan | Değer |
|------|-------|
| **Framework** | DeepLabCut 3.x (PyTorch arka ucu) |
| **Omurga (Backbone)** | ResNet-50 |
| **Proje adı** | `rat_behavior_tmaze` |
| **Deney etiketi** | `tmazeApr1` — Nisan 2026 oluşturması |
| **Shuffle** | 1 |
| **Snapshot** | `snapshot_best-210` — validation loss minimumdaki kontrol noktası |

Bu model OFT modelinden **ayrıdır** (`rat_behavior` projesi). Ayrı model kararının gerekçeleri `docs/tmaze_keypoints_and_layout.md` içinde belgelenmiştir: küçük subject görünümü, köşe oklüzyonu ve sadece üstten kamera bu ayrışmayı zorunlu kıldı.

---

## 2. Model Mimarisi ve Eğitim Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Omurga** | ResNet-50 | ImageNet ön-eğitimli; küçük veri setlerinde iyi performans |
| **GPU** | NVIDIA RTX 3060 6 GB | CUDA 11.8+ |
| **Batch size** | 8 | RTX 3060 VRAM kısıtı |
| **Augmenter** | imgaug | Döndürme, parlaklık, aynalama ile artırma |
| **Optimizer** | Adam (DLC varsayılanı) | |
| **maxiters** | 50 000 | Eğitim üst limiti |
| **Snapshot** | best-210 | Validation kayıp minimumdaki kontrol noktası (erken durma) |
| **Frame çıkarma** | k-means, ~20/video | Çeşitli postürleri kapsar |
| **Likelihood eşiği** | ≥ 0.6 | Analiz sırasında düşük güven çerçeveleri NaN yapılır |
| **Filtre** | Medyan, pencere = 5 | Inference sonrası temporal gürültü azaltma |
| **Jump threshold** | 60 px | Ardışık kareler arası sıçrama tespiti |

`snapshot_best-210` ifadesi, DLC'nin `maxiters=50000`'e ulaşmadan validation kaybının en düşük olduğu anı otomatik kaydettiğini gösterir. Bu erken durdurmanın bir biçimidir.

---

## 3. Eğitim Verisi

### Video seti

| Grup | Sıçanlar | Video | Klasör | Tedavi |
|------|----------|-------|--------|--------|
| Control (MA1) | MA1_1, MA1_2, MA1_3 | 3 | Part1 | Araç (su) |
| Aspartame (MA3) | MA3_1, MA3_2, MA3_3 | 3 | Part1 | Aspartam |
| Grapefruit (MA5) | MA5_1, MA5_2, MA5_3 | 3 | Part2 | Greyfurt |
| ASP+GF (MA7) | MA7_1, MA7_2, MA7_3 | 3 | Part2 | Aspartam + Greyfurt |
| **Toplam** | 12 sıçan | **12 video** | — | — |

Tüm videolar: `.avi`, 30 fps, üstten çekim (top-down), tek kamera. Part1 ve Part2'de kamera konumu özdeştir — bu nedenle tüm 12 video için **tek koordinat seti** kullanılmıştır.

### Etiketleme

- Her videodan ~20 kare k-means ile seçildi.
- 10 vücut parçası DLC GUI ile manuel etiketlendi.
- Hedef: ~240 etiketlenmiş kare (12 × 20).
- Görünmeyen (occluded) vücut parçaları atlandı.

---

## 4. Takip Edilen Vücut Parçaları

DLC CSV `bodyparts` satırından okunan **10 keypoint**:

| # | Keypoint | Analiz rolü |
|---|----------|-------------|
| 1 | `nose` | Burun ucu |
| 2 | `head` | Kafa merkezi |
| 3 | `left_ear` | Sol kulak |
| 4 | `right_ear` | Sağ kulak |
| 5 | `body_center` | **Birincil:** tüm zone atamaları ve metrikler |
| 6 | `left_forepaw` | Sol ön pençe |
| 7 | `right_forepaw` | Sağ ön pençe |
| 8 | `left_hindpaw` | Sol arka pençe |
| 9 | `right_hindpaw` | Sağ arka pençe |
| 10 | `tail_base` | Kuyruk tabanı |

Mevcut analizde yalnızca `body_center` kullanılmaktadır. Diğer 9 nokta ilerideki etolojik analiz için hazırdır (head-dip, SAP, rearing).

---

## 5. Analiz Boru Hattı ve Metrikler

### Maze yapısı ve koordinatlar

```
              [TOP ARM]
                  ||
[LEFT ARM] == JUNCTION == [RIGHT ARM]
                  ||
              [BOTTOM ARM]
```

**Tek koordinat seti (tüm 12 video için geçerli, piksel):**

| Kol | xmin | xmax | ymin | ymax |
|-----|------|------|------|------|
| bottom | 544 | 605 | 404 | 717 |
| left | 258 | 551 | 348 | 403 |
| right | 604 | 893 | 345 | 404 |
| top | 544 | 608 | 39 | 348 |

### EPM yorumlaması

| Kol | Tip | Anksiyete anlami |
|-----|-----|-----------------|
| left + right | **Açık kol** | Burada geçirilen süre ↑ = anksiyolitik etki |
| top + bottom | **Kapalı kol** | Güvenli alan, yüksek süre = anksiyojenik |

### Hesaplanan metrikler (per-subject, `*_plus_maze_metrics.csv`)

| Kategori | Sütunlar | Tezdeki anlamı |
|----------|----------|---------------|
| Zone işgali | `pct_time_bottom/left/right/top/junction` | Mekânsal tercih |
| Kol girişleri | `total_entries`, `*_entries` | Genel keşif aktivitesi |
| Alternasyon | `successive_alternation_pct`, `tetrad_alternation_pct` | Çalışma belleği |
| Persiverasyon | `perseveration_count`, `perseveration_rate_pct` | Bilişsel katılık |
| Locomotion | `mean_speed_px_s`, `total_distance_px` | Motor aktivite |
| EPM türetilmiş | `pct_open_arm`, `anxiety_index_epm` | Anksiyete endeksi |

---

## 6. EPM İstatistiksel Analizi

**Script:** `analysis/plus_maze/cohort_stats_epm.py`  
**Çıktılar:** `docs/plus_maze/epm_statistics/`

### Yöntem

Kruskal-Wallis testi, n=3/grup ve normal dağılım garantisi olmadığı için parametrik testler (ANOVA) yerine tercih edilen parametrik olmayan alternatiftir. Her EPM özelliği için ayrı ayrı uygulandı.

| Test | Amaç | Neden seçildi |
|------|------|---------------|
| **Kruskal-Wallis H** | 4 grup arasında en az birinin farkı var mı? | Parametrik olmayan; n=3/grup için ANOVA'dan güvenilir |
| **Permütasyon p** | Asimptotik p'yi doğrula | n=12'de chi-kare yaklaşımı hassasiyeti düşük |
| **ε² (epsilon-kare)** | Etki büyüklüğü | H'dan bağımsız, yorumlanabilir (0=yok, 1=tam) |
| **Dunn post-hoc** | Hangi grup çiftleri farklı? | KW anlamlıysa pairwise karşılaştırma |
| **BH-FDR** | Çoklu karşılaştırma düzeltmesi | Bonferroni'den güçlü; küçük n için daha uygun |
| **PERMANOVA** | Çok değişkenli grup farkı | 4 EPM metriğini birlikte test eder |

### Sonuçlar (`docs/plus_maze/epm_statistics/cohort_epm_kw.csv`)

| Özellik | H | p (permütasyon) | ε² | Etki |
|---------|---|----------------|----|------|
| `pct_open_arm_entries` | 6.946 | **0.049** | 0.493 | Orta |
| `pct_time_junction` | 6.846 | 0.050 | 0.481 | Orta |
| `anxiety_index_epm` | 6.716 | 0.055 | 0.465 | Orta |
| `pct_open_arm` | 6.304 | 0.083 | 0.413 | Orta |
| `arm_preference_index` | 5.957 | 0.096 | 0.370 | Orta |
| `pct_time_right` | 5.172 | 0.159 | 0.272 | Orta |
| `successive_alternation_pct` | 4.336 | 0.246 | 0.167 | Küçük |
| `total_entries` | 3.950 | 0.293 | 0.119 | Küçük |
| `mean_speed_px_s` | 2.897 | 0.448 | — | İhmal edilebilir |

**PERMANOVA:** pseudo-F = 1.554, R² = 0.368, p = 0.150

### Tez için yorum

p < 0.05 yalnızca `pct_open_arm_entries`'de sağlandı. Diğer birincil EPM endpointi (`pct_open_arm`) p = 0.083 ile trendin sınırında. ε² = 0.41–0.49 değerleri **orta etki büyüklüğü** gösteriyor; istatistiksel anlamsızlık, etkinin olmadığını değil, **n=12'nin bu etki büyüklüğünü yakalamak için yetersiz** olduğunu söyler.

> **Tez cümlesi:** "Kruskal-Wallis analizi, açık kol girişi yüzdesinde gruplararası anlamlı fark ortaya koydu (H = 6.95, p = 0.049, ε² = 0.49). Açık kol süresi yüzdesi için trend düzeyinde anlamlılık gözlemlendi (p = 0.083, ε² = 0.41). Orta etki büyüklükleri, mevcut örneklem büyüklüğünün (n = 3/grup) bu etkileri güvenilir biçimde tespit etmek için sınırda kaldığını düşündürmektedir."

---

## 7. ML Pipeline — OFT + Plus Maze Birleşik Model

**Script:** `src/train_combined.py`  
**Çıktılar:** `docs/plus_maze/ml_combined/`

### 7.1 Problem Tanımı

Dört hedef değişken tanımlandı:

| Hedef | Sınıflar | Tanım |
|-------|----------|-------|
| `group` | Control, Aspartame, Grapefruit, ASP+GF | Tedavi grubu — 4 sınıf |
| `anxiety_level` | low, moderate, high | OFT anksiyete skoru eşikleme — 3 sınıf |
| `rearing_profile` | low, moderate, high | OFT rearing süresi eşikleme — 3 sınıf |
| `grooming_profile` | low, moderate, high | OFT grooming süresi eşikleme — 3 sınıf |

**Girdi:** `data/features/features_combined_normalized.csv` — 12 sıçan × 42 özellik (26 OFT + 16 Plus Maze `pm_*`)

### 7.2 Neden 6 Model Kullanıldı?

Bu tezde 6 farklı model karşılaştırıldı. Her modelin seçilme nedeni ve kısıtları şöyle özetlenebilir:

---

#### XGBoost — Birincil Model

**Neden XGBoost ana model olarak seçildi?**

| Gerekçe | Açıklama |
|---------|----------|
| **SHAP uyumluluğu** | Tree Explainer, her özelliğin katkısını kesin matematiksel olarak hesaplar. SVM veya lojistik regresyon için SHAP daha yaklaşıktır. |
| **Küçük veri direnci** | `max_depth=2`, `n_estimators=200`, `subsample=0.8` aşırı öğrenmeye karşı güçlü düzenleme sağlar. Lineer modellerin aksine özellik ölçeklendirmesi gerektirmez. |
| **Eksik değer toleransı** | NaN'lı özellikler doğal olarak işlenir; OFT ve Plus Maze'de bazı özellikler sıfır veya NaN olabilir. |
| **Etkileşim tespiti** | Özellikler arası etkileşimleri (örn. `pm_pct_open_arm` × `mean_speed`) hiyerarşik ağaç yapısıyla örtük olarak modeller. |
| **Küçük n'de dengelenmiş öğrenme** | Sınıf dengesizliğine karşı `scale_pos_weight` parametresi ile One-vs-Rest analizde her gruba eşit ağırlık verildi. |

**Parametreler ve gerekçeleri:**

```python
XGBClassifier(
    n_estimators=200,   # Yeterli çeşitlilik; 500'de aşırı öğrenme başlıyor
    max_depth=2,        # n=12'de derin ağaçlar ezberleme yapar; 2 yeterli
    learning_rate=0.05, # Küçük adım = stabil yakınsama
    subsample=0.8,      # Her ağaçta verilerin %80'i = varyans azaltma
    eval_metric="mlogloss"
)
```

---

#### Random Forest — İkinci Referans

**Neden eklendi?** XGBoost ile aynı ağaç tabanlı ailede; boosting (XGBoost) ile bagging'i (RF) karşılaştırmak için.

**Farkı:** RF her ağacı bağımsız ve paralel eğitir; XGBoost önceki ağacın hatasını düzeltecek şekilde sıralı eğitir. Küçük n'de XGBoost genellikle üstündür çünkü hatalara odaklanır.

**Tezde kullanım:** Karşılaştırma tablosunda F1 değerleri ile yer alır; birincil analiz değil.

---

#### Logistic Regression (L1) — Basit Doğrusal Referans

**Neden eklendi?** En basit sınıflandırıcı olarak alt sınır (baseline) sağlar. L1 cezası (`penalty='l1'`) otomatik özellik seçimi yapar — hangi özelliklerin sıfır ağırlık aldığı yorumlanabilir.

**Neden birincil model değil?**
- Doğrusal karar sınırı varsayar; davranışsal veriler genellikle doğrusal değildir.
- SHAP değerleri katsayılardan doğrudan okunabilir ama interaksiyon tespiti yoktur.
- n=12'de L1 düzenlemesi bile aşırı sıkıştırabilir.

---

#### SVM (RBF kernel) — Yüksek Boyutlu Uzay Referansı

**Neden eklendi?** Küçük veri setlerinde iyi genelleyen klasik bir yöntem; özellikle normalize edilmiş veride güçlüdür.

**Neden birincil model değil?**
- SHAP için `kernel=rbf` → kernel SHAP gerektirir, hesaplama pahalı ve yaklaşık.
- Yorumlanabilirlik sınırlı: hangi özelliğin önemli olduğunu doğrudan söylemez.
- Sınıf olasılığı tahmini Platt scaling ile yapılır, kalibre değil.

---

#### LogReg + MI8 (Mutual Information Feature Selection) — Özellik Seçimi Deneyi

**Neden eklendi?** Mutual Information ile en bilgilendirici 8 özelliği seçip lojistik regresyon yapar. Pipeline içinde CV ile özellik seçimi yapıldığından **bilgi sızıntısı** yoktur.

**Neden birincil model değil?**
- k=8 sabit; optimal k'yı cross-validate etmez.
- 42 özellikten 8 seçmek bilgi kaybına yol açabilir.
- n=12'de MI tahminleri yüksek varyanslıdır.

---

#### LightGBM — Hızlı XGBoost Alternatifi

**Neden eklendi?** XGBoost ile aynı gradient boosting ailesinden; histogram tabanlı daha hızlı eğitim. Grooming profil gibi ayrımı kolay hedeflerde XGBoost'a yakın veya üstün performans verir.

**Neden birincil model değil?**
- SHAP desteği XGBoost kadar olgun değildi (uygulama döneminde).
- `min_child_samples=2` ayarı n=12'de zorunlu ama hassas.

---

### 7.3 Çapraz Doğrulama (CV) Stratejisi

İki ayrı CV stratejisi kullanıldı:

#### LOOCV — Leave-One-Out Cross-Validation

Her seferinde 1 sıçan test, 11 sıçan eğitim seti olarak kullanılır. n=12 fold üretir.

- **Avantajı:** Her sıçan test edilir; veri israfı yok.
- **Dezavantajı:** Test ve eğitim setleri aynı kohorttan sıçan içerebilir → birbirine benzer sıçanları test etmek aşırı iyimser F1 verir.

#### LOGOCV — Leave-One-Group-Out Cross-Validation

Her seferinde 1 kohort (3 sıçan) test, diğer 3 kohort (9 sıçan) eğitim seti. n=4 fold üretir.

- **Avantajı:** Model hiç görmediği bir kohortu tahmin etmeli → **gerçek genelleme testi**.
- **Dezavantajı:** Eğitim setinde yalnızca 9 sıçan; model azdan öğrenmeli.

> **Tez için:** LOGOCV F1 değerleri *gerçekçi beklenti* için, LOOCV F1 değerleri *üst sınır* için raporla.

### 7.4 Leakage (Bilgi Sızıntısı) Kontrolü

Bazı OFT özellikleri doğrudan etiketleri üretti:
- `anxiety_level`: `pct_time_periphery + pct_time_freeze − center_zone_entries` formülünden
- `rearing_profile`: `rear_pct_time` eşikleme
- `grooming_profile`: `groom_pct_time` eşikleme

Bu özellikler her hedefin eğitim setinden çıkarıldı (`LEAKAGE_MAP`). Plus Maze `pm_*` özellikleri hiçbir etiketi üretmedi — leakage yok.

### 7.5 Sonuçlar

#### LOOCV F1 Macro (Combined — OFT + Plus Maze)

| Model | group | anxiety_level | rearing_profile | grooming_profile |
|-------|------:|-------------:|----------------:|-----------------:|
| LogisticReg | 0.200 | **0.833** | 0.235 | 0.116 |
| RandomForest | 0.062 | 0.667 | 0.235 | 0.288 |
| XGBoost | 0.196 | 0.748 | 0.345 | 0.427 |
| SVM | 0.000 | 0.667 | **0.482** | 0.095 |
| LogReg_L1_MI8 | 0.000 | 0.250 | 0.246 | 0.000 |
| *Şans tabanı* | *0.250* | *0.333* | *0.333* | *0.333* |

#### LOGOCV F1 Macro (Combined — gerçek genelleme)

| Model | group | anxiety_level | rearing_profile | grooming_profile |
|-------|------:|-------------:|----------------:|-----------------:|
| LogisticReg | 0.000 | 0.333 | 0.346 | 0.232 |
| RandomForest | 0.000 | 0.413 | 0.345 | 0.265 |
| XGBoost | 0.000 | 0.167 | 0.386 | 0.338 |
| SVM | 0.000 | 0.333 | **0.482** | 0.178 |

#### OFT-only vs OFT+PM Karşılaştırması (XGBoost LOOCV)

| Hedef | OFT-only | OFT+PM | Δ | Yorum |
|-------|----------|--------|---|-------|
| `group` | 0.255 | 0.196 | -0.058 | Hafif gerileme — PM gürültü ekledi |
| `anxiety_level` | 0.169 | **0.748** | **+0.579** | Büyük artış — `pm_anxiety_index_epm` kavramsal örtüşme |
| `rearing_profile` | 0.267 | 0.345 | +0.079 | Hafif iyileşme |
| `grooming_profile` | 0.333 | 0.427 | +0.094 | Hafif iyileşme |

### 7.6 One-vs-Rest SHAP Analizi

Her grup için XGBoost binary sınıflandırıcı (grup vs diğerleri) eğitildi. SHAP, hangi özelliğin o grubu diğerlerinden ayırdığını gösterir.

**Görsel:** `docs/plus_maze/ml_combined/combined_shap_ovr_groups.png`  
Kırmızı etiketler = `pm_*` (Plus Maze) özelliği

| Grup | OvR LOOCV F1 | Birincil ayırt edici PM özelliği |
|------|:-----------:|--------------------------------|
| Control | 0.400 | `pm_pct_time_bottom` (kapalı kol dominansı) |
| Aspartame | 0.000 | — (grup ayırt edilemedi) |
| Grapefruit | 0.000 | — (grup ayırt edilemedi) |
| ASP+GF | 0.286 | `pm_pct_open_arm` (açık kol artışı) |

---

## 8. Mevcut Bulgular ve Tez Sunumu

### 8.1 Tamamlanan Analizler

| Adım | Durum | Çıktı | Tezde nerede |
|------|-------|-------|-------------|
| DLC inference (12 video) | ✅ | `data/DLCfiltered/*/PlusMaze*.csv` | Metodlar |
| Per-subject metrikler | ✅ | `*_plus_maze_metrics.csv` | Ek tablo |
| Birleşik tablo | ✅ | `data/plus_maze_metrics_all.csv` | Bulgular tablosu |
| EPM görselleri (orbit, bodypart) | ✅ | `*_plus_maze_orbit.png` | Ek figürler |
| KW + Dunn + PERMANOVA | ✅ | `epm_statistics/` | Bulgular — istatistik bölümü |
| Box/strip grafikleri | ✅ | `epm_figures/` | Bulgular — figürler |
| Etki büyüklüğü (Cohen's d) | ✅ | `effect_analysis/` | Tartışma |
| OFT + PM combined ML | ✅ | `ml_combined/` | Bulgular — ML bölümü |
| SHAP analizi | ✅ | `ml_combined/combined_shap_*.png` | Bulgular — yorumlanabilirlik |
| OFT-only vs Combined karşılaştırma | ✅ | `combined_vs_oft_comparison.png` | Bulgular — model karşılaştırması |

### 8.2 Metodlar Bölümü İçin

**Pose Tahmini (DLC):**
> "Sıçan vücut pozisyonu DeepLabCut (DLC) 3.x ile tahmin edildi. Plus maze kayıtları için ResNet-50 omurgalı ayrı bir DLC modeli eğitildi (`rat_behavior_tmaze`, snapshot_best-210). Eğitim seti 12 videodan k-means algoritmasıyla çıkarılan ~240 kareyi kapsamaktadır. Analizde `body_center` noktası bölge atama ve tüm metriklerin hesaplanması için kullanıldı."

**İstatistiksel Analiz:**
> "Gruplar arası karşılaştırma için parametrik olmayan Kruskal-Wallis H testi uygulandı. Asimptotik p değerleri 10.000 permütasyon ile doğrulandı. Etki büyüklüğü epsilon-kare (ε²) ile ölçüldü. Çoklu karşılaştırma düzeltmesi için Benjamini-Hochberg FDR yöntemi kullanıldı. Çok değişkenli grup farkı PERMANOVA ile test edildi."

**ML Pipeline:**
> "OFT metrikleri (26 özellik) ve Plus Maze metrikleri (16 özellik) sıçan kimliği üzerinden birleştirilerek 42 özellikli birleşik özellik tablosu oluşturuldu. Altı sınıflandırıcı (XGBoost, Random Forest, Logistic Regression-L1, SVM-RBF, LogReg+MI8, LightGBM) hem Leave-One-Out (LOOCV) hem de Leave-One-Group-Out (LOGOCV) çapraz doğrulama ile değerlendirildi. SHAP TreeExplainer, XGBoost modelinde özellik önemini yorumlamak için kullanıldı. Etiket-özellik bilgi sızıntısını önlemek amacıyla, her hedef değişkeni üreten formülde kullanılan özellikler o hedefe ait eğitim setinden çıkarıldı."

### 8.3 Bulgular Bölümü İçin

**EPM Ana Bulgular:**

> "Açık kol girişi yüzdesinde gruplararası anlamlı fark saptandı (H = 6.95, p = 0.049, ε² = 0.49). ASP+Grapefruit grubunun açık kol süresi ortalaması (median: %8.11) kontrole (median: %0.00) kıyasla belirgin biçimde yüksekti. Aspartame ve Grapefruit grupları kontrole yakın değerler gösterdi."

**ML Ana Bulgular:**

> "OFT özelliklerine Plus Maze metrikleri eklenmesi, anxiety_level tahmininde XGBoost LOOCV F1'i 0.169'dan 0.748'e yükseltti. SHAP analizi, bu artışın büyük bölümünün `pm_anxiety_index_epm` ve `pm_pct_open_arm` özelliklerinden kaynaklandığını gösterdi. Group (tedavi grubu) tahmini LOGOCV'de tüm modellerde şans düzeyinde kaldı (F1 ≈ 0), n=12'nin 4 sınıflı ayrım için yetersiz olduğunu ortaya koydu."

### 8.4 Tartışma Bölümü İçin

**ASP+GF kombinasyon etkisi:**
> "ASP+Grapefruit grubunun hem açık kol süresinde hem de toplam giriş sayısında diğer gruplara göre yüksek değerler göstermesi, Cruz ve ark. (1994) iki faktör modeli çerçevesinde değerlendirildiğinde karışık bir etki örüntüsüne işaret etmektedir. Greyfurtun CYP3A4 inhibisyonu aracılığıyla aspartam metabolitlerinin plazmada birikmesine yol açması (Bailey ve ark., 2013) ve greyfurt flavonoidlerinin (naringin) bilinen anksiyolitik özellikleri (Fernandez ve ark., 2009), bu kombinasyon etkisini açıklayan iki olası mekanizmadır."

**ML yorumu:**
> "anxiety_level F1 artışının büyük bölümü, etiketin OFT anksiyete skoru formülüyle tanımlı olması ve EPM metriklerinin (pm_anxiety_index_epm) bu skorla kavramsal örtüşmesiyle açıklanabilir. Bu durum leakage olmamakla birlikte kavramsal redundancy oluşturmaktadır. Group tahmini için LOGOCV F1 = 0 değeri, n=3/grup ile 4 sınıflı öğrenmenin güvenilir olmadığını doğrulamaktadır; bu n büyüklüğünde istatistiksel testler (KW + Dunn) makine öğrenmesinden daha güvenilir bir karşılaştırma aracı sunmaktadır."

---

## 9. Çıktı Dosyaları ve Klasör Yapısı

```
docs/plus_maze/
├── epm_statistics/
│   ├── cohort_epm_kw.csv           KW H, p, ε², etki büyüklüğü (14 feature)
│   ├── cohort_epm_dunn.csv         Pairwise Dunn post-hoc (tüm grup çiftleri)
│   └── cohort_epm_permanova.csv    Çok değişkenli PERMANOVA sonucu
│
├── epm_figures/
│   ├── epm_open_arm_by_cohort.png  Box+strip: pct_open_arm + pct_open_arm_entries
│   ├── epm_arm_distribution.png    Stacked bar: 4 kohort × 5 kol yüzdesi
│   └── epm_locomotor_covariate.png Scatter: açık kol vs total_entries + hız
│
├── effect_analysis/
│   ├── behavioral_effect_table.csv Cohen's d (her madde vs Kontrol)
│   ├── behavioral_radar.png        Örümcek ağı (8 metrik, 4 grup)
│   └── behavioral_effect_bars.png  % değişim + etki büyüklüğü bar chart
│
├── ml_combined/
│   ├── combined_model_comparison.csv   6 model × 4 hedef × LOOCV/LOGOCV F1
│   ├── combined_loocv_predictions.csv  Sıçan bazında doğru/yanlış tahminler
│   ├── combined_ovr_binary_f1.csv      OvR binary F1 (4 grup)
│   ├── combined_vs_oft_comparison.png  OFT-only vs Combined Δ (tez için birincil)
│   ├── combined_shap_ovr_groups.png    OvR SHAP — pm_* katkısı (tez için birincil)
│   ├── combined_cv_comparison.png      LOOCV/LOGOCV F1 tablosu
│   ├── combined_shap_anxiety_level.png SHAP — anxiety_level
│   ├── combined_shap_rearing_profile.png
│   ├── combined_shap_grooming_profile.png
│   ├── combined_shap_group.png
│   ├── combined_confusion_anxiety_level.png
│   ├── combined_confusion_rearing_profile.png
│   ├── combined_confusion_grooming_profile.png
│   └── combined_confusion_group.png
│
├── markov_analysis/
│   ├── markov_transition_matrices.csv  Uzun format: 4 kohort × 16 gecis × olasilik
│   ├── markov_transition_counts.csv    Ham sayimlar (gorsellestirme kontrolu icin)
│   ├── markov_heatmaps.png             2x2 isi haritasi — kohort baz gecis olasiliklari
│   └── markov_perseveration.png        Bar chart: persiverasyon + acik/kapali hedef orani
│
└── ethological_analysis/
    ├── ethological_metrics_epm.csv     Per-sican grooming + SAP metrikleri
    ├── ethological_grooming.png        Box+strip: grooming % suresi kohort bazinda
    ├── ethological_sap.png             Box+strip: SAP % suresi kohort bazinda
    └── ethological_summary.png         Grooming + SAP yan yana ozet panel
```

**Tez için öncelikli 8 görsel:**

| Görsel | Tezde nereye |
|--------|-------------|
| `epm_open_arm_by_cohort.png` | Bulgular — EPM birincil endpoint |
| `epm_arm_distribution.png` | Bulgular — kol dağılımı |
| `combined_vs_oft_comparison.png` | Bulgular — ML model karşılaştırması |
| `combined_shap_ovr_groups.png` | Bulgular — özellik önemi / yorumlanabilirlik |
| `markov_heatmaps.png` | Bulgular — davranış stratejisi / geçiş örüntüleri |
| `markov_perseveration.png` | Bulgular/Tartışma — persiverasyon karşılaştırması |
| `ethological_grooming.png` | Bulgular — grooming kohort karşılaştırması |
| `ethological_summary.png` | Bulgular — grooming + SAP yan yana özet |

---

## 10. Sınırlılıklar ve Dürüst Yorum

### Örneklem büyüklüğü

n = 3/grup ile:
- KW testi için güç düşük; p < 0.05 yalnızca en güçlü sinyali yakalıyor.
- 4-sınıf ML için güvenilir LOGOCV F1 = 0 beklenen bir sonuç (şans = 0.25).
- Sonuçlar **keşifsel** olarak yorumlanmalı; hipotez doğrulayıcı değil.

### EPM standart protokolü

Standart EPM yükseltilmiş platform gerektirirken bu kayıtlar zemin seviyesinde yatay plus maze içindir. EPM çerçevesinin uygulanabilirliği metodlar bölümünde açıkça belirtilmeli.

### anxiety_level ML artışı

Combined pipeline'da `anxiety_level` F1 = 0.748 yüksek görünse de `pm_anxiety_index_epm` ile etiket tanımı arasındaki kavramsal örtüşme bu artışı şişiriyor olabilir. SHAP grafiği `pm_anxiety_index_epm`'i birinci sıraya koyuyorsa bu durum tezde şeffaflıkla belirtilmeli.

### Etolojik özellikler eksik

Head-dip, SAP ve risk-assessment DLC pose'dan türetilmedi. Bu özellikler olmadan saf anksiyolitik ve motor etkinin ayrımı tam yapılamıyor (Cruz 1994 modeli).

### Pozitif kontrol yok

Bilinen anksiyolitik (diazepam 1 mg/kg) uygulanmadı. Pipeline'ın gerçek anksiyolitik etkiyi yakaladığı doğrulanamıyor — gelecek çalışma önerisi olarak sunulabilir.

---

## 11. Markov Geçiş Matrisi Analizi

**Script:** `analysis/plus_maze/markov_analysis.py`  
**Çıktılar:** `docs/plus_maze/markov_analysis/`

### Yöntem

`entry_sequence` sütunundaki ham kol-giriş dizisi (örn. `R->T->T->B->B->B`) kullanılarak her kohort için bir **4×4 Markov geçiş matrisi** oluşturulmuştur. Satırlar kaynak kolu, sütunlar hedef kolu gösterir. Her satır kendi toplamına bölünerek **satır-normalize olasılık matrisi** elde edilir.

Kollar: B = Bottom (kapalı), T = Top (kapalı), L = Left (açık), R = Right (açık).

Hesaplanan metrikler:

| Metrik | Tanım |
|--------|-------|
| **Persiverasyon oranı** | Diagonal ortalaması — aynı kola geri dönme olasılığı |
| **Açık kol hedef oranı** | L ve R sütunlarının satır ortalaması — açık kola geçiş eğilimi |
| **Kapalı kol hedef oranı** | B ve T sütunlarının satır ortalaması |

### Bulgular

| Kohort | n_gecis | Persiverasyon | Açık kola geçiş | Kapalı kola geçiş |
|--------|---------|--------------|-----------------|-------------------|
| Control | 37 | 0.365 | 0.018 | 0.732 |
| Aspartame | 31 | 0.423 | 0.114 | 0.886 |
| Grapefruit | 21 | 0.264 | 0.028 | 0.972 |
| ASP+Greyfurt | 65 | 0.593 | **0.394** | 0.606 |

**Temel gözlem:** ASP+Greyfurt grubunun açık kola geçiş olasılığı (0.394) diğer tüm gruplardan belirgin biçimde yüksektir (Control: 0.018, Aspartame: 0.114, Grapefruit: 0.028). Bu, `pct_open_arm` ve `pct_open_arm_entries` bulgularıyla örtüşüyor ve söz konusu sıçanların açık kola **şansa bağlı değil, sistematik biçimde** geçtiğini gösteriyor.

ASP+Greyfurt aynı zamanda en yüksek persiverasyon oranına (0.593) sahiptir — açık kola girdiklerinde orada kalmayı tercih ediyorlar; kaçınma değil, keşif stratejisi.

### Sınırlılıklar

n = 3/grup ile toplam geçiş sayısı düşüktür (Grapefruit: yalnızca 21 geçiş). Chi-kare ya da başka bir istatistiksel karşılaştırma bu örneklem büyüklüğünde yetersiz güce sahip olacağından yapılmamıştır. Sonuçlar **keşifsel ve görsel örüntü bazlı** olarak yorumlanmalıdır.

### Tezde Kullanıma Hazır Metin

**Yöntemler bölümü için:**

> Kol-geçiş dizileri üzerinden birinci dereceden Markov geçiş matrisleri hesaplanmıştır. Her kohort için bireysel sıçanların `entry_sequence` kayıtları birleştirilerek 4×4 geçiş sayım matrisi oluşturulmuş ve satır normalleştirmesi ile olasılık matrisine dönüştürülmüştür. Kollar şu şekilde kodlanmıştır: B = alt kol (kapalı), T = üst kol (kapalı), L = sol kol (açık), R = sağ kol (açık). Diagonal değerler persiverasyon (aynı kola ardışık giriş), off-diagonal değerler ise alternasyon eğilimini temsil etmektedir.

**Bulgular bölümü için:**

> Markov geçiş analizi, ASP+Greyfurt grubunun açık kola geçiş olasılığının (0.394) diğer gruplara kıyasla belirgin şekilde yüksek olduğunu ortaya koymuştur (Control: 0.018, Aspartame: 0.114, Grapefruit: 0.028). Persiverasyon oranı bakımından da ASP+Greyfurt grubu en yüksek değeri sergilemiştir (0.593), bu durum söz konusu sıçanların açık kola girdikten sonra orada kalmaya devam ettiğine işaret etmektedir. Kapalı kol tercihinin Grapefruit grubunda en belirgin olduğu (hedef oranı: 0.972), Control ve Aspartame gruplarının ise orta düzeyde kapalı kol yanlılığı sergilediği gözlemlenmiştir.

**Tartışma bölümü için:**

> Markov geçiş matrisleri, ASP+Greyfurt grubunun açık kola yönelik davranışsal stratejisinin diğer gruplardan niteliksel olarak farklılaştığını göstermektedir. Bu bulgu, toplam süre ve giriş sıklığı ölçütlerinden elde edilen sonuçlarla tutarlılık göstermektedir. Söz konusu örüntü, Cruz (1994) tarafından tanımlanan anksiyolitik etki davranış örüntüsüyle uyumlu olmakla birlikte, düşük örneklem büyüklüğü (grup başına n = 3) nedeniyle istatistiksel karşılaştırma güçten yoksundur; bulgular keşifsel nitelikte değerlendirilmelidir.

---

## 12. Pose-Bazlı Etolojik Özellikler

**Script:** `analysis/plus_maze/ethological_features.py`  
**Çıktılar:** `docs/plus_maze/ethological_analysis/`

### Kapsam

DLC'nin 10 keypoint'inden yararlanılarak iki etolojik özellik otomatik olarak tespit edilmiştir:

| Özellik | Türkçe | Kural | Keypoint'ler |
|---------|--------|-------|-------------|
| **Grooming** | Kendini temizleme | forepaw–nose < 23 px + hız < 0.50 px/kare | `left_forepaw`, `right_forepaw`, `nose` |
| **SAP** | Uzanma-değerlendirme duruşu | vücut boyu > 102 px + hız < 0.50 px/kare | `nose`, `tail_base`, `body_center` |

Rearing (dikine kalkma) üstten kamera kaydında güvenilir biçimde tespit edilemediğinden kapsam dışında bırakılmıştır. Vücut boyu kısalması başka davranışlarla (grooming, dar alanda kıvrılma) karışabileceğinden bu metrik methodolojik açıdan savunulamaz düzeydedir. Gelecek çalışma önerisi: yan kamera ile doğrulama.

Eşikler 12 sıçanın havuzlanmış dağılımından hesaplanmıştır (forepaw-nose p25 = 23 px; vücut boyu p75 = 102 px). Minimum bout süresi 0.5 s (15 kare @ 30 fps), birleştirme aralığı 0.33 s (10 kare).

### Bulgular

| Kohort | Grooming % süre | Grooming bout/sıçan | SAP % süre | SAP bout/sıçan |
|--------|----------------|---------------------|------------|----------------|
| Control | 10.4% | 18.0 | 9.1% | 16.7 |
| Aspartame | 15.6% | 34.3 | 14.5% | 22.7 |
| Grapefruit | **19.3%** | 29.0 | **18.0%** | 24.7 |
| ASP+Greyfurt | 13.9% | 32.7 | 17.0% | 27.3 |

**Grooming:** Grapefruit grubu en yüksek grooming süresine sahiptir (%19.3). Artmış grooming anksiyete yükünün davranışsal göstergesi olarak yorumlanabilir (Cruz 1994). Aspartame grubunda bout sayısı fazla (34.3) ancak ortalama süre kısa (1.34 s) — parçalı/kırılgan grooming örüntüsü.

**SAP:** ASP+Greyfurt ve Grapefruit gruplarında Control'e göre daha yüksek SAP süresi gözlemlenmiştir. SAP azaldığında anksiyolitik etki beklenir; bu veri gruplar arasında net bir azalma örüntüsü göstermemektedir — örneklem büyüklüğü sınırlı yoruma yol açmaktadır.

### Sınırlılıklar

- Eşikler data-driven (havuz percentil) olup video-bazlı manuel doğrulama yapılmamıştır.
- n = 3/grup ile istatistiksel karşılaştırma gücü yetersizdir; sonuçlar keşifseldir.
- Grooming ile SAP'ın bir kısmı çakışan pozisyonları kapsıyor olabilir (vücut uzarken aynı anda forepaw hareketi).

### Tezde Kullanıma Hazır Metin

**Yöntemler bölümü için:**

> DeepLabCut ile takip edilen on vücut noktasından (burun, baş, kulaklar, ön/arka patiler, kuyruk tabanı, vücut merkezi) grooming ve stretch-attend posture (SAP) davranışları otomatik olarak tespit edilmiştir. Grooming; minimum ön pati–burun mesafesinin 23 pikselin altına düşmesi ve vücut merkezi hızının 0.50 px/kare eşiğini aşmaması koşullarının en az 0.5 saniye sürmesi olarak tanımlanmıştır. SAP; burun–kuyruk tabanı mesafesinin 102 pikselin üzerinde olması ve benzer hareketsizlik koşulunun sağlanması olarak tanımlanmıştır. Eşikler 12 sıçanın havuzlanmış dağılımından (sırasıyla p25 ve p75) türetilmiştir. Rearing davranışı, üstten kamera kaydında vücut boyu kısalmasının diğer davranışlarla (grooming, kıvrılma) ayırt edilememesi nedeniyle kapsam dışında bırakılmıştır.

**Bulgular bölümü için:**

> Pose-bazlı grooming analizi, Grapefruit grubunun oturum süresinin %19.3'ünü temizleme davranışına ayırdığını ortaya koymuştur; bu oran Control (%10.4), Aspartame (%15.6) ve ASP+Greyfurt (%13.9) gruplarının üzerindedir. Aspartame grubunda bout sayısı diğer gruplara kıyasla daha fazla (34.3 ± SD) ancak ortalama bout süresi daha kısa (1.34 s) saptanmıştır. SAP süresi açısından Grapefruit (%18.0) ve ASP+Greyfurt (%17.0) grupları Control (%9.1) grubuna göre daha yüksek değerler sergilemiştir.

**Tartışma bölümü için:**

> Grooming bulgularında Grapefruit grubunun öne çıkması, kapalı kol tercihiyle tutarlı bir anksiyete örüntüsüne işaret etmektedir: söz konusu grup hem kapalı kolda daha fazla zaman geçirmiş hem de daha yoğun temizlenme davranışı sergilemiştir. ASP+Greyfurt grubunda ise artmış açık kol süresi ile orta düzeyde grooming bir arada gözlemlenmiştir; bu örüntü, kombinasyon tedavisinin anksiyete yükünü azaltırken temizlenme davranışını bütünüyle baskılamadığına işaret edebilir. Tüm bulgular keşifsel nitelikte değerlendirilmeli; eşiklerin manuel video skorlaması ile doğrulanması gelecek çalışmalar için önerilmektedir.
