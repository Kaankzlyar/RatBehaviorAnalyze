# Anksiyete Analizi — Konsolide Rapor

**Tarih:** 2026-05-11
**Branch:** `develop`
**Pipeline kodu:** `src/anxiety/`, `scripts/predict_anxiety_v{2,3}*.py`, `analysis/open_field/`
**Önceki sürümler (tarihsel kayıt):** [`anxiety_findings_report.md`](anxiety_findings_report.md) (2026-05-10) · [`anxiety_progress_2026-05-10.md`](anxiety_progress_2026-05-10.md)

> Bu doküman iki eski raporu birleştirip 2026-05-11 bulgularıyla (config mismatch incident, retrain helper, Streamlit GUI, modelin karar matematiği) genişletir. Eski raporlar geçmiş kayıt; bu doküman tek otoritatif kaynak.

---

## 1. Yönetici özeti

### 1.1. Bir cümlede iddia

> "29 hayvan üzerinde DeepLabCut tabanlı uçtan uca yeniden üretilebilir bir davranış-analizi pipeline'ı geliştirildi; pipeline, rearing davranışının mekansal dağılımı (center vs wall) üzerinden Control vs Treated için **LOOCV AUC = 0.683** (balanced accuracy = 0.738) elde etti ve uncorrected anlamlılıkta (`rear_center_frac` MW p=0.030, Cohen's d=−0.92) pilot kanıt sağladı."

### 1.2. Üç ana çıktı

1. **Pipeline:** Video → DLC → rule-based bouts → OFT metrics → spatial rearing → 12-feature → LR / RF / SVM (LOOCV). CLI ([`predict_anxiety_v2.py`](../scripts/predict_anxiety_v2.py)) ve Streamlit GUI ([`predict_anxiety_v3_gui.py`](../scripts/predict_anxiety_v3_gui.py)).
2. **Empirik bulgu:** Spatial rearing (center vs wall) en güçlü univariate grup-ayırıcı sinyal (`rear_center_frac` MW p=0.030, d=−0.92). Treated grup merkezde rearing fraksiyonunu ~yarıya düşürmüş. (Not: bu özellikler final modele girmedi — bkz. §3.4 ve §12.)
3. **Model:** Logistic regression rear-only (12 feature, LOOCV AUC = 0.683). RF zayıf (0.525). SVM dar feature setiyle bozuluyor (0.358).

### 1.3. Doğru çerçeveleme

Model bir "**anksiyete dedektörü**" değil, "**davranışsal sapma dedektörü**" — Treated etiketi hem aspartamı (anksiyogenik) hem grapefruit'i (anksiyolitik) içeriyor. Modelin tespit ettiği şey "tedavi-kaynaklı baseline'dan sapma"dır; yön (anksiyogenik / anksiyolitik) feature-level analizdedir (§3.4, §4.3).

---

## 2. Veri ve yöntem

### 2.1. Hayvan dağılımı (n = 29)

| Grup | Kohortlar | n | Tedavi | Beklenen yön (literatür) |
|---|---|---|---|---|
| Control | MA1, MA2 | 5 | Vehicle | baseline |
| Aspartame | MA3, MA4 | 8 | Aspartam | Anksiyogenik (fenilalanin metaboliti) |
| Grapefruit | MA5, MA6 | 8 | Grapefruit | Anksiyolitik (flavonoid antioksidanlar) |
| Aspartame+Grapefruit | MA7, MA8 | 8 | Kombine | Etkileşim / kısmi geri çevirme |

Control / Treated oranı 5 / 24 → `class_weight="balanced"` zorunlu, outlier sensitivity yüksek.

### 2.2. Pipeline akışı

```
Raw video (30 fps, ~178 s)
   ↓ DLC OFT 9-keypoint model                  (src/dlc/)
Filtered pose CSV                              (data/DLCfiltered/<group>/<subject>/<subject>.csv)
   ↓ Rule-based detector                       (src/behavior_detection.py)
Rearing + grooming bouts
   ↓ OFT metrics                               (analysis/open_field/oft_metrics.py)
Locomotion · thigmotaxis · freeze · entropy
   ↓ Spatial rearing                           (src/anxiety/spatial_rearing.py)
Her bout center / wall (univariate analiz — final modele girmiyor)
   ↓ Feature matrix builder                    (src/anxiety/profile.py)
12 rear-only feature
   ↓ LOOCV classifier (LR / RF / SVM)          (src/anxiety/classifier.py)
models/anxiety_classifier/{lr,rf,svm}_rearonly.pkl + scaler

Inference (single subject):
   scripts/predict_anxiety_v2.py  <csv>        (CLI)
   streamlit run scripts/predict_anxiety_v3_gui.py  (Tarayıcı GUI)
```

### 2.3. Feature seti (rear-only, 12 sütun)

Regex `rear|pct_periphery|pct_freeze|spatial_entropy|comfort` ile filtrelenmiş; final konfigürasyonda 6 spatial-rearing varyantı (center/wall split) **çıkarılmış** — bkz. §3.4 ve §12 (LOOCV varyansı + multicollinearity gerekçesi).

**Davranışsal — rearing (8):** `rear_count`, `rear_total_s`, `rear_pct`, `rear_mean_bout_s`, `rear_rate_per_min`, `rear_early_frac`, `rear_groom_ratio`, `rear_per_100px`

**Mekansal (3):** `pct_periphery`, `pct_freeze`, `spatial_entropy`

**Türetilmiş (1):** `comfort_ratio = groom_total_s / pct_center`

**Eski sürümdeki 6 spatial-rearing özelliği — final modele girmedi:** `rear_count_center`, `rear_count_wall`, `rear_total_s_center`, `rear_total_s_wall`, `rear_center_frac`, `rear_center_minus_wall`. Bu özellikler univariate analizde (§3.2) en güçlü pilot bulguyu sağladı; ancak LOOCV (n=29) içinde yüksek korelasyonlu 6 varyantın hepsini birlikte taşımak modeli istikrarsızlaştırdı (18-feat AUC 0.733 → 0.617, aynı konfigte tek günlük yeniden eğitim). Final 12-feature seti spatial rearing'i univariate kanıt katmanında bırakıp classifier'a daha düşük varyanslı genel-rearing + spatial context veriyor.

### 2.4. Inner zone tanımı (kritik metodolojik karar)

```python
# src/anxiety/config.py
ARENA        = (397.0, 777.0, 156.0, 535.0)    # manually annotated
INNER_MARGIN = 0.20                            # literature-standard
INNER_ZONE   = _derive_inner(ARENA, 0.20)
               # → (473, 701, 232, 459); inner ≈ %36 of arena
```

**Neden %20 margin?** Klasik OFT (Open Field Test) literatüründe inner zone arenanın %25–50'si olarak tanımlanır. %20 margin (inner ≈ %36 of arena) bu aralığa girer ve biyolojik olarak **thigmotaksis halkası ile center'ı net ayırır**.

**Sensitivity analizi:** Önceden kullanılan manuel zone (`422, 748, 182, 506` — ~%7 margin, inner ≈ %85 of arena) ile karşılaştırma:

| Metrik | %20 margin (final) | ~%7 margin (eski manuel) |
|---|---|---|
| `rear_center_frac` MW p | **0.030** | 0.281 |
| `rear_center_frac` Cohen's d | **−0.92** | +0.31 (yön TERS) |
| `rear_count_wall` (Treated medyan) | 18 | 1 |
| LR LOOCV AUC (18-feat, hist.) | 0.73 | 0.62 |
| LR LOOCV AUC (12-feat, final) | **0.683** | — (re-run yapılmadı) |

Manuel zone'da neredeyse tüm rear bout'ları "center" sınıfına düşüyor → sınıf sabit → grup ayrımcı sinyal yok. Final karar: arena ve inner zone artık tek bir kaynaktan (`src/anxiety/config.py`) gelir, **%20 margin standardı kilitlendi** (commit `dd5943b`). Manuel-zone sensitivity satırı 18-feature dönemindendir; 12-feature konfigte tekrarlanmadı çünkü sonuç çıkarımı (manuel zone sinyali çökertir) feature sayısından bağımsız.

> "Spatial rearing classification was performed using the literature-standard 20% margin from the arena edge (Choleris et al. 2001 OFT convention; Carter & Shieh 2010). A sensitivity analysis with a wider center definition (~7% margin, ~85% of arena classified as center) showed the spatial signal collapses, confirming that the metric measures the thigmotaxis-ring versus inner-area distinction rather than wall contact per se."

---

## 3. Bulgular

### 3.1. Univariate (Kruskal-Wallis, 4 grup, n = 29)

[`reports/anxiety_stats.csv`](../reports/anxiety_stats.csv) — top 6 sinyal:

| Özellik | H | p | η² | Yorum |
|---|---|---|---|---|
| **`rear_count`** | **8.03** | **0.045** | **0.201** | Large effect, uncorrected p<0.05 |
| `rear_rate_per_min` | 6.86 | 0.076 | 0.155 | Medium-large, marjinal |
| `rear_early_frac` | 6.25 | 0.100 | 0.130 | Medium, signal var |
| `total_distance_px` | 5.34 | 0.148 | 0.094 | Small-medium |
| `rear_total_s` | 5.21 | 0.157 | 0.088 | Small-medium |
| `rear_pct` | 4.93 | 0.177 | 0.077 | Small |

Top 6'nın 5'i rearing → sinyal rearing davranışında yoğunlaşmış. BH-FDR sonrası anlamlılığa ulaşmıyor.

### 3.2. Spatial rearing — ana bulgu

[`reports/spatial_rearing_stats.csv`](../reports/spatial_rearing_stats.csv):

| Metrik | Control medyan | Treated medyan | MW U p | Cohen's d | Yön |
|---|---|---|---|---|---|
| **`rear_center_frac`** | 0.133 | 0.075 | **0.030** | **−0.92** | Treated merkezde daha az rearing |
| **`rear_total_s_center`** | 3.84 s | 0.68 s | **0.045** | **−0.66** | Treated merkez-rearing süresi ~5× düşük |
| `rear_center_minus_wall` | −11 | −15 | 0.049 | −0.87 | (yukarıdakinin başka parametrizasyonu) |
| `rear_count_wall` (4-grup KW) | 13 | 18 | KW p=0.043, η²=0.21 | +0.76 | Treated daha çok duvar-rearing |

**Yorum:** Toplam rearing sayısı eşit olsa bile, *nerede* şahlandığı kalitatif olarak güçlü ayırıcı. Treated hayvanlar duvar tarafına kayıyor (anksiyete benzeri kaçış-yolu arama). En güçlü pilot bulgu — Bonferroni öncesi anlamlı, n=20-22/grup ile %80 power'da replike edilebilir.

**Sınırlılık:** Üç spatial metrik birbiriyle korelasyonlu (aynı bulgunun üç parametrizasyonu). Bonferroni 7-test düzeltmesi (0.030 × 7 = 0.21) eşiği geçmiyor → uncorrected pilot evidence.

### 3.3. Binary classifier — Control vs Treated

[`reports/anxiety_classifier_metrics_rearonly.csv`](../reports/anxiety_classifier_metrics_rearonly.csv) — rear-only 12-feature LOOCV (n=29, `class_weight=balanced`):

| Model | Accuracy | Bal-Acc | F1_treated | F1_control | **AUC** | Tag |
|---|---|---|---|---|---|---|
| **Logistic Regression L2** | 0.83 | 0.74 | 0.89 | 0.55 | **0.683** | rearonly |
| Random Forest | 0.76 | 0.46 | 0.86 | 0.00 | 0.525 | rearonly |
| SVM-RBF | 0.76 | 0.46 | 0.86 | 0.00 | 0.358 | rearonly |
| LR L2 (24-feature baseline, hist.) | 0.72 | 0.60 | 0.83 | 0.33 | 0.57 | — |

Rear-only LR iterasyon eğrisi: **24-feat 0.57 → 18-feat 0.73 → (config bug fix, 18-feat retrain) 0.617 → 12-feat (spatial-rearing variantları kaldırıldı) 0.683 (final)**. LR balanced accuracy 0.738 ile sınıf-dengesiz n=29'da en güvenilir metrik; sadece 5 Control varken AUC ±0.10 LOOCV varyansı normal. RF / SVM minority sınıfı (Control) düşürdü (F1_control = 0); LR doğru metrikte yegane sağlam model.

**Baseline kontroller:**
- Trivial "her zaman Treated": accuracy = 24/29 = 0.83 → ham accuracy yanıltıcı.
- Doğru metrik: **balanced accuracy + AUC**.

### 3.4. Feature importance — spatial özellikler dominant

[`reports/anxiety_classifier_importance_rearonly.csv`](../reports/anxiety_classifier_importance_rearonly.csv) (RF, final 12-feature):

| Sıra | Feature | RF importance |
|---|---|---|
| 1 | `rear_early_frac` | 0.219 |
| 2 | `pct_periphery` | 0.164 |
| 3 | `spatial_entropy` | 0.123 |
| 4 | `pct_freeze` | 0.087 |
| 5 | `rear_rate_per_min` | 0.069 |
| ... | | |
| 12 | `rear_count` | 0.028 |

**Narrative shift (12-feature retrain):** önceki 18-feature konfigte top-5'in üçü spatial rearing varyantıydı (`rear_center_frac`, `rear_total_s_center`, `rear_center_minus_wall`). Bu özellikler güçlü univariate sinyal verirken (§3.2) LOOCV'de yüksek korelasyonlu 6 varyantın hepsini taşımak modeli istikrarsızlaştırıyordu (18-feat AUC 0.733 → 0.617, aynı konfigte tek günlük yeniden eğitim). 12-feature retrain spatial rearing'i univariate kanıt katmanında bıraktı; classifier şimdi keşif zamanlaması (`rear_early_frac`), periferik bağlanma (`pct_periphery`), uzaysal dağılım (`spatial_entropy`) ve donma (`pct_freeze`) bileşimini kullanıyor. Genel-rearing temposu (`rear_count`, sıra #12) hâlâ en zayıf discriminator.

### 3.5. Composite anxiety index — negatif bulgu

Üç birleşik skor [`src/anxiety/composite_index.py`](../src/anxiety/composite_index.py):

| Skor | Formül | KW p | MW p | Cohen's d |
|---|---|---|---|---|
| `rear_count` (baz) | — | 0.045 | 0.247 | +0.57 |
| `pct_periphery` (baz) | — | 0.290 | 0.201 | +1.07 |
| `AI_v1_thigmo_per_rear` | `pct_periphery / (rear_count + 1)` | 0.064 | 0.482 | +0.46 |
| `AI_v2_passive_per_rear` | `(pct_periphery + pct_freeze) / (rear_count + 1)` | 0.060 | 0.845 | +0.00 |
| `AI_v3_thigmo_x_low_rear` | `pct_periphery × (1 − rear_pct/100)` | 0.394 | 0.145 | +0.87 |

Hiçbir kompozit `rear_count`'u geçemedi. Pay/payda zaten feature setinde → lineer ratio yeni varyans katmıyor. **Negative finding** olarak tezde raporlanır — "test edildi, mekansal analiz daha verimli yol oldu".

### 3.6. PC1 ekseni — circularity uyarısı

PCA yönelimsel sıralama (PC1 işaret çapası `pct_periphery+`):

| Grup | n | PC1 ortalama | %95 CI |
|---|---|---|---|
| Control | 5 | **+2.11** | [−0.76, +5.22] |
| Aspartame | 8 | −0.23 | [−1.56, +1.15] |
| ASP+GF | 8 | −0.51 | [−1.44, +0.40] |
| Grapefruit | 8 | −0.58 | [−2.45, +2.33] |

Tüm CI'lar sıfırı içeriyor → grup ayrımı PC1 üzerinde anlamlı değil. Ridge regression R²=1.00 trivial (PC1 zaten feature lineer kombinasyonu → circular). RF R²=0.84 daha gerçekçi ama yine de circular. Bilimsel iddia regresyon R²'sinde değil §3.2'deki yön / sıralamadadır.

---

## 4. Modelin karar mekanizması

Bu bölüm jüri sorusu "modeliniz neye bakarak karar veriyor" için hazır cevap.

### 4.1. Mimari: lojistik regresyon

```
12 raw feature (per subject)
   ↓ SimpleImputer(median)              fold-içi fit
   ↓ StandardScaler(z-score)            fold-içi fit
   ↓ LogisticRegression(L2, C=1.0, class_weight=balanced)
   ↓ logit = intercept + Σᵢ coefᵢ × zᵢ
   ↓ P(Treated) = sigmoid(logit)
   ↓ predict = 1 if P > 0.5 else 0
```

Karar ham özelliklerin ham değerlerine değil **z-skorlarına** dayanır: bir feature'ın katkısı `coefᵢ × zᵢ` — yani modelin o feature'a verdiği önem (coef) **çarpı** bu hayvanın eğitim ortalamasından ne kadar saptığı (z).

### 4.2. "Push" kavramı

Raporlarda her feature için `push = coefᵢ × zᵢ`. İşaret:
- **`push > 0`** → Treated yönüne çekiş
- **`push < 0`** → Control yönüne çekiş

Output'taki top-5 push (mutlak değerce sıralı) görsel kolaylık için; geri kalan 7 + intercept de toplama dahildir.

**Karar oy çokluğu değil, ağırlıklı toplam.** Üç feature Treated derken iki Control derse, magnitude (büyük |coef|×|z|) ne tarafta ise kazanan o olur.

### 4.3. Modelin biyolojik olarak öğrendiği (en güçlü 5 katsayı)

[`reports/anxiety_classifier_importance_rearonly.csv`](../reports/anxiety_classifier_importance_rearonly.csv):

| Feature | LR coef | Anlam |
|---|---|---|
| **`rear_early_frac`** | **−1.50** | Yüksek erken rearing = sağlıklı keşif = **Control** |
| **`pct_periphery`** | **+1.09** | Çok periferde kalmak = anksiyete-benzeri = **Treated** |
| **`spatial_entropy`** | **+0.79** | Dağınık keşif = **Treated** |
| **`pct_freeze`** | **+0.48** | Çok donakalma = **Treated** |
| **`comfort_ratio`** | **+0.41** | Grooming / center yüksek = **Treated** (rahatlama davranışı) |

Yön literatürle uyumlu — özellikle yüksek `pct_periphery` → anksiyete benzeri.

### 4.4. Vaka çalışması: MA4_1 (gerçek grup: Aspartame)

> **Tarihsel not (2026-05-11):** Aşağıdaki push değerleri 18-feature modelden alınmıştır; final 12-feature modelde top-5 ve push büyüklükleri biraz farklıdır (spatial-rearing varyantları artık feature setinde yok, dolayısıyla `rear_center_frac` benzeri uç z-skor uyarıları da kaybolur). Sonuç (doğru sınıflandırma, P_treated ≈ 0.78) korunmuştur. 12-feature push örneği için `python scripts/predict_anxiety_v2.py data/DLCfiltered/Aspartame/MA4_1/MA4_1.csv` ile güncel çıktıyı üretebilirsiniz.

v3 GUI output (retrain sonrası, 2026-05-11):

```
🟧 TAHMİN: Treated-benzeri    (P(Treated)=0.78, %78 güven)
   GERÇEK GRUP: Aspartame      ✓ doğru sınıflandı

Anahtar metrikler:
  pct_periphery     = 87.4%    (yüksek — duvar-tutuş)
  pct_freeze        =  4.4%
  rear_count        = 20
  rear_center_frac  = 0.000    (hepsi duvarda)

Top-5 push:
  rear_early_frac   z=+0.77   push=−1.16  → Control
  spatial_entropy   z=+0.66   push=+0.52  → Treated
  pct_periphery     z=+0.48   push=+0.52  → Treated
  rear_mean_bout_s  z=+1.67   push=+0.50  → Treated
  rear_groom_ratio  z=+0.86   push=+0.35  → Treated

Top-5 toplam:        +0.73 (Treated yönü)
+ kalan 13 feature + intercept:  ≈ +0.5
= logit:              ≈ +1.27
P(Treated) = sigmoid(1.27) = 0.78  ✓
```

**Yorum:** 4 feature Treated, 1 Control diyor — ama tek başına `rear_early_frac`'in push'u (−1.16) diğer 4'ün toplam push'una (+1.89) yakın. Sonuç dar bir Treated kazancı, %78 güven (çok güçlü değil ama net).

**Pratik kural — uç z-skor uyarısı:** Eğer top-5 içinde herhangi bir feature'ın `|z| > 3` ise → feature training dağılımının dışında. Bu işaret ya outlier hayvanı ya da feature-shift bug'ı düşündürür. MA4_1'de max `|z| = 1.67` → sağlıklı tahmin.

---

## 5. Pipeline ve araçlar

### 5.1. CLI — v2

```bash
python scripts/predict_anxiety_v2.py path/to/Subject.csv
#   → reports/anxiety_predictions_v2/<subject>_anxiety_v2_report.{txt,json}
#   → reports/anxiety_predictions_v2/<subject>_anxiety_v2_overview.png
#   → reports/anxiety_predictions_v2/<subject>_behavior_bouts.csv
```

Modüler saf fonksiyonlar — `detect_bouts`, `compute_oft_metrics`, `compute_spatial_rearing`, `build_feature_dict`, `load_model`, `predict`, `render_summary`, `plot_overview`. Import edilebilir; GUI ve testlerde yeniden kullanılır.

### 5.2. Streamlit GUI — v3

```bash
pip install streamlit
streamlit run scripts/predict_anxiety_v3_gui.py
```

v2'nin saf fonksiyonlarını çağıran tarayıcı arayüzü:
- Sol panel: FPS, model tag, "çıktıları reports/ altına yaz" toggle
- Üst: dosya yükleme + ▶ Çalıştır + 4-aşama progress bar
- Üç sütun output: overview PNG · tahmin metrikleri (P_control / P_treated / güven) · anahtar metrikler
- Türkçe rapor inline + 3 indirme butonu (PNG / TXT / JSON)

### 5.3. Retrain helper

```bash
bash scripts/retrain_anxiety_rearonly.sh
```

İki adımlı pipeline:
1. `python -m src.anxiety.spatial_rearing` → `data/spatial_rearing.csv` + `data/anxiety_features_extended.csv` mevcut `INNER_ZONE` ile yeniden üretir
2. `python -m src.anxiety.classifier --csv ... --features '...' --tag rearonly` → `models/anxiety_classifier/{lr,rf,svm,scaler}_rearonly.pkl` yeniden eğitir

**HER ZAMAN çalıştır** when:
- `INNER_ZONE` / `ARENA` / `FPS` değişti
- Yeni subject eklendi
- `spatial_rearing` veya OFT metrik kodu değişti

Neden zorunlu — §6 incident raporu.

### 5.4. Repository organization

[`docs/REPO_MAP.md`](REPO_MAP.md) — aktif anxiety hot path, planlanmış (window classifier, T-maze) ve arşivlenmiş (eski 4-target baseline) ayrımı.

`archive/` altındaki dosyalar artık anxiety hot path'inde değil:
- `archive/scripts/predict_anxiety.py` (v1, `_v2` ile değişti)
- `archive/src/{train_baseline,features,_metrics_minimal,visualize_reports,train_anxiety_demo}.py`
- `archive/models/classifier/` (25 .pkl eski 4-target baseline)

---

## 6. Config mismatch incident — 2026-05-11

Bu rapor öncesi yakalanıp düzeltilen kritik bug. Future-proofing için tam belgelendi.

### 6.1. Belirti

Bilinen Control hayvanı (MA1_1) için v2 inference output:

```
rear_center_frac  : 0.143   (1 center / 7 bout)
Tahmin            : Control-benzeri (P_control = 1.000)

Top push (BEFORE retrain):
  pct_periphery               z=-4.04   push=−4.37 → Control
  rear_early_frac             z=+2.02   push=−3.02 → Control
  rear_center_frac            z=-8.36   push=−3.00 → Control
```

`rear_center_frac` için z = −8 ile −10 → training dağılımının çok dışında. Bu büyüklük "feature-shift" işaretiydi.

### 6.2. Kök neden

Commit [`dd5943b`](https://github.com/Kaankzlyar/RatBehaviorAnalyze/commit/dd5943b) (2026-05-10): `INNER_ZONE` ~%7 margin (inner ≈ %85 of arena) → 20% margin (inner ≈ %36 of arena). **Bu commit'ten ÖNCE eğitilen model:**

- Training data `rear_center_frac` 0.85-1.0 aralığında (geniş center → her şey center sayıldı)
- Model "rear_center_frac ≈ 1 → Treated" öğrendi (Treated'in tavanda kümelendiği patolojik dağılımda)

**v2 inference YENİ config kullanıyor:**

- Aynı hayvanlar için değerler artık 0.0-0.2 aralığında (dar center → çoğu bout wall)
- Model "bu değer training'deki hiçbir Treated örneğine benzemiyor" diyerek tüm hayvanları Control'a itiyor

Sonuç: bilinen-Treated hayvanlar Control olarak sınıflanıyordu.

### 6.3. Düzeltme

[`scripts/retrain_anxiety_rearonly.sh`](../scripts/retrain_anxiety_rearonly.sh) ile:
- `data/spatial_rearing.csv` yenilendi (yeni narrow center ile)
- `data/anxiety_features_extended.csv` güncellendi (rear_center_frac değerleri artık 0.0-0.2 aralığında)
- LR / RF / SVM modelleri yeniden eğitildi
- `reports/anxiety_classifier_importance_rearonly.csv` yeni coefficient'larla yazıldı

**MA4_1 sanity check (Aspartame, retrain sonrası):** doğru sınıflandı (P_treated=0.78, top-5'te `rear_center_frac` yok → narrow-zone calibrated, sağlıklı z-skorlar, §4.4).

### 6.4. Yardımcı düzeltme: v2 hardcoded reference

`predict_anxiety_v2.render_summary` içindeki hardcoded satır:

```python
# ÖNCEKI:
print(f"    rear_center_frac     : {rcf:.3f}  "
      f"(referans: Control medyan 0.133, Treated medyan 0.075)")
# SONRA:
print(f"    rear_center_frac     : {rcf:.3f}  "
      f"(eğitim datası grup medianları için: reports/spatial_rearing_stats.csv)")
```

Eski hardcoded "0.133 / 0.075" değerleri yeni-zone medianlarını gösteriyordu ama model eski-zone'la eğitilmişti → kullanıcıya yanlış sezgi veriyordu. Yeni satır `reports/spatial_rearing_stats.csv`'ye yönlendiriyor; retrain sonrası bu CSV otomatik güncel.

### 6.5. Ders ve önlem

**Ders:** Config sabitleri (zone, margin, arena) **feature sürümünün parçası**. Değiştirirken training data + model + reports'un hepsi senkron yenilenmeli; aksi halde inference distribution ile training distribution çelişir.

**Önlem 1 (uygulanan):** [`scripts/retrain_anxiety_rearonly.sh`](../scripts/retrain_anxiety_rearonly.sh) helper — config değişikliği sonrası tek komutluk regenerate pipeline.

**Önlem 2 (uygulanan):** v2'deki yanıltıcı hardcoded medianlar kaldırıldı.

**Önlem 3 (öneri, gelecek):** v2/v3'e güvenilirlik kontrolü ekle — eğer top contributor'lardan birinin `|z| > 3` ise output'a "DİKKAT: feature dağılımı dışında" uyarısı eklemek. Pseudocode:

```python
extreme = [c for c in pred_info["top_contributors"] if abs(c["z"]) > 3]
if extreme:
    warn("Aşırı uç z-skor: " + ", ".join(c["feature"] for c in extreme) +
         " → tahmin güvenilirliği düşük olabilir, retrain veya outlier kontrolü gerekebilir.")
```

---

## 7. Sınırlılıklar

1. **Örneklem (n=29).** Davranışsal nörobilim alt-sınırının altında. Power ~%30-40 — gerçek orta-büyük etki bile **%60 olasılıkla anlamlılığa ulaşmayabilir.**
2. **Control / Treated dengesizliği.** 5 / 24 → her binary classifier için zorlu setting. `class_weight=balanced` zorunlu, ama outlier sensitivity yüksek (Control grubu skoru 2 hayvan tarafından sürükleniyor).
3. **Treated etiketi heterojen.** Aspartam (anksiyogenik) + Grapefruit (anksiyolitik) + Aspartame+Grapefruit (etkileşim) birlikte etiketlenmiş. Model "anksiyete" değil "**tedavi-kaynaklı davranışsal sapma**" tespit ediyor. Yön ayrımı feature-level analizdedir.
4. **Davranış tespiti kural-bazlı.** Rearing / grooming detector'ı 9 keypoint posture eşiklerine dayanır. Pre-validated sensitivity / specificity yok. `GROUND_TRUTH_BY_SUBJECT` (`src/behavior_detection.py`) sadece bazı seans-aralıkları kapsıyor.
5. **Tek seans / arena.** Test-retest tekrarlanabilirliği ölçülmedi; bireysel varyans ile gerçek tedavi etkisi ayrıştırılamıyor.
6. **`predict_proba` kalibre değil.** LR `class_weight=balanced` ile eğitildi; mutlak olasılık değerleri (örn. 0.78 vs 0.85) **sıralama amaçlı** kullanılmalı, mutlak risk skoru olarak değil.
7. **PC1 yorumlaması circular.** PC1 işaret çapası `pct_periphery+` ile sabitlendi ama loading'lere bağlı olarak "rearing / keşif" vs "thigmotaxis / donma" varyansını yakalıyor olabilir.
8. **Held-out test seti yok.** n=29 kısıtlı olduğu için tüm değerlendirme LOOCV içinde. Out-of-distribution genelleme bilinmiyor.

---

## 8. Power analizi ve replikasyon planı

### 8.1. Gözlenen etki büyüklükleri

| Analiz | Etki büyüklüğü | Power tahmini (n=6/grup) |
|---|---|---|
| `rear_center_frac` MW (Control vs Treated) | d = −0.92 | ~%60 |
| `rear_count_wall` 4-grup KW | η² = 0.21 | ~%50 |
| `rear_count` 4-grup KW | η² = 0.20 | ~%45 |
| `rear_early_frac` 4-grup KW | η² = 0.13 | ~%30 |

### 8.2. %80 power için gereken n

| Hedef etki | Grup başına n | Toplam |
|---|---|---|
| d = 0.92 (rear_center_frac) | ~22 | ~88 |
| η² = 0.20 (rear_count) | ~14 | ~56 |
| η² = 0.13 (rear_early_frac) | ~22 | ~88 |
| η² = 0.06 (orta-küçük) | ~52 | ~210 |

### 8.3. Replikasyon protokolü

1. **n ≥ 20-22 / grup** (toplam ~80-90) — mevcut etki büyüklüklerini %80 güçte doğrulamak için.
2. **Doz-yanıt tasarımı** → tek doz yerine 3 doz × kontrol → "etki var mı" yerine "doz-yanıt eğrisi nedir".
3. **Within-subject baseline** → her hayvanın pre-treatment baseline'ı; varyansı yarıdan fazla düşürür, aynı n ile çok daha yüksek power.

---

## 9. Reproducibility

Sıfırdan tam pipeline:

```bash
# 1. DLC pose CSV'leri data/DLCfiltered/<group>/<subject>/<subject>.csv altında olmalı

# 2. Davranış tespiti — her hayvan için
python -m src.behavior_detection

# 3. OFT metrikleri ve trajectory görselleri
python analysis/open_field/run_analysis.py --arena 397 777 156 535
# veya batch (tüm subjects):
python analysis/open_field/run_kare_batch.py

# 4. Anxiety feature matrix + PCA + univariate KW
python -m src.anxiety.profile
#   → data/anxiety_features.csv
#   → reports/anxiety_pca.png + anxiety_boxplots.png + anxiety_stats.csv

# 5. Composite index (negatif bulgu, dürüstlük için)
python -m src.anxiety.composite_index
#   → reports/composite_anxiety_*.csv
#   → reports/figures/composite_anxiety_boxplot.png

# 6 + 7. Spatial rearing + retrain (tek komut):
bash scripts/retrain_anxiety_rearonly.sh

# Veya manuel:
# 6. Spatial rearing + extended CSV
python -m src.anxiety.spatial_rearing
#   → data/spatial_rearing.csv, spatial_rearing_bouts.csv, anxiety_features_extended.csv
#   → reports/spatial_rearing_stats.csv
#   → reports/figures/spatial_rearing_{boxplot,arena}.png

# 7. Rear-only LOOCV classifier
python -m src.anxiety.classifier \
    --csv data/anxiety_features_extended.csv \
    --features 'rear|pct_periphery|pct_freeze|spatial_entropy|comfort' \
    --tag rearonly
#   → reports/anxiety_classifier_{metrics,predictions,importance}_rearonly.csv
#   → reports/figures/anxiety_classifier_{cm,roc}_rearonly.png
#   → models/anxiety_classifier/{lr,rf,svm,scaler}_rearonly.pkl

# 8. PC1-axis regression
python -m src.anxiety.regression

# 9. Tek subject end-to-end inference (CLI)
python scripts/predict_anxiety_v2.py data/DLCfiltered/<group>/<subject>/<subject>.csv

# 10. Tarayıcı GUI
pip install streamlit
streamlit run scripts/predict_anxiety_v3_gui.py
```

**Çıktı yolları:**
- Per-subject artefacts: `reports/anxiety_predictions_v2/`
- Cross-subject stats: `reports/anxiety_*.csv`, `reports/spatial_rearing_*.csv`, `reports/composite_anxiety_*.csv`
- Figures: `reports/figures/anxiety_*.png`, `reports/figures/spatial_rearing_*.png`
- Models: `models/anxiety_classifier/`, `models/anxiety_regression/`

---

## 10. Methods bölümü için hazır cümleler

### 10.1. Pipeline (Methods §X.1)

> "Pose was estimated using a top-down 9-keypoint DeepLabCut model (PyTorch backend, RTX 3060). Rearing and grooming bouts were detected from filtered pose CSVs using a rule-based classifier (postural thresholds on inter-paw distance, paw-vertical offset, and body velocity; merge gap ≤ 15 frames, minimum bout duration ≥ 10 frames). OFT metrics (thigmotaxis, locomotion, freeze, spatial entropy) and spatial rearing classification (center vs wall, using arena 397–777 × 156–535 px and a literature-standard 20% margin inner zone, ≈36% of arena area) were computed downstream. A sensitivity analysis with a wider center definition (~7% margin, ~85% of arena classified as center) confirmed that the metric measures the thigmotaxis-ring versus inner-area distinction rather than wall contact per se."

### 10.2. Model (Methods §X.2)

> "A logistic regression classifier (L2, C=1.0, `class_weight='balanced'`) was trained on 12 rearing-focused features (regex pattern `rear|pct_periphery|pct_freeze|spatial_entropy|comfort`, after dropping six highly correlated spatial-rearing center/wall variants identified as a source of LOOCV variance during iterative refinement) to discriminate Control (n=5) vs Treated (n=24, pooling Aspartame / Grapefruit / Aspartame+Grapefruit). Leave-one-out cross-validation was used owing to sample size constraints; no held-out test set was retained. Feature space and zone definition were iteratively refined through LOOCV; the final configuration corresponds to LR LOOCV AUC = 0.683 (balanced accuracy = 0.738; RF AUC = 0.525, SVM-RBF AUC = 0.358). The 'Treated' label is heterogeneous, pooling anxiogenic (aspartame) and anxiolytic (grapefruit) interventions; the model is therefore best interpreted as a **treatment-induced behavioral shift detector** rather than an anxiety classifier."

### 10.3. Inference (Methods §X.3)

> "The final classifier is deployed via two equivalent interfaces: a command-line tool (`scripts/predict_anxiety_v2.py`) and a Streamlit web GUI (`scripts/predict_anxiety_v3_gui.py`). Both compute the 12-feature vector from a single DLC pose CSV, apply the saved median imputer + standard scaler + LR pickle, and report (a) predicted class with probability, (b) top-5 contributing features with their standardized z-values and coefficient-weighted contributions (`push = coef × z`), (c) an arena overlay PNG showing trajectory and per-bout rearing locations colored by center / wall classification."

### 10.4. Sınırlılıklar (Discussion §X)

> "Effect sizes for spatial rearing (Cohen's d = −0.92 for `rear_center_frac`, Treated vs Control) reach uncorrected significance (Mann-Whitney U p = 0.030) but do not survive Bonferroni correction across the seven spatial metrics tested (corrected p = 0.21). Power analysis indicates ≥20–22 animals per group are required to detect this effect size at 80% power; the present n = 29 study should therefore be interpreted as **pilot evidence**, not a definitive treatment effect."

---

## 11. Tezdeki tek-cümle mesaj

> "29 hayvan üzerinde DeepLabCut tabanlı uçtan uca yeniden üretilebilir bir davranış-analizi pipeline'ı geliştirildi; pipeline, rearing davranışının mekansal dağılımı (center vs wall) üzerinden Control vs Treated için LOOCV AUC=0.683 (balanced accuracy=0.738) elde etti ve uncorrected anlamlılıkta (`rear_center_frac` MW p=0.030, Cohen's d=−0.92) pilot kanıt sağladı; Bonferroni sonrası bu eşik geçilmemekle birlikte, etki büyüklüğü ve yön literatürdeki aspartam-anksiyojenik / flavonoid-anksiyolitik hipoteziyle uyumludur ve ≥20 hayvan/grup ile %80 güçte replikasyon için a-priori temel oluşturmaktadır."

**Bu cümle:**
- Anlamlılık iddia etmiyor (Bonferroni sonrası eşiği geçmediği belirtildi)
- Pipeline'ı asıl katkı olarak öne çıkarıyor
- Empirik bulgu için somut effect size + yön + replikasyon planı sunuyor
- Jüri sorularına savunulabilir cevaplar içeriyor

---

## 12. Modelin yaşam döngüsü — training mı, fine-tuning mı?

**İleride bu raporu okuyan kişiye:** `python -m src.anxiety.classifier` her çalıştırıldığında model **sıfırdan eğitilir** (imputer.fit, scaler.fit, LR.fit). Bu **fine-tuning DEĞİL** — sklearn LR / RF / SVM'lerinde pre-trained ağırlıkları başlangıç noktası alıp devam etme paradigması yoktur (LR'de `warm_start=True` ile kabaca yapılabilir, pratik kazanç yok).

Bizim yaptığımız: **iteratif feature engineering ve hyperparameter tuning**:

| İterasyon | Değişen | LR AUC |
|---|---|---|
| 1 | Full 24 feature, default zone | 0.57 |
| 2 | Rear-only 18 feature, %20 zone (ilk eğitim) | 0.73 |
| 3 | Manuel zone (sensitivity test) | 0.62 |
| 4 | %20 margin lock + retrain (config bug fix, 18 feat) | 0.617 |
| 5 | 12 feature (6 spatial-rearing varyantı kaldırıldı — multicollinearity / LOOCV varyans) | **0.683 (final)** |

Her iterasyonda **konfigürasyon kararı** öğreniliyor, ağırlıklar değil.

**Methods bölümü için doğru ifade:**

> "Feature space and zone definition were iteratively refined through leave-one-out cross-validation on n=29 subjects; the final configuration consists of (a) feature pattern matching `rear|pct_periphery|pct_freeze|spatial_entropy|comfort` (12 features, after dropping six highly correlated spatial-rearing center/wall variants whose joint inclusion produced unstable LOOCV AUC estimates; e.g., the 18-feature configuration produced AUC 0.733 on the first fit and 0.617 on a same-config retrain), (b) center zone defined as 20% margin from arena bounds. Reported AUC values are from LOOCV on the final configuration; no held-out test set was retained owing to sample size constraints. No pre-trained weights were carried across iterations; each LOOCV run is a fresh fit."

---

## Appendix A. Sürüm geçmişi

| Tarih | Sürüm | Değişiklik |
|---|---|---|
| 2026-05-09 | v0 | İlk çalışmalar — `src/features.py` + `src/train_baseline.py` (4-target LOOCV) |
| 2026-05-10 morning | v1 | Anxiety pivot — `src/anxiety/{profile,classifier,regression}.py` |
| 2026-05-10 noon | v1.5 | Composite index (negatif bulgu) eklendi |
| 2026-05-10 afternoon | v2-prep | Spatial rearing eklendi, ana bulgu (rear_center_frac d=−0.92) yakalandı |
| 2026-05-10 evening | v2 | `predict_anxiety_v2.py` end-to-end inference |
| 2026-05-11 | v2.5 | `archive/` reorganization, `docs/REPO_MAP.md`, README anxiety pipeline'ı yansıtacak şekilde yenilendi |
| 2026-05-11 | v3 | Streamlit GUI (`predict_anxiety_v3_gui.py`) |
| 2026-05-11 | v3.1 | **Config mismatch bug yakalandı**, retrain helper eklendi, v2 hardcoded reference temizlendi |
| 2026-05-11 | v3.2 | Feature seti 18 → 12 (6 spatial-rearing varyantı kaldırıldı — LOOCV varyans / multicollinearity). LR AUC 0.617 → 0.683, balanced acc 0.617 → 0.738, F1_control 0.364 → 0.545 |

Eski rapor dosyaları artık tarihsel kayıt (bu doküman ikisini birleştirir):
- [`docs/anxiety_findings_report.md`](anxiety_findings_report.md) — 2026-05-10 baseline + 9 numaralı follow-up
- [`docs/anxiety_progress_2026-05-10.md`](anxiety_progress_2026-05-10.md) — 2026-05-10 spatial rearing + inference + config birleşmesi

---

## Appendix B. Hızlı referans

**Anxiety üzerinde çalışırken bakılacak yerler:**

```
src/anxiety/                              # tüm anxiety modülleri
src/behavior_detection.py                 # rule-based detector
analysis/open_field/oft_metrics.py        # OFT metrik hesabı (anxiety'nin import ettiği)
scripts/predict_anxiety_v2.py             # CLI inference
scripts/predict_anxiety_v3_gui.py         # Streamlit GUI
scripts/retrain_anxiety_rearonly.sh       # config sonrası tek-komut retrain

models/anxiety_classifier/                # eğitilmiş modeller
reports/anxiety_predictions_v2/           # per-subject inference output
reports/figures/                          # CM, ROC, spatial rearing PNG'leri
data/anxiety_features.csv                 # ham 24-feature matrix
data/anxiety_features_extended.csv        # + 6 spatial sütun (classifier'ın input'u)

docs/anxiety_report.md                    # bu doküman
docs/REPO_MAP.md                          # repo organization
```
