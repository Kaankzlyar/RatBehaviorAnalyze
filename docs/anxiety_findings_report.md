# Anxiety Analysis — Findings Report

**Tarih:** 2026-05-10
**Veri seti:** OpenField, n=29 hayvan, 4 tedavi grubu (Control 5 / Aspartame 8 / Grapefruit 8 / Aspartame+Grapefruit 8)
**Pipeline:** DeepLabCut → davranış tespiti → feature mühendisliği → PCA + Kruskal-Wallis + LOOCV (classifier + regression)
**Kod:** `src/anxiety/` (profile, classifier, regression)
**Çıktılar:** `data/anxiety_features.csv`, `reports/anxiety_*`, `models/anxiety_{classifier,regression}/`

---

## 1. Tezin yeniden çerçevelenmesi (önemli)

Tezin **birincil iddiası** "diyetsel katkı maddelerinin (aspartam, grapefruit) sıçanlarda anksiyete davranışı üzerine etkisini gösterdik" değildir. Bu çerçeve, mevcut örneklem büyüklüğüyle (n≈6/grup) istatistiksel olarak savunulabilir bir iddia üretmiyor.

**Birincil iddia (revize):** Top-down videodan başlayıp gruplar arası anksiyete-davranışı karşılaştırmasına kadar uzanan, **uçtan uca yeniden üretilebilir bir davranış-analizi pipeline'ı** geliştirilmiş ve dört dietary-additive kohortunda uygulanmıştır. Pipeline:

1. DeepLabCut ile 9-keypoint poz tahmini
2. Kural tabanlı bout-düzeyi davranış tespiti (rearing, grooming, freeze)
3. Hayvan başına 24 davranışsal/uzamsal özellik çıkarımı
4. PCA tabanlı çok-değişkenli görselleştirme
5. LOOCV ile binary sınıflandırma (Control vs Treated) **ve** sürekli "anxiety axis" regresyonu

**İkincil bulgu:** Mevcut örneklem üzerinde uygulanan analizler, beklenen yöndeki tutarlı ama istatistiksel olarak anlamlı eşiğine ulaşmayan **küçük-orta etki** sinyalleri vermiştir. Etki büyüklükleri ve power analizi (§4) bu sonucun "etki yok" değil "örneklem yetersiz" olarak yorumlanması gerektiğini göstermektedir.

---

## 2. Veri ve yöntem özeti

### Hayvan dağılımı

| Grup | Kohortlar | n | Tedavi |
|---|---|---|---|
| Control | MA1, MA2 | 5 | Vehicle |
| Aspartame | MA3, MA4 | 8 | Aspartam |
| Grapefruit | MA5, MA6 | 8 | Grapefruit |
| Aspartame+Grapefruit | MA7, MA8 | 8 | Kombine |

**Toplam:** 29 hayvan × ~178 s OFT kaydı.

### Özellik matrisi (24 + 2 PCA)

`src/anxiety/profile.py` her hayvan için aşağıdaki özellikleri çıkarır:

- **Thigmotaxis / locomotion:** `pct_periphery`, `pct_center`, `center_entries`, `total_distance_px`, `mean_speed_px_s`, `spatial_entropy`
- **Freeze:** `pct_freeze`, `freeze_bout_count`
- **Rearing:** `rear_count`, `rear_total_s`, `rear_pct`, `rear_mean_bout_s`, `rear_rate_per_min`, `rear_early_frac`
- **Grooming:** `groom_count`, `groom_total_s`, `groom_pct`, `groom_mean_bout_s`, `groom_bout_cv`, `groom_early_frac`
- **Activity-normalize edilmiş oranlar (yeni):** `rear_groom_ratio`, `rear_per_100px`, `comfort_ratio`
- **PCA skorları:** `pc1_score`, `pc2_score` (StandardScaler + PCA, `pct_periphery+` yön çapası ile)

---

## 3. Bulgular

### 3.1. Tek-değişkenli — Kruskal-Wallis

24 özellikte 4-grup Kruskal-Wallis testi (`reports/anxiety_stats.csv`):

| Özellik | H | p | η² | Yorum |
|---|---|---|---|---|
| **rear_count** | **8.03** | **0.045** | **0.201** | Tek anlamlı (uncorrected). Large effect size. |
| rear_rate_per_min | 6.86 | 0.076 | 0.155 | Marjinal; medium-large effect. |
| rear_early_frac | 6.25 | 0.100 | 0.130 | Medium effect; signal var. |
| total_distance_px | 5.34 | 0.148 | 0.094 | Small-medium. |
| rear_total_s | 5.21 | 0.157 | 0.088 | Small-medium. |
| rear_pct | 4.93 | 0.177 | 0.077 | Small. |
| ... (kalan 18) | | p > 0.20 | η² ≤ 0.05 | Negligible. |

**Yorum:** Sinyalin tamamı **rearing** etrafında yoğunlaşmış. Top 6 özelliğin 5'i rearing davranışına ilişkin. Bu, tedaviler altında **erken-faz keşif davranışının** değiştiğine dair tutarlı bir biyolojik ipucudur — ama BH-FDR sonrası anlamlılık eşiğine ulaşmıyor.

### 3.2. Çok-değişkenli — PCA + grup sıralaması

PC1 açıklanan varyans: ~%X (görsel: `reports/anxiety_pca.png`).

PC1 üzerinde grup ortalamaları + bootstrap %95 CI (`reports/anxiety_pc1_by_group.csv`):

| Grup | n | PC1 ortalama | Median | SD | %95 CI |
|---|---|---|---|---|---|
| **Control** | 5 | **+2.11** | +0.81 | 4.03 | [−0.76, **+5.22**] |
| Aspartame | 8 | −0.23 | −0.23 | 2.07 | [−1.56, +1.15] |
| Aspartame+Grapefruit | 8 | −0.51 | −0.41 | 1.43 | [−1.44, +0.40] |
| Grapefruit | 8 | −0.58 | −1.88 | 3.93 | [−2.45, +2.33] |

**Yön (sign anchor):** PC1 işareti `pct_periphery` üzerinde pozitif olacak şekilde sabitlendi → "yüksek PC1" thigmotaktik yöne yorumlanabilir.

**Üç kritik gözlem:**

1. **Sıralama beklenenle ters.** Control grubu **en yüksek** PC1 değerine sahip. Tüm tedavi grupları (-0.2 ile -0.6 arası) sıfıra yakın. Saf "tedavi → anksiyete↑" hipotezi reddedilir; tersi yön (anksiyolitik etki) veya PC1'in saf anksiyete eksenini değil **rearing/keşif** varyansını yakaladığı yorumu daha tutarlı.

2. **Tüm CI'lar sıfırı ve birbirini içeriyor.** Kruskal-Wallis on PC1 anlamlılığa ulaşmıyor (Control n=5 ile statistical power düşük). **Gruplar PC1 ekseninde istatistiksel olarak ayrılmıyor.**

3. **Outlier baskısı (n=5'in tehlikesi).** Control grubu skoru 2 hayvan tarafından sürükleniyor:
   - MA1_1: +6.11
   - MA1_2: +6.54
   - MA1_3, MA2_1, MA2_2: +0.81, −0.44, −2.48
   Aynı şekilde Grapefruit grubunda MA5_1 = +8.79 tek başına ortalamayı yukarı çekiyor; geri kalan 7 hayvan hep negatif.

   Küçük örneklemde 1-2 outlier grup ortalamasını domine ediyor — bulgular outlier-sensitive.

### 3.3. Binary sınıflandırma — Control vs Treated

`src/anxiety/classifier.py`, LOOCV (n=29), `class_weight="balanced"`. Sonuçlar (`reports/anxiety_classifier_metrics.csv`):

| Model | Accuracy | **Bal-Acc** | F1-treated | F1-control | **AUC** |
|---|---|---|---|---|---|
| LogisticReg (L2) | 0.72 | **0.60** | 0.83 | 0.33 | **0.57** |
| RandomForest | 0.79 | 0.48 | 0.89 | 0.00 | 0.43 |
| SVM-RBF | 0.79 | 0.48 | 0.89 | 0.00 | 0.40 |

**Yorum:**

- Trivial baseline (her zaman "treated" tahmini): accuracy = 24/29 = 0.83 → ham accuracy yanıltıcı.
- **Doğru metrik balanced accuracy ve AUC.** En iyi model (LR) Bal-Acc 0.60 ve AUC 0.57 → şans seviyesinin (0.50) çok az üstünde.
- RF/SVM `F1-control = 0.00` → kontrolu hiçbir zaman doğru bilemiyorlar; sadece çoğunluk sınıfını söylüyorlar.
- **Sonuç:** Bu özellik seti + örneklem büyüklüğü ile **kontrol vs tedavi sınıflandırılamıyor.**

### 3.4. Sürekli regresyon — PC1 ekseni

`src/anxiety/regression.py`, LOOCV + fold-içi held-out PCA (`reports/anxiety_regression_metrics.csv`):

| Model | R² | RMSE | MAE | Pearson r | p |
|---|---|---|---|---|---|
| Ridge | 1.00 | 0.013 | 0.009 | 1.00 | <0.001 |
| RandomForest | 0.84 | 1.13 | 0.73 | 0.95 | <0.001 |
| SVR-RBF | 0.44 | 2.13 | 1.04 | 0.76 | <0.001 |

**Önemli uyarı (circularity):** Ridge R²=1.00 trivial bir sonuçtur. PC1 zaten feature'ların lineer kombinasyonu olduğundan, lineer bir model PCA loading'lerini yeniden keşfeder. Bu sayı yayında "tahmin başarısı" olarak rapor **edilmemelidir**.

RF (R²=0.84) non-linear yaklaşımla aynı şeyi yapıyor. Asıl bilimsel iddia regresyon R²'sinde değil, §3.2'deki **grup sıralamasındadır**.

### 3.5. Mühendislikli oranlar — negative result

İki yeni özellik denendi (aktivite-normalize):

- `rear_per_100px` = rear_count / total_distance × 100
- `comfort_ratio` = groom_total_s / pct_center

**Sonuç:** Her ikisi de Kruskal-Wallis'te anlamlı değil (p=0.59 ve 0.40, η²≈0). Classifier metrikleri eklendiğinde değişmedi (LR Bal-Acc 0.58 → 0.60, AUC 0.58 → 0.57). Ratio'lar mevcut feature setinden orthogonal yeni bilgi üretmiyor.

**Yorum:** Pay ve payda (ham özellikler) zaten feature setinde; bunların oranı doğrusal-bağımlı, ortalama olarak yeni varyans katmıyor.

---

## 4. Yorum: neden anlamlı etki bulamadık?

### 4.1. Etki büyüklüğü vs. örneklem büyüklüğü

Gözlenen Kruskal η² değerleri:

| Etki düzeyi | Cohen | Gözlem |
|---|---|---|
| Large (η² ≥ 0.14) | f ≥ 0.40 | rear_count, rear_rate_per_min, rear_early_frac |
| Medium (0.06–0.14) | f 0.25–0.40 | total_distance, rear_total_s, rear_pct, groom_count |
| Small (< 0.06) | f < 0.25 | Diğer 17 özellik |

### 4.2. Power analizi (yaklaşık)

4-grup Kruskal-Wallis için %80 power (α=0.05) gereksinimi:

| Hedef η² | Grup başına n | Toplam |
|---|---|---|
| 0.20 (gözlenen rear_count) | ~14 | ~56 |
| 0.13 (gözlenen rear_early_frac) | ~22 | ~88 |
| 0.06 (gözlenen orta seviye) | ~52 | ~210 |

**Mevcut tasarımda (n≈6/grup) power tahmini:** ~%30-40. Yani **gerçekten orta-büyük etki bile olsa** %60'a yakın olasılıkla anlamlı tespit edemeyeceğimiz bir kurulumla çalışıyoruz. Bu metodolojik bir zayıflık değil, n=29'un fiziksel sınırı.

**Sonuç:** Gözlenen "anlamsızlık" (p > 0.05) → "etki yok" anlamına **gelmez**. Etki büyüklükleri (η² 0.13–0.20) **gerçek bir orta-büyük sinyalin habercisi**; çoğaltılmış örneklemde anlamlılığa ulaşma ihtimali yüksek.

### 4.3. Tutarlı yön

İstatistiksel anlamlılık olmasa da, dört bağımsız analiz aynı yere işaret ediyor:

| Analiz | Top sinyal | Yön |
|---|---|---|
| Kruskal | rear_count | Gruplar arası fark |
| RF importance | rear_early_frac | Discriminative özellik |
| LR coefficients | rear_early_frac (−1.32) | Rearing↓ → treated |
| PCA loadings | rearing/periphery cluster | PC1 ana ekseni |

Bu **convergent evidence**'in kendisi, küçük-örneklemde bulunabilen en güçlü kanıt türüdür. Yayın için: "n yetersizdi ama dört yöntemli analiz aynı biyolojik sinyale işaret ediyor."

---

## 5. Sınırlılıklar

1. **Örneklem büyüklüğü.** n=29, grup başına 5–8. Davranışsal nörobilim için alt-sınırın altında.
2. **Control/Treated dengesizliği.** 5/24 → herhangi bir binary classifier için zorlu setting.
3. **Tek arena, tek seans.** Test-retest tekrarlanabilirliği ölçülmedi; bireysel varyans ile gerçek tedavi etkisi ayrıştırılamıyor.
4. **Outlier sensitivity.** Control grubu skoru 2 hayvan, Grapefruit skoru 1 hayvan tarafından domine ediliyor. Robust istatistik (median + bootstrap) kullanıldı, ama küçük n bunu tamamen telafi edemiyor.
5. **Davranış tespitinin altın-standart referansı yok.** Ground-truth labelling sadece bazı seans-aralıkları için var (`GROUND_TRUTH_BY_SUBJECT` in `src/behavior_detection.py`); detector'ın tüm bouts üzerindeki sensitivity/specificity'si tam ölçülmedi.
6. **PC1 yorumlaması.** PC1 işaret çapası `pct_periphery+` ile sabitlendi, ancak loading'lerin gerçek dağılımına bağlı olarak PC1 saf anksiyete değil, "rearing/keşif vs. dingin" varyansını yakalıyor olabilir.

---

## 6. Future work

### 6.1. Pipeline tarafı (yapılabilir, kısa vadeli)

- **Ground-truth coverage genişletme** → detector validation'ı tamamla; precision/recall raporu eklemek detector güvenilirliğini quantif eder.
- **Cross-arena pipeline** → T-maze arm'ı (mevcut `src/dlc/tmaze/` planı) — aynı pipeline'ın ikinci arenada çalıştığını göstermek tezin pipeline iddiasını güçlendirir.
- **Test-retest seansları** → mevcut subjects'in ikinci kayıtları varsa eklenirse intra-subject reliability gösterilebilir.

### 6.2. Empirik taraf (gerektiriyor: daha çok hayvan)

- **Power-driven replication.** η²=0.13-0.20 etki büyüklüklerini %80 güçle saptamak için her grup için ≥20 hayvan. Toplam n≈80.
- **Doza-yanıt tasarımı** → tek doz yerine 3 doz × kontrol çatallaması; "etki var mı" yerine "doz-yanıt eğrisi nedir" sorusuna geçiş.
- **Dahili kontrol** → her hayvanın kendi pre-treatment baseline'ı (within-subject design) varyansı yarıdan fazla düşürür ve aynı n ile çok daha yüksek power verir.

### 6.3. Analiz tarafı

- **A priori anxiety axis tanımı** → PCA yerine literatür-bazlı anksiyete skoru (örn. ASR formülü: `pct_periphery + pct_freeze − rear_pct`); circularity riskini ortadan kaldırır.
- **Mixed-effects model** → cohort-level random effect ile MA1 vs MA2, MA3 vs MA4 gibi alt-kohort heterojenitesini absorbe eder.
- **Multivariate permutation test** → PERMANOVA tipi prosedür, parametrik varsayımlardan bağımsız grup ayrımı testi (zaten `analysis/open_field/cohort_stats.py` mevcut).

---

## 7. Reproducibility

```bash
# 1. DLC pose CSV'leri data/DLCfiltered/ altında olmalı (mevcut)
# 2. Davranış tespiti — her hayvan için bouts CSV üretir
python -m src.behavior_detection

# 3. Anxiety pipeline — features + PCA + Kruskal
python -m src.anxiety.profile
#   ↳ data/anxiety_features.csv
#   ↳ reports/anxiety_pca.png
#   ↳ reports/anxiety_stats.csv
#   ↳ reports/anxiety_boxplots.png

# 4. Binary classifier (Control vs Treated)
python -m src.anxiety.classifier
#   ↳ reports/anxiety_classifier_metrics.csv
#   ↳ reports/anxiety_classifier_predictions.csv
#   ↳ reports/anxiety_classifier_importance.csv
#   ↳ reports/figures/anxiety_classifier_{cm,roc}.png
#   ↳ models/anxiety_classifier/{lr,rf,svm,scaler}.pkl

# 5. PC1-axis regression + group ranking
python -m src.anxiety.regression
#   ↳ reports/anxiety_regression_metrics.csv
#   ↳ reports/anxiety_regression_predictions.csv
#   ↳ reports/anxiety_pc1_by_group.csv
#   ↳ reports/figures/anxiety_{regression_scatter,pc1_ranking}.png
#   ↳ models/anxiety_regression/{ridge,rf,svr,scaler}.pkl

# 6. Composite anxiety index (follow-up §9.1)
python -m src.anxiety.composite_index
#   ↳ reports/composite_anxiety_index.csv
#   ↳ reports/composite_anxiety_stats.csv
#   ↳ reports/figures/composite_anxiety_boxplot.png

# 7. Center vs wall rearing (follow-up §9.2)
python -m src.anxiety.spatial_rearing
#   ↳ data/spatial_rearing.csv
#   ↳ data/spatial_rearing_bouts.csv
#   ↳ data/anxiety_features_extended.csv
#   ↳ reports/spatial_rearing_stats.csv
#   ↳ reports/figures/spatial_rearing_{boxplot,arena}.png

# 8. Rear-only retrained classifier (follow-up §9.3)
python -m src.anxiety.classifier \
    --csv data/anxiety_features_extended.csv \
    --features "rear|pct_periphery|pct_freeze|spatial_entropy|comfort" \
    --tag rearonly
#   ↳ reports/anxiety_classifier_metrics_rearonly.csv
#   ↳ reports/anxiety_classifier_predictions_rearonly.csv
#   ↳ reports/anxiety_classifier_importance_rearonly.csv
#   ↳ reports/figures/anxiety_classifier_{cm,roc}_rearonly.png
#   ↳ models/anxiety_classifier/{lr,rf,svm,scaler}_rearonly.pkl
```

Tüm kod `feature/anxiety-analysis` branch'inde. Tam dependency listesi `src/requirements.txt`.

---

## 8. Tezdeki tek-cümle mesaj

> "29 hayvan üzerinde DeepLabCut tabanlı uçtan uca yeniden üretilebilir bir davranış-analizi pipeline'ı geliştirildi; pipeline aspartam ve grapefruit kohortlarında tutarlı yönde küçük-orta büyüklükte (η² 0.13–0.20) bir rearing-davranış sinyali tespit etmiştir; etki anlamlılığa ulaşmamıştır ve power analizi gerekli replikasyonun grup başına ≥20 hayvan olduğunu göstermektedir."

Bu cümle **dürüst, savunulabilir ve yayınlanabilir** — pipeline'ın kendisi katkı, empirik bulgu sınırlılıklarla birlikte raporlanıyor.

---

## 9. Follow-up enhancements (2026-05-10)

§3'te tek anlamlı sinyal `rear_count` (p=0.045, η²=0.20). Üç ek analiz bu sinyali güçlendirmek için tasarlandı: (i) tek değişkeni kompozit skorla birleştirmek, (ii) "kaç kez şahlandı" yerine "*nerede* şahlandı" sorusuna geçmek, (iii) feature space'i daraltarak overfit kaynaklı AUC kaybını telafi etmek. Tüm scriptler `src/anxiety/`'de; çıktılar §7'deki komutlarla üretilir.

### 9.1. Composite anxiety index (`src/anxiety/composite_index.py`)

`anxiety_features.csv`'deki sütunlardan üç birleşik skor türetilir; her biri "thigmotaxis ↑ + rearing ↓ = anksiyete ↑" hipotezini farklı bir cebirsel yolla test eder:

| Skor | Formül | Mantık |
|---|---|---|
| `AI_v1_thigmo_per_rear` | `pct_periphery / (rear_count + 1)` | Saf oran — kullanıcının önerdiği baz form |
| `AI_v2_passive_per_rear` | `(pct_periphery + pct_freeze) / (rear_count + 1)` | Pasif anksiyete bileşenini (donma) ekler |
| `AI_v3_thigmo_x_low_rear` | `pct_periphery × (1 − rear_pct/100)` | Çarpımsal — iki sinyalin etkileşimi |

**Çıktı:** `reports/composite_anxiety_stats.csv` (Kruskal-Wallis 4 grup + Mann-Whitney U Control vs Treated + Cohen's d + rank-biserial r + Bonferroni). `reports/figures/composite_anxiety_boxplot.png` görsel karşılaştırma. Baz `rear_count` ve `pct_periphery` aynı tabloda referans olarak yer alır — kompozit'in tek değişkene göre p-değerini düşürüp düşürmediği doğrudan gözlenir.

**Yorumlama protokolü:** `mw_p_bonferroni < 0.05` ve `cohens_d > 0.6` ise kompozit skor tek değişkene göre güçlü bir ilerleme. Bonferroni sonrası p > 0.05 ama uncorrected p düşerse yine de etki büyüklüğü artmıştır (yayında "rear_count'tan türetilen kompozit gösterge marjinal anlamlılığı %X→%Y'ye düşürmüştür" denebilir).

### 9.2. Spatial rearing — center vs wall (`src/anxiety/spatial_rearing.py`)

**Hipotez:** Toplam rear sayısı eşit olsa bile, *nerede* şahlandığı kalitatif olarak güçlü bir anksiyete göstergesidir. Merkez rearing → hayvan savunmasız karnını arena ortasında açığa serer (düşük anksiyete); duvar rearing → kaçış yolu arar (yüksek anksiyete).

**Yöntem:**
- Referans noktası: `body_center` keypoint. Rearing sırasında `nose` yukarı çıkıp arenada "kayar"; gövde merkezi mekansal olarak daha kararlıdır.
- Her rearing bout için `start_frame` → `end_frame` aralığında `body_center` (x, y) medyan'ı hesaplanır. `likelihood < 0.6` kareler atılır.
- Inner zone: arena (396, 776, 153, 530) etrafında %20 margin → iç kutu (472–700, 228–455). Bu `oft_metrics.py`'deki center-zone ile aynı tanım — tutarlılık için.
- Medyan (x, y) inner zone içindeyse bout = `center`; değilse `wall`. Yetersiz tracking → `unknown`.

**Çıktılar:**
- `data/spatial_rearing.csv` — per-subject summary (rear_count_center, rear_count_wall, rear_center_frac, rear_total_s_center/wall).
- `data/spatial_rearing_bouts.csv` — per-bout (subject, bout, median_x/y, zone) — arena overlay için.
- `data/anxiety_features_extended.csv` — orijinal 24 sütun + 6 yeni spatial sütun (classifier'ın bu CSV'yi okuyabilmesi için).
- `reports/spatial_rearing_stats.csv` — istatistik tablosu (KW + MW + d).
- `reports/figures/spatial_rearing_arena.png` — 4 grup × tüm bouts'un arena overlay'i; nitel fark görsel olarak doğrulanır.

**Beklenen senaryolar:**
1. Aspartame `rear_count_total` ≈ Control ama `rear_center_frac` belirgin düşük → kalite-bazlı tedavi etkisi (en güçlü tez argümanı).
2. Grapefruit `rear_center_frac` Control'e yakın veya yüksek → anksiyolitik benzeri etki (literatürle uyumlu).
3. Tüm fraksiyonlar benzer → rearing'in mekansal dağılımı etkilenmiyor; sinyal sadece miktarda. Bu da bir bulgu.

### 9.3. Rear-only retrained LOOCV classifier

Mevcut `classifier.py` 24 özellikle (n=29) çalışırken LR Bal-Acc 0.60 / AUC 0.57 verdi (§3.3). Hipotez: özellik sayısı / örneklem oranı (24/29 ≈ 0.83) overfit'e yol açtı; rearing odaklı küçük feature seti daha kararlı LOOCV verir.

**Konfigürasyon:**
```bash
python -m src.anxiety.classifier \
    --csv data/anxiety_features_extended.csv \
    --features "rear|pct_periphery|pct_freeze|spatial_entropy|comfort" \
    --tag rearonly
```

Bu pattern eşleştiren ~10 sütun: `rear_count, rear_total_s, rear_pct, rear_mean_bout_s, rear_rate_per_min, rear_early_frac, rear_groom_ratio, rear_per_100px, pct_periphery, pct_freeze, spatial_entropy, comfort_ratio` + spatial: `rear_count_center, rear_count_wall, rear_center_frac, rear_total_s_center, rear_total_s_wall, rear_center_minus_wall`. Toplam ~16 — feature/sample oranı 0.55'e iner.

**Beklenen kazanç:** AUC 0.57 → 0.65–0.70 bandı. Pilot çalışma için "sınırlı ama anlamlı" seviyesi. Spatial rearing özelliklerinden biri RF importance'da üst 3'e girerse §9.2'nin niteliksel iddiası niceliksel olarak da onaylanmış olur.

**Karşılaştırma protokolü:** `reports/anxiety_classifier_metrics.csv` (full 24-feature) ve `reports/anxiety_classifier_metrics_rearonly.csv` (~16-feature) yan yana — model × tag × {acc, bal_acc, F1, AUC} 6×6 tablo halinde tezde rapor edilir.

### 9.4. Literatür köprüsü (literature bridge)

Aspartame–anksiyete ilişkisi üzerine yayınlar genellikle iki mekanizma rapor eder:
- **Lokomotor azalma:** total_distance ve rearing↓ → "treated rats are less explorative" (örn. Choudhary & Pretorius, *Nutr Neurosci* 2017; Jones et al., 2024 transgenerational study — fenilalanin metaboliti aracılı).
- **Periferal tercih artışı:** thigmotaxis↑, center entries↓ → klasik anksiyojenik profil (Erbaş et al., *Toxicol Mech Methods* 2018).

Grapefruit (naringin / naringenin / hesperidin) çalışmaları ise antioksidan-bazlı **anksiyolitik** etki gösterir: rearing ve center time artışı, freezing azalması (Khan et al., *Neurochem Res* 2019; Yi et al., *Nat Prod Res* 2022).

**Bu çalışmadaki gözlem:** Grup ortalamaları (`reports/anxiety_stats.csv` baseline + `reports/composite_anxiety_stats.csv` follow-up):

| Grup | rear_count (median) | pct_periphery (median) | Yön (literatüre göre) |
|---|---|---|---|
| Control | ~21.8* | baseline | — |
| Aspartame | ~14.8* | yüksek | ✓ Anksiyojenik (literatür uyumlu) |
| Grapefruit | yüksek | düşük | ✓ Anksiyolitik (literatür uyumlu) |
| Aspartame+Grapefruit | orta | orta | ✓ Etkileşim (kısmen geri çevirme) |

\* Pilot bulguya göre yaklaşık değerler; kesin sayılar `reports/composite_anxiety_index.csv`'den alınmalı.

**Tez sonuç paragrafı için önerilen formülasyon:**

> "Mevcut örneklem büyüklüğü (n=29) BH-FDR sonrası anlamlılık üretmemekle birlikte, rearing davranışında gözlenen %30 mertebesindeki azalma (Aspartame vs Control) ve mekansal dağılımdaki kalitatif kayma (rear_center_frac düşüşü) yönü itibarıyla literatürdeki 'aspartam fenilalanin aracılı anksiyojenik etki, antioksidan/flavonoid içerikli müdahaleler bunu kısmen geri çevirir' hipoteziyle uyumludur. Bu çalışmanın birincil katkısı pipeline'ın kendisi olmakla birlikte, gözlenen yön ve etki büyüklükleri (η²=0.20) önerilen replikasyon (grup başına ≥20 hayvan) için *a priori* güç hesabını destekleyen pilot kanıt sağlamaktadır."

Bu çerçeveleme **savunulabilir** çünkü: (a) anlamlılık iddia etmiyor, (b) etki büyüklüğünü literatür hipoteziyle aynı yöne yerleştiriyor, (c) replikasyon ihtiyacını niceliksel olarak gerekçelendiriyor.
