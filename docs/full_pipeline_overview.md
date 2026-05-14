# Anksiyete Tahmin Pipeline'ı — Tam İşleyiş Raporu

**Tarih:** 2026-05-14
**Branch:** `develop`
**Kapsam:** Bu doküman `docs/anxiety_report.md` (konsolide bilimsel rapor), `scripts/predict_anxiety_v2.py` (CLI inference), `scripts/predict_anxiety_v3_gui.py` (Streamlit GUI), `src/behavior_detection.py`, `src/anxiety/*.py`, `analysis/open_field/oft_metrics.py` ve eğitilmiş model artifact'larını birleştirip uçtan uca pipeline'ı **tek bir doküman içinde** anlatır.

> Bu, projenin **tek-stop pipeline referansıdır**: girdi formatı, threshold'lar, modüller, model mimarisi, eğitim mekanizması, bilimsel bulgular, sınırlılıklar ve replikasyon planı bir arada. Kod akışı + bilimsel kanıt iki katmanlı sunulur.

---

## İçindekiler

1. [Tek paragraflık özet](#1-tek-paragraflık-özet)
2. [Veri seti ve deney tasarımı](#2-veri-seti-ve-deney-tasarımı)
3. [Pipeline akışı — uçtan uca](#3-pipeline-akışı--uçtan-uca)
4. [Davranış tespiti — kural-bazlı detector detayı](#4-davranış-tespiti--kural-bazlı-detector-detayı)
5. [OFT metrikleri](#5-oft-metrikleri)
6. [Spatial rearing](#6-spatial-rearing)
7. [Feature matrix — 12 model özelliği](#7-feature-matrix--12-model-özelliği)
8. [Model eğitimi (offline)](#8-model-eğitimi-offline)
9. [Inference (her yeni sıçan)](#9-inference-her-yeni-sıçan)
10. [Streamlit GUI](#10-streamlit-gui)
11. [Bilimsel bulgular](#11-bilimsel-bulgular)
12. [Modelin karar mekanizması](#12-modelin-karar-mekanizması)
13. [Config mismatch incident (2026-05-11)](#13-config-mismatch-incident-2026-05-11)
14. [İterasyon tarihi](#14-iterasyon-tarihi)
15. [Sınırlılıklar](#15-sınırlılıklar)
16. [Power analizi ve replikasyon](#16-power-analizi-ve-replikasyon)
17. [Reproducibility — sıfırdan tam pipeline](#17-reproducibility--sıfırdan-tam-pipeline)
18. [Methods bölümü için hazır cümleler](#18-methods-bölümü-için-hazır-cümleler)
19. [Dosya haritası](#19-dosya-haritası)

---

## 1. Tek paragraflık özet

29 hayvanlık (Control n=5, Treated n=24 — aspartame + grapefruit + kombine) Open Field deneylerinden, DeepLabCut ile 9-keypoint posture verisi çıkarıldı. Kural-bazlı bir detector rearing/grooming bout'larını tespit etti; `analysis/open_field/oft_metrics.py` thigmotaxis/freeze/spatial entropy metriklerini hesapladı; `src/anxiety/spatial_rearing.py` her rear bout'unu center/wall olarak etiketledi. **12 rearing-odaklı feature** (regex `rear|pct_periphery|pct_freeze|spatial_entropy|comfort`) bir **L2 lojistik regresyon** sınıflandırıcısına LOOCV ile beslendi → **AUC 0.683, balanced accuracy 0.738, F1_treated 0.894**. Inference için iki eşdeğer arayüz dağıtıldı: CLI (`predict_anxiety_v2.py`) ve Streamlit GUI (`predict_anxiety_v3_gui.py`). Pipeline **bir "anksiyete dedektörü" değil**, tedavi-kaynaklı davranışsal sapma dedektörüdür; ana pilot bulgu spatial rearing'in yön sinyali (`rear_center_frac` MW p=0.030, Cohen's d=−0.92) — Bonferroni sonrası anlamlılığa ulaşmıyor ama replikasyon için a-priori temel.

---

## 2. Veri seti ve deney tasarımı

### 2.1. Hayvan dağılımı (n=29)

| Grup | Kohortlar | n | Tedavi | Beklenen yön (literatür) |
|---|---|---|---|---|
| Control | MA1, MA2 | 5 | Vehicle | baseline |
| Aspartame | MA3, MA4 | 8 | Aspartam | Anksiyogenik (fenilalanin metaboliti) |
| Grapefruit | MA5, MA6 | 8 | Grapefruit | Anksiyolitik (flavonoid antioksidanlar) |
| Aspartame+Grapefruit | MA7, MA8 | 8 | Kombine | Etkileşim / kısmi geri çevirme |

`is_treated = (group != "Control")` ile binary etiket — `class_weight="balanced"` zorunlu (5/24 oran).

### 2.2. Video ve pose

- **Video:** 30 fps, ~178 saniye, tek hayvan / arena.
- **Arena:** manuel anote edilmiş bounding box `(397, 777, 156, 535)` piksel (`src/anxiety/config.py`).
- **DLC modeli:** top-down 9-keypoint OFT modeli (PyTorch backend, RTX 3060). Filtered CSV → `data/DLCfiltered/<group>/<subject>/<subject>.csv`.

### 2.3. Inner zone — %20 margin kararı

```python
# src/anxiety/config.py
ARENA        = (397.0, 777.0, 156.0, 535.0)   # px, manuel anote
INNER_MARGIN = 0.20                            # literatür standardı
INNER_ZONE   = _derive_inner(ARENA, 0.20)      # → (473, 701, 232, 459)
                                                # inner ≈ %36 of arena alanı
```

Klasik OFT literatüründe inner zone arenanın %25–50'sidir (Choleris et al. 2001; Carter & Shieh 2010). %20 margin (inner ≈ %36) bu aralığa girer ve **thigmotaksis halkası ile center'ı net ayırır** — kritik metodolojik karar (§13'teki sensitivity analizi).

---

## 3. Pipeline akışı — uçtan uca

`predict_anxiety_v2.py` ve `predict_anxiety_v3_gui.py` aynı çekirdek fonksiyonları çağırır; GUI sadece v2'yi import edip sarmalar (`import predict_anxiety_v2 as v2`, satır 35).

```
Raw video (30 fps, ~178 s)
   ↓ DLC OFT 9-keypoint model                  src/dlc/
Filtered pose CSV                              data/DLCfiltered/<group>/<subject>/<subject>.csv
   ↓ Rule-based detector                       src/behavior_detection.py
Rearing + grooming bouts
   ↓ OFT metrics                               analysis/open_field/oft_metrics.py
Locomotion · thigmotaxis · freeze · entropy
   ↓ Spatial rearing                           src/anxiety/spatial_rearing.py
Her bout center / wall (univariate analiz)
   ↓ Feature matrix builder                    src/anxiety/profile.py  (training)
                                               predict_anxiety_v2.build_feature_dict (inference)
12 rear-only feature
   ↓ LOOCV classifier (LR / RF / SVM)          src/anxiety/classifier.py
models/anxiety_classifier/{lr,rf,svm,scaler}_rearonly.pkl

Inference (single subject):
   scripts/predict_anxiety_v2.py  <csv>        (CLI)
   streamlit run scripts/predict_anxiety_v3_gui.py  (Tarayıcı GUI)
```

Inference, GUI'de 4 progress adımına bölünür: bout tespiti (0.30) → OFT metrikleri (0.50) → spatial rearing (0.65) → tahmin (0.80) → rapor (0.90) → tamamlandı (1.0).

---

## 4. Davranış tespiti — kural-bazlı detector detayı

`src/behavior_detection.py`. Bu modül DLC posture eşiklerine dayalı **rule-based** bir detector; pre-validated sensitivity/specificity yok (sınırlılık §15.4).

### 4.1. Genel mantık

**Önce grooming, sonra rearing.** İkisi de 2D projeksiyonda bedeni sıkıştırır; ayırt edici "burun-ön ayak mesafesi" (grooming'de düşük, rearing'de değil).

```
GROOMING posture (4 sufficient branch — OR mantığı, hepsi body_vel < GROOM_MAX_VEL):
  (a) Tight posture   : nose2fp < GROOM_NOSE2FP_TIGHT, fp_hp_vert düşük
  (b) Loose posture   : fp_hp_vert düşük + body_still (1 sn pencere)
  (c) Upright/sit-up  : fp_hp_vert yüksek + nose_y top wall'a uzak
  (d) Nose-occluded   : fp_hp_vert yüksek + body_still + arena içinde

REARING (5 complementary cue — OR mantığı):
  R1 Compact rearing       : htdist < 75 AND NOT grooming_posture
  R2 Top-wall extended     : fp_hp_vert > 45 AND nose_y < 165
  R3 Bottom strong         : fp_hp_vert < −80 AND nose_y near bottom
  R4 Bottom compact        : fp_hp_vert < −45 AND htdist < 105 AND nose near bottom
  R5 Side-wall             : htd_y ∈ [40, 60] AND nose at L/R boundary
  R6 Wall-press            : nose beyond arena + htdist < 115

Final: grooming = grooming_posture AND NOT rearing
```

### 4.2. Threshold sabitleri

Tüm threshold'lar `src/behavior_detection.py` üst kısmında sabit; jüri "neden bu sayı" sorduğunda doğrudan kod yorumlarındaki gerekçeye gönderilebilir.

| Sabit | Değer | Anlam |
|---|---|---|
| `REAR_COMPACT_HTDIST` | 75 px | Compact rearing için head-to-tail mesafe üst sınırı |
| `REAR_EXTEND_FPHP` | 45 px | Top-wall rearing için hindpaw_y − forepaw_y alt sınırı |
| `REAR_NOSE_Y_MAX` | 165 px | Top-wall rearing için nose_y üst sınırı (küçük y = yukarı) |
| `REAR_SIDE_HTD_Y_MIN/MAX` | 40–60 px | Side-wall rearing için head-tail y aralığı |
| `REAR_BOTTOM_STRONG_FPHP` | −80 px | Bottom rearing (güçlü) için fp_hp_vert eşiği |
| `REAR_BOTTOM_COMPACT_FPHP` | −45 px | Bottom rearing (compact) için fp_hp_vert eşiği |
| `REAR_BOTTOM_HTDIST` | 105 px | Bottom compact rearing için body squish eşiği |
| `REAR_BOTTOM_NOSE_Y` | 500 px | Bottom rearing için nose_y alt sınırı |
| `REAR_WALL_PRESS_HTDIST` | 115 px | Wall-press rearing için body squish eşiği |
| `GROOM_MAX_FPHP` | 10 px | Grooming için forepaw NOT elevated above hindpaw |
| `GROOM_MAX_VEL` | 25 px/s | Grooming için max body velocity (≈stationary) |
| `BODY_STILL_RANGE` | 15 px | 30-frame penceresinde body_center yayılımı (still tanımı) |
| `LIKELIHOOD_THRESH` | 0.6 | DLC keypoint güven alt sınırı |
| `INTER_BOUT_GAP` | 15 frame | Bu boşluk altındaki bout'lar birleştirilir |
| `MIN_BOUT_FRAMES` | 10 frame | Daha kısa bout'lar atılır |
| `DEFAULT_FPS` | 30 | Video frame rate |

### 4.3. Frame → bout dönüşümü

```python
# scripts/predict_anxiety_v2.py:80-99 — v2.detect_bouts
raw     = load_dlc_csv(csv_path)
masked  = mask_low_likelihood(raw, LIKELIHOOD_THRESH=0.6)
feat    = bd_compute_features(raw, masked, fps=30)        # posture features
labels  = classify_frames(feat)                            # "rearing"/"grooming"/"other"
rear_b  = frames_to_bouts(np.where(labels == "rearing")[0],
                          gap=INTER_BOUT_GAP, min_dur=MIN_BOUT_FRAMES)
groom_b = frames_to_bouts(np.where(labels == "grooming")[0], ...)
```

`bouts_to_dataframe` çıktısı: kolonlar `behaviour, bout, start_frame, end_frame, start_s, end_s, duration_s`.

### 4.4. Ground-truth doğrulama

`GROUND_TRUTH_BY_SUBJECT` (kod içinde) bazı MA1_2 / MA5_1 / MA7_1 seans aralıkları için manuel anotasyon içeriyor (grooming bout sınırları). Sensitivity/specificity tüm dataset üzerinde sistemik değerlendirilmedi — sınırlılık.

---

## 5. OFT metrikleri

`analysis/open_field/oft_metrics.py`. `v2.compute_oft_metrics` (satır 102-111) bu modülün dört helper'ını ardı ardına çağırır.

```python
x, y = load_body_center(csv, likelihood=0.6, jump=60, smooth=5)
# smoothed body_center serisi (NaN için interp + 5-frame moving average)

locomotion_metrics(x, y, fps)   → {total_distance_px, mean_speed_px_s, ...}
thigmotaxis_metrics(x, y, INNER_ZONE) → {pct_time_periphery, pct_time_center,
                                          center_zone_entries}
freezing_metrics(x, y, fps)     → {pct_time_freeze, freeze_bout_count}
spatial_entropy(x, y, ARENA)    → 0–1 normalize Shannon entropi
```

**Spatial entropy formülü:**
```
H(p) = −Σᵢ pᵢ log pᵢ        # arenayı NxN grid'e böl, pᵢ = i hücresinde geçirilen kare oranı
H_norm = H(p) / log(N²)      # 0 (tek hücreye sıkışmış) ↔ 1 (üniform yayılım)
```

**Freezing tanımı:** `oft_metrics.py` default'ta hız < 5 px/s eşiğinin altındaki ardışık kare blokları freeze sayılır; sub-second jitter'ı filtrelemek için bir minimum-duration uygulanır.

---

## 6. Spatial rearing

`src/anxiety/spatial_rearing.py` (eğitim CSV'sini üretir) ve `v2.compute_spatial_rearing` (inference). Her rearing bout için:

1. Bout süresince body_center'ın **medyan (x, y)** noktasını al.
2. `INNER_ZONE = (473, 701, 232, 459)` içinde mi? → `center` / değilse `wall` / NaN → `unknown`.
3. Bout listesini ve count/duration özetlerini biriktir.

`v2.compute_spatial_rearing` dönen alanlar (`scripts/predict_anxiety_v2.py:114-146`):

```
rear_count_center      : merkezde başlayan rearing bout sayısı
rear_count_wall        : duvar yakınında başlayan rearing bout sayısı
rear_count_unknown     : sınıflanamayan (NaN body_center) bout sayısı
rear_center_frac       : center / (center + wall)
rear_total_s_center    : center bout'larının toplam süresi
rear_total_s_wall      : wall bout'larının toplam süresi
rear_center_minus_wall : center - wall (count cinsinden)
_bout_zones            : her bout için {bout, start_s, duration_s, median_x,
                          median_y, zone} — overlay PNG'sinde renkli nokta
```

> **Kritik:** Bu altı spatial-rearing varyantı modelin **input feature seti içinde değil**. 18→12 retrain'de LOOCV varyansını düşürmek için (multicollinearity) **kaldırıldı**. Hâlâ:
> - Univariate kanıt katmanında (`reports/spatial_rearing_stats.csv`) — pilot bulgu.
> - Overlay görselinde (turuncu daire = wall, lime yıldız = center).
> - JSON raporda explicit ayrı blok olarak.

---

## 7. Feature matrix — 12 model özelliği

`v2.build_feature_dict` (satır 149-202) toplam **29 alanlık** sözlük üretir; model bunlardan regex `rear|pct_periphery|pct_freeze|spatial_entropy|comfort` ile eşleşen 12'sini kullanır.

### 7.1. Modelin kullandığı 12 feature

**Davranışsal — rearing (8):**

| Feature | Formül / Anlam |
|---|---|
| `rear_count` | rearing bout sayısı |
| `rear_total_s` | Σ rear_bout duration |
| `rear_pct` | 100 × rear_total_s / session_s |
| `rear_mean_bout_s` | rear_total_s / rear_count |
| `rear_rate_per_min` | 60 × rear_count / session_s |
| `rear_early_frac` | İlk 90 s içindeki rear süresi / toplam (EARLY_S sabit) |
| `rear_groom_ratio` | rear_total_s / groom_total_s |
| `rear_per_100px` | 100 × rear_count / total_distance_px (yatay aktiviteden bağımsız dikey ölçü) |

**Mekansal (3):**

| Feature | Anlam |
|---|---|
| `pct_periphery` | Duvar kenarında geçen zaman % (thigmotaxis) |
| `pct_freeze` | Donakalma süresinin yüzdesi |
| `spatial_entropy` | 0–1 normalize Shannon entropi (mekansal yayılım) |

**Türetilmiş (1):**

| Feature | Formül |
|---|---|
| `comfort_ratio` | groom_total_s / pct_time_center — merkezdeki grooming yoğunluğu |

### 7.2. Modele girmeyen 17 feature

`build_feature_dict` ayrıca şunları da üretir (CSV'de var, model regex'i seçmedi):

```
pct_center, center_entries, total_distance_px, mean_speed_px_s,
freeze_bout_count, groom_count, groom_total_s, groom_pct,
groom_mean_bout_s, groom_bout_cv, groom_early_frac,
rear_count_center, rear_count_wall, rear_center_frac,
rear_total_s_center, rear_total_s_wall, rear_center_minus_wall
```

Bunlar **GUI'nin Metrikler sekmesinde** ve **JSON raporda** gösterilir, ama prediction logit'ine girmez. Spatial-rearing 6'lısı univariate kanıt olarak ayrı katmandadır (§11.2).

---

## 8. Model eğitimi (offline)

`src/anxiety/classifier.py`. Bir kez çalıştırılır; sonuçlar pickle olarak kaydedilir.

### 8.1. Üç model — hepsi `class_weight="balanced"`

```python
LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                   class_weight="balanced", max_iter=1000, random_state=42)
RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2,
                       class_weight="balanced", random_state=42, n_jobs=-1)
SVC(kernel="rbf", C=1.0, gamma="scale",
    class_weight="balanced", probability=True, random_state=42)
```

`class_weight="balanced"` Control:Treated = 5:24 dengesizliğini Control loss'una ~4.8× ağırlık vererek kompanse eder. Bu olmadan modeller her zaman "Treated" tahmin ederdi (trivial accuracy = 24/29 = 0.83).

### 8.2. LOOCV mekanizması

`loocv_run` (satır 135-160):

```python
for train_idx, test_idx in LeaveOneOut().split(X):
    X_tr, X_te = X[train_idx], X[test_idx]    # 28 train, 1 test
    y_tr       = y[train_idx]

    # 🔑 Imputer ve Scaler fold-içi fit edilir — data leakage yok
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_te_i = imp.transform(X_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_i)
    X_te_s = sc.transform(X_te_i)

    m = model.__class__(**model.get_params())  # taze model
    m.fit(X_tr_s, y_tr)

    pred[test_idx]  = m.predict(X_te_s)
    proba[test_idx] = m.predict_proba(X_te_s)[:, 1]
```

Bu 29 kere döner; sonunda 29 prediction + proba tüm hayvanlar için biriktirilir → AUC, balanced accuracy, F1 bunların üzerinden hesaplanır.

### 8.3. Tam-veri yeniden eğitim + pickle

`fit_full_and_save` (satır 228-260) LOOCV bittikten **sonra** tüm 29 hayvanla bir kez daha fit eder ve pickle'lar:

```
models/anxiety_classifier/
├── scaler_rearonly.pkl   ← {imputer, scaler, features}  ← inference için
├── lr_rearonly.pkl       ← LogReg                       ← inference için
├── rf_rearonly.pkl       ← RandomForest
└── svm_rearonly.pkl      ← SVM-RBF
```

**Önemli nüans:** LOOCV metrikleri tüm 28-fold modellerin generalization tahminidir; inference'ta kullanılan pickle ise 29-of-29 ile eğitilmiş tek modeldir. AUC raporu **bu pickle'a değil**, LOOCV ensemble'a aittir.

### 8.4. Çıktılar

```
reports/anxiety_classifier_metrics_rearonly.csv      ← LR/RF/SVM × {acc, bal_acc, F1, AUC}
reports/anxiety_classifier_predictions_rearonly.csv  ← fold başına subject_id, true, pred, proba
reports/anxiety_classifier_importance_rearonly.csv   ← RF importance + LR coef (feature başına)
reports/figures/anxiety_classifier_cm_rearonly.png   ← 3 model confusion matrix
reports/figures/anxiety_classifier_roc_rearonly.png  ← aggregate ROC eğrileri
```

### 8.5. Tek-komut retrain helper

```bash
bash scripts/retrain_anxiety_rearonly.sh
```

İki adımı sırayla çalıştırır:

1. `python -m src.anxiety.spatial_rearing` → `data/spatial_rearing.csv` + `data/anxiety_features_extended.csv` mevcut `INNER_ZONE` ile yeniden üretir.
2. `python -m src.anxiety.classifier --csv ... --features '...' --tag rearonly` → pickle'ları yeniden yazar.

**Her zaman çalıştır** when:
- `INNER_ZONE` / `ARENA` / `FPS` değişti
- Yeni subject eklendi
- `spatial_rearing.py` veya `oft_metrics.py` kodu değişti

Neden zorunlu → §13 config mismatch incident.

---

## 9. Inference (her yeni sıçan)

### 9.1. `predict_anxiety_v2.predict` adım adım

`scripts/predict_anxiety_v2.py:224-254`:

```python
cols   = bundle["features"]                  # 12 isimden oluşan liste
imp    = bundle["imputer"]                   # eğitimde fit edilmiş SimpleImputer
sc     = bundle["scaler"]                    # eğitimde fit edilmiş StandardScaler

x      = [[feat[c] for c in cols]]            # 12 değer, NaN olabilir
x_imp  = imp.transform(x)                    # median imputation (eğitim medyanı)
x_std  = sc.transform(x_imp)                 # z-score (eğitim mean/std)

pred   = model.predict(x_std)[0]             # 0 / 1
proba  = model.predict_proba(x_std)[0]       # [P_control, P_treated]

# Top-5 push
coef    = model.coef_.ravel()                # 12-uzun
contrib = coef * x_std[0]                    # her feature için push
top5    = abs(contrib) sıralı ilk 5
```

### 9.2. sklearn 1.7+ uyumluluğu

```python
# scripts/predict_anxiety_v2.py:233-234
if not hasattr(model, 'multi_class'):
    model.multi_class = 'auto'
```

Eski sklearn (≤1.4) ile pickle'lanan LogReg modeli sklearn 1.7+ altında yüklenirken `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'` atar. Bu shim onu engeller (commit `ad53e74`).

### 9.3. Çıktı dosyaları

`reports/anxiety_predictions_v2/` altında her sıçan için:

| Dosya | İçerik |
|---|---|
| `<subject>_anxiety_v2_report.txt` | Türkçe okunabilir özet (`render_summary`, satır 263-302) — anahtar metrikler + tahmin + top-5 push |
| `<subject>_anxiety_v2_report.json` | Tüm 29 feature + tahmin + JSON contributors + spatial_rearing bloğu |
| `<subject>_anxiety_v2_overview.png` | Arena overlay (`plot_overview`, satır 305-339) |
| `<subject>_behavior_bouts.csv` | Ham bout dataframe (debugging) |

**Overlay PNG detayı:** Arena çerçevesi siyah dolgu çizgi; iç bölge kesik çizgi. Body trajectory `tab:red` (Treated tahmin) veya `tab:blue` (Control tahmin) çizilir alpha=0.35. Rearing noktaları:
- Wall rear → turuncu daire (s=80, edge=siyah)
- Center rear → lime yıldız (s=110, marker="*", edge=siyah)
- Y ekseni `invert_yaxis()` ile çevrilir (görüntü koordinatları uyumu).

---

## 10. Streamlit GUI

`scripts/predict_anxiety_v3_gui.py` — v2'nin saf fonksiyonlarını tarayıcıda sarmalar. Bilimsel mantık eklemez.

### 10.1. Sayfa yapısı

1. **Sidebar (sabit):** Model özeti tablosu — `Lojistik Regresyon` / `LOOCV n=29` / `12 feature` / `AUC 0.683` / `F1 0.894` / `Doğruluk %82.8`. 3-adım kullanım kılavuzu.
2. **Hero header:** Gradient banner + algoritma/sample size/AUC/F1 chip'leri.
3. **Dosya yükleme:** Sürükle-bırak CSV alanı (Türkçe etiketli — özel CSS) + ▶ Çalıştır butonu.
4. **Progress bar:** 4 adımlı (yukarıdaki §3 ile aynı yüzdeler).
5. **4 sekme sonuç paneli:**

| Sekme | İçerik |
|---|---|
| 🎯 **Tahmin** | Arena overlay (1.35:1 oranıyla geniş) · Tahmin kartı (Tedavi/Kontrol Grubu + güven %) · Olasılık barları (`fig_probability_bars`) · 4 temel metrik (merkez %, çevre %, donakalma %, rear sayısı) açılır panelde · Karar katkıları yatay bar grafiği (`fig_feature_contributions`) — kırmızı = Tedavi yönü, mavi = Kontrol yönü · Top-5 feature açılır panel: Türkçe ad + açıklama + formül + ölçülen değer + z-skoru + push değeri |
| 🧠 **Davranış** | `run_open_field_analysis` (subprocess) çıktıları: davranış zaman çizgisi PNG, thigmotaxis görseli + stat tablosu (merkez/çevre/diğer %), KDE ısı haritası, bout tablosu ilk 20 satır |
| 📊 **Metrikler** | 12 model feature'ı için açılır panel (Türkçe ad + açıklama + formül) iki sütun grid · Davranış özet kartları (rearing/grooming/freezing, count/duration/%) · Mekansal Rearing kutusu (merkez/duvar/merkez oranı 3 sütunlu metric) |
| ⬇ **İndir** | TXT/JSON/PNG için 3 download butonu (önizleme expander'lı) · Analiz görselleri için ayrı PNG butonları |

### 10.2. `FEATURE_INFO` sözlüğü

GUI'nin satır 258-461 arasında her feature için **isim (uzun + kısa), Türkçe açıklama, formül** içeren bir sözlük tutulur. Bu sözlük 28 feature'ı kapsar (model 12'sini kullansa da, GUI hepsini gösterir). Örnek:

```python
"rear_early_frac": {
    "name":       "Erken Rearing Oranı",
    "name_short": "Erken Rearing",
    "desc": "Farenin hareketinin erken bölümündeki (ilk 90 saniye) "
            "rearing'in toplam rearing'e oranı. Yüksek değer hareketin "
            "başında yoğun keşfi, düşük değer sonradan ısınan keşfi gösterir.",
    "formula": "erken_pencerede_rear_süresi / toplam_rear_süresi",
},
```

### 10.3. Opsiyonel "tam OFT analiz" subprocess

`run_open_field_analysis` (satır 567-648):

```python
subprocess.Popen([sys.executable,
                  analysis/open_field/run_analysis.py,
                  "--arena", *arena_bounds,
                  "--inner-zone", *inner_zone_bounds,
                  "--csv", csv_path,
                  "--fps", "30", "--likelihood", "0.6",
                  "--jump-thresh", "60", "--smooth", "5",
                  ...])
```

Üretilen artifact'lar dizinde glob ile taranıp GUI'de gösterilir:

```
<subject>_behavior_timeline.png      <subject>_orbit_grid.png
<subject>_orbit_bp_<bodypart>.png    <subject>_thigmotaxis.png
<subject>_bodypart_bp_<bodypart>.png <subject>_heatmap_kde.png
<subject>_bodypart_heatmaps.png      <subject>_heatmap_histogram.png
<subject>_behavior_bouts.csv         <subject>_behavior_frames.csv
```

**Bu adım pipeline kararını değiştirmez** — sadece görsel zenginlik. `fast_mode=True` heatmap+bodypart adımlarını atlatır. Timeout 600 sn.

### 10.4. GUI varsayılanları

`scripts/predict_anxiety_v3_gui.py:698-709` — sidebar'da artık seçilemez, kod-içi sabit:

```python
fps             = 30.0          # v2.DEFAULT_FPS
tag             = "rearonly"    # model variant
run_analysis    = True          # tam OFT analiz aç
arena_bounds    = (397, 777, 156, 535)
auto_inner      = True          # %20 margin otomatik türet
save_to_reports = True          # reports/anxiety_predictions_v2/ altına da yaz
fast_mode       = False         # tüm görseller dahil
```

`auto_inner=True` ile inner zone GUI kodunda da %20 margin'la türetilir (satır 822-828) — `src/anxiety/config.py` ile birebir aynı formül, ayrı kaynaktan kaçınılmış.

---

## 11. Bilimsel bulgular

### 11.1. Univariate (Kruskal-Wallis, 4 grup)

`reports/anxiety_stats.csv` — top 6 sinyal:

| Özellik | H | p | η² | Yorum |
|---|---|---|---|---|
| **`rear_count`** | **8.03** | **0.045** | **0.201** | Large effect, uncorrected p<0.05 |
| `rear_rate_per_min` | 6.86 | 0.076 | 0.155 | Medium-large, marjinal |
| `rear_early_frac` | 6.25 | 0.100 | 0.130 | Medium |
| `total_distance_px` | 5.34 | 0.148 | 0.094 | Small-medium |
| `rear_total_s` | 5.21 | 0.157 | 0.088 | Small-medium |
| `rear_pct` | 4.93 | 0.177 | 0.077 | Small |

Top 6'nın 5'i rearing → **sinyal rearing davranışında yoğunlaşmış**. BH-FDR sonrası anlamlılığa ulaşmıyor.

### 11.2. Spatial rearing — ana bulgu

`reports/spatial_rearing_stats.csv`:

| Metrik | Control medyan | Treated medyan | MW U p | Cohen's d | Yön |
|---|---|---|---|---|---|
| **`rear_center_frac`** | 0.133 | 0.075 | **0.030** | **−0.92** | Treated merkezde daha az rearing |
| **`rear_total_s_center`** | 3.84 s | 0.68 s | **0.045** | **−0.66** | Treated merkez-rearing süresi ~5× düşük |
| `rear_center_minus_wall` | −11 | −15 | 0.049 | −0.87 | (aynı bulgunun farklı parametrizasyonu) |
| `rear_count_wall` (4-grup KW) | 13 | 18 | KW p=0.043, η²=0.21 | +0.76 | Treated daha çok duvar-rearing |

**Yorum:** Toplam rearing sayısı eşit olsa bile, *nerede* şahlandığı kalitatif ayırıcı. Treated hayvanlar duvar tarafına kayıyor (anksiyete-benzeri kaçış-yolu arama). **En güçlü pilot bulgu.** Bonferroni 7-test (0.030 × 7 = 0.21) eşiği geçmez → uncorrected pilot kanıt.

### 11.3. Binary classifier — Control vs Treated

`reports/anxiety_classifier_metrics_rearonly.csv` — rear-only 12-feature LOOCV (n=29, `class_weight=balanced`):

| Model | Accuracy | Bal-Acc | F1_treated | F1_control | **AUC** |
|---|---|---|---|---|---|
| **Logistic Regression L2** | **0.83** | **0.74** | **0.89** | **0.55** | **0.683** |
| Random Forest | 0.76 | 0.46 | 0.86 | 0.00 | 0.525 |
| SVM-RBF | 0.76 | 0.46 | 0.86 | 0.00 | 0.358 |
| LR L2 (24-feature baseline, hist.) | 0.72 | 0.60 | 0.83 | 0.33 | 0.57 |

LR balanced accuracy 0.738 ile sınıf-dengesiz n=29'da en güvenilir metrik; AUC ±0.10 LOOCV varyansı normal. RF/SVM minority sınıfı düşürdü (F1_control = 0); LR doğru metrikte tek sağlam model.

**Baseline kontrol:** Trivial "her zaman Treated" → accuracy = 0.83 (ham accuracy yanıltıcı). Doğru metrik: **balanced accuracy + AUC**.

### 11.4. Feature importance (final 12-feature)

`reports/anxiety_classifier_importance_rearonly.csv` — Random Forest importance:

| Sıra | Feature | RF importance |
|---|---|---|
| 1 | `rear_early_frac` | 0.219 |
| 2 | `pct_periphery` | 0.164 |
| 3 | `spatial_entropy` | 0.123 |
| 4 | `pct_freeze` | 0.087 |
| 5 | `rear_rate_per_min` | 0.069 |
| ... | | |
| 12 | `rear_count` | 0.028 |

Classifier şu kombinasyona dayanıyor: **keşif zamanlaması** (`rear_early_frac`), **periferik bağlanma** (`pct_periphery`), **uzaysal dağılım** (`spatial_entropy`), **donma** (`pct_freeze`). Genel rearing temposu (`rear_count`) en zayıf discriminator.

### 11.5. Composite anxiety index — negatif bulgu

Üç birleşik skor `src/anxiety/composite_index.py`:

| Skor | Formül | KW p | Cohen's d |
|---|---|---|---|
| `rear_count` (baz) | — | 0.045 | +0.57 |
| `pct_periphery` (baz) | — | 0.290 | +1.07 |
| `AI_v1_thigmo_per_rear` | `pct_periphery / (rear_count + 1)` | 0.064 | +0.46 |
| `AI_v2_passive_per_rear` | `(pct_periphery + pct_freeze) / (rear_count + 1)` | 0.060 | +0.00 |
| `AI_v3_thigmo_x_low_rear` | `pct_periphery × (1 − rear_pct/100)` | 0.394 | +0.87 |

Hiçbir kompozit `rear_count`'u geçemedi. Pay/payda zaten feature setinde → lineer ratio yeni varyans katmıyor. **Negative finding** olarak tezde raporlanır.

### 11.6. PC1 ekseni — circularity uyarısı

PCA yönelimsel sıralama (PC1 işaret çapası `pct_periphery+`):

| Grup | n | PC1 ortalama | %95 CI |
|---|---|---|---|
| Control | 5 | **+2.11** | [−0.76, +5.22] |
| Aspartame | 8 | −0.23 | [−1.56, +1.15] |
| ASP+GF | 8 | −0.51 | [−1.44, +0.40] |
| Grapefruit | 8 | −0.58 | [−2.45, +2.33] |

Tüm CI'lar sıfırı içeriyor → grup ayrımı PC1'de anlamlı değil. Ridge regression R²=1.00 trivial (PC1 zaten feature lineer kombinasyonu → circular). RF R²=0.84 daha gerçekçi ama yine circular. Bilimsel iddia regresyon R²'sinde değil §11.2'deki yön / sıralamadadır.

---

## 12. Modelin karar mekanizması

### 12.1. Mimari

```
12 raw feature (per subject)
   ↓ SimpleImputer(median)              fold-içi fit (training)
                                         eğitim medyanı uygulanır (inference)
   ↓ StandardScaler(z-score)            fold-içi fit (training)
                                         eğitim mean/std uygulanır (inference)
   ↓ LogisticRegression(L2, C=1.0, class_weight=balanced)
   ↓ logit = intercept + Σᵢ coefᵢ × zᵢ
   ↓ P(Treated) = sigmoid(logit)
   ↓ predict = 1 if P > 0.5 else 0
```

Karar ham özelliklerin **ham değerlerine** değil **z-skorlarına** dayanır: bir feature'ın katkısı `coefᵢ × zᵢ` — modelin o feature'a verdiği önem (`coef`) **çarpı** bu hayvanın eğitim ortalamasından sapması (`z`).

### 12.2. "Push" kavramı

Raporlarda her feature için `push = coefᵢ × zᵢ`. İşaret:

- **`push > 0`** → Treated yönüne çekiş (GUI'de kırmızı)
- **`push < 0`** → Control yönüne çekiş (GUI'de mavi)

Output'taki top-5 push (mutlak değerce sıralı) görsel kolaylık için; geri kalan 7 + intercept de toplama dahildir.

**Karar oy çokluğu değil, ağırlıklı toplam.** Üç feature Treated derken iki Control derse, magnitude (büyük |coef|×|z|) ne tarafta ise kazanan o olur.

### 12.3. Modelin biyolojik olarak öğrendiği

En güçlü 5 katsayı (`reports/anxiety_classifier_importance_rearonly.csv`):

| Feature | LR coef | Anlam |
|---|---|---|
| **`rear_early_frac`** | **−1.50** | Yüksek erken rearing = sağlıklı keşif = **Control** |
| **`pct_periphery`** | **+1.09** | Çok periferde kalmak = anksiyete-benzeri = **Treated** |
| **`spatial_entropy`** | **+0.79** | Dağınık keşif = **Treated** |
| **`pct_freeze`** | **+0.48** | Çok donakalma = **Treated** |
| **`comfort_ratio`** | **+0.41** | Grooming / center yüksek = **Treated** (rahatlama davranışı) |

Yön literatürle uyumlu — özellikle yüksek `pct_periphery` → anksiyete-benzeri.

### 12.4. Vaka çalışması: MA4_1 (gerçek grup: Aspartame)

GUI/CLI output (retrain sonrası):

```
🟧 TAHMİN: Treated-benzeri    (P(Treated)=0.78, %78 güven)
   GERÇEK GRUP: Aspartame     ✓ doğru sınıflandı

Anahtar metrikler:
  pct_periphery     = 87.4%    (yüksek — duvar-tutuş)
  pct_freeze        =  4.4%
  rear_count        = 20
  rear_center_frac  = 0.000    (hepsi duvarda)

Top-5 push (örnek 18-feature dönemi verisinden):
  rear_early_frac   z=+0.77   push=−1.16  → Control
  spatial_entropy   z=+0.66   push=+0.52  → Treated
  pct_periphery     z=+0.48   push=+0.52  → Treated
  rear_mean_bout_s  z=+1.67   push=+0.50  → Treated
  rear_groom_ratio  z=+0.86   push=+0.35  → Treated

Top-5 toplam:        +0.73 (Treated yönü)
+ kalan 7 feature + intercept:  ≈ +0.5
= logit:              ≈ +1.27
P(Treated) = sigmoid(1.27) = 0.78  ✓
```

**Yorum:** 4 feature Treated, 1 Control diyor — ama tek başına `rear_early_frac`'in push'u (−1.16) diğer 4'ün toplam push'una (+1.89) yakın. Sonuç dar bir Treated kazancı, %78 güven.

### 12.5. Uç z-skor uyarısı (pratik kural)

`|z| > 3` olan bir feature → training dağılımının dışında. Bu işaret ya outlier hayvanı ya da feature-shift bug'ı düşündürür. MA4_1'de max `|z| = 1.67` → sağlıklı tahmin. Otomatik uyarı henüz kodda yok (`anxiety_report.md` §6.5 öneri olarak duruyor):

```python
extreme = [c for c in pred_info["top_contributors"] if abs(c["z"]) > 3]
if extreme:
    warn("Aşırı uç z-skor: " + ", ".join(c["feature"] for c in extreme))
```

---

## 13. Config mismatch incident (2026-05-11)

Rapor öncesi yakalanıp düzeltilen kritik bug. Future-proofing için tam belgelendi.

### 13.1. Belirti

Bilinen Control hayvanı (MA1_1) için v2 inference output:

```
rear_center_frac  : 0.143   (1 center / 7 bout)
Tahmin            : Control-benzeri (P_control = 1.000)

Top push (BEFORE retrain):
  pct_periphery     z=-4.04   push=−4.37 → Control
  rear_early_frac   z=+2.02   push=−3.02 → Control
  rear_center_frac  z=-8.36   push=−3.00 → Control
```

`rear_center_frac` için z = −8 ile −10 → training dağılımının çok dışında. Bu büyüklük "feature-shift" işaretiydi.

### 13.2. Kök neden

Commit `dd5943b` (2026-05-10): `INNER_ZONE` ~%7 margin (inner ≈ %85 of arena) → %20 margin (inner ≈ %36 of arena). **Bu commit'ten ÖNCE eğitilen model:**

- Training data `rear_center_frac` 0.85-1.0 aralığında (geniş center → her şey center sayıldı).
- Model "rear_center_frac ≈ 1 → Treated" öğrendi (Treated'in tavanda kümelendiği patolojik dağılımda).

**v2 inference YENİ config kullanıyor:**

- Aynı hayvanlar için değerler artık 0.0-0.2 aralığında (dar center → çoğu bout wall).
- Model "bu değer training'deki hiçbir Treated örneğine benzemiyor" diyerek tüm hayvanları Control'a itiyor.

Sonuç: bilinen-Treated hayvanlar Control olarak sınıflanıyordu.

### 13.3. Düzeltme

`scripts/retrain_anxiety_rearonly.sh` ile:
- `data/spatial_rearing.csv` yenilendi (yeni narrow center ile).
- `data/anxiety_features_extended.csv` güncellendi.
- LR / RF / SVM modelleri yeniden eğitildi.
- `reports/anxiety_classifier_importance_rearonly.csv` yeni coefficient'larla yazıldı.

**MA4_1 sanity check (retrain sonrası):** doğru sınıflandı (P_treated=0.78, top-5'te `rear_center_frac` yok → narrow-zone calibrated, sağlıklı z-skorlar).

### 13.4. Yardımcı düzeltme: hardcoded reference

`predict_anxiety_v2.render_summary` içindeki hardcoded "Control medyan 0.133, Treated medyan 0.075" satırı yeni-zone medianlarını gösteriyordu ama model eski-zone'la eğitilmişti → yanlış sezgi veriyordu. Yeni satır `reports/spatial_rearing_stats.csv`'ye yönlendiriyor.

### 13.5. Ders

**Config sabitleri (zone, margin, arena) feature sürümünün parçası.** Değiştirirken training data + model + reports'un hepsi senkron yenilenmeli; aksi halde inference distribution ile training distribution çelişir.

**Önlem 1 (uygulanan):** `retrain_anxiety_rearonly.sh` helper.
**Önlem 2 (uygulanan):** v2'deki yanıltıcı hardcoded medianlar kaldırıldı.
**Önlem 3 (öneri):** v2/v3'e `|z| > 3` otomatik uyarısı eklemek (henüz yok).

### 13.6. Inner zone sensitivity analizi (kanıt)

| Metrik | %20 margin (final) | ~%7 margin (eski manuel) |
|---|---|---|
| `rear_center_frac` MW p | **0.030** | 0.281 |
| `rear_center_frac` Cohen's d | **−0.92** | +0.31 (yön TERS) |
| `rear_count_wall` (Treated medyan) | 18 | 1 |
| LR LOOCV AUC (18-feat, hist.) | 0.73 | 0.62 |
| LR LOOCV AUC (12-feat, final) | **0.683** | — |

Manuel zone'da neredeyse tüm rear bout'ları "center" sınıfına düşüyor → sınıf sabit → grup ayrımcı sinyal yok.

---

## 14. İterasyon tarihi

| İterasyon | Değişen | LR AUC |
|---|---|---|
| 1 | Full 24 feature, default zone | 0.57 |
| 2 | Rear-only 18 feature, %20 zone (ilk eğitim) | 0.73 |
| 3 | Manuel zone (sensitivity test, ~%7 margin) | 0.62 |
| 4 | %20 margin lock + retrain (config bug fix, 18 feat) | 0.617 |
| 5 | **12 feature** (6 spatial-rearing varyantı kaldırıldı — multicollinearity / LOOCV varyans) | **0.683 (final)** |

**Kritik:** `python -m src.anxiety.classifier` her çalıştırıldığında model **sıfırdan eğitilir** (imputer.fit, scaler.fit, LR.fit). Bu **fine-tuning DEĞİL** — sklearn LR/RF/SVM'lerinde pre-trained ağırlıkları başlangıç alıp devam etme paradigması yoktur. Her iterasyonda **konfigürasyon kararı** öğreniliyor, ağırlıklar değil.

### 14.1. Sürüm geçmişi (özet)

| Tarih | Sürüm | Değişiklik |
|---|---|---|
| 2026-05-09 | v0 | İlk çalışmalar — `src/features.py` + `src/train_baseline.py` (4-target LOOCV) |
| 2026-05-10 sabah | v1 | Anxiety pivot — `src/anxiety/{profile,classifier,regression}.py` |
| 2026-05-10 öğle | v1.5 | Composite index (negatif bulgu) eklendi |
| 2026-05-10 öğleden sonra | v2-prep | Spatial rearing eklendi, ana bulgu (d=−0.92) yakalandı |
| 2026-05-10 akşam | v2 | `predict_anxiety_v2.py` end-to-end inference |
| 2026-05-11 | v2.5 | `archive/` reorganization, `docs/REPO_MAP.md`, README yenilendi |
| 2026-05-11 | v3 | Streamlit GUI (`predict_anxiety_v3_gui.py`) |
| 2026-05-11 | v3.1 | **Config mismatch bug yakalandı**, retrain helper eklendi |
| 2026-05-11 | v3.2 | 18 → 12 feature. LR AUC 0.617 → 0.683, F1_control 0.364 → 0.545 |

---

## 15. Sınırlılıklar

1. **Örneklem (n=29).** Davranışsal nörobilim alt-sınırının altında. Power ~%30-40 — gerçek orta-büyük etki bile %60 olasılıkla anlamlılığa ulaşmayabilir.
2. **Control / Treated dengesizliği.** 5 / 24 → her binary classifier için zorlu setting. `class_weight=balanced` zorunlu, ama outlier sensitivity yüksek.
3. **Treated etiketi heterojen.** Aspartam (anksiyogenik) + Grapefruit (anksiyolitik) + Aspartame+Grapefruit (etkileşim) birlikte etiketlenmiş. Model "anksiyete" değil "**tedavi-kaynaklı davranışsal sapma**" tespit ediyor.
4. **Davranış tespiti kural-bazlı.** Rearing / grooming detector'ı 9 keypoint posture eşiklerine dayanır. Pre-validated sensitivity / specificity yok. `GROUND_TRUTH_BY_SUBJECT` sadece bazı seans-aralıkları kapsıyor.
5. **Tek seans / arena.** Test-retest tekrarlanabilirliği ölçülmedi; bireysel varyans ile gerçek tedavi etkisi ayrıştırılamıyor.
6. **`predict_proba` kalibre değil.** LR `class_weight=balanced` ile eğitildi; mutlak olasılık değerleri (0.78 vs 0.85) **sıralama amaçlı** kullanılmalı, mutlak risk skoru olarak değil.
7. **PC1 yorumlaması circular.** PC1 zaten feature'ların lineer kombinasyonu.
8. **Held-out test seti yok.** n=29 kısıtlı olduğu için tüm değerlendirme LOOCV içinde. Out-of-distribution genelleme bilinmiyor.

---

## 16. Power analizi ve replikasyon

### 16.1. Gözlenen etki büyüklükleri

| Analiz | Etki büyüklüğü | Power tahmini (n=6/grup) |
|---|---|---|
| `rear_center_frac` MW (Control vs Treated) | d = −0.92 | ~%60 |
| `rear_count_wall` 4-grup KW | η² = 0.21 | ~%50 |
| `rear_count` 4-grup KW | η² = 0.20 | ~%45 |
| `rear_early_frac` 4-grup KW | η² = 0.13 | ~%30 |

### 16.2. %80 power için gereken n

| Hedef etki | Grup başına n | Toplam |
|---|---|---|
| d = 0.92 (rear_center_frac) | ~22 | ~88 |
| η² = 0.20 (rear_count) | ~14 | ~56 |
| η² = 0.13 (rear_early_frac) | ~22 | ~88 |
| η² = 0.06 (orta-küçük) | ~52 | ~210 |

### 16.3. Replikasyon protokolü

1. **n ≥ 20-22 / grup** (toplam ~80-90) — mevcut etki büyüklüklerini %80 güçte doğrulamak için.
2. **Doz-yanıt tasarımı** → tek doz yerine 3 doz × kontrol → "etki var mı" yerine "doz-yanıt eğrisi nedir".
3. **Within-subject baseline** → her hayvanın pre-treatment baseline'ı; varyansı yarıdan fazla düşürür.

---

## 17. Reproducibility — sıfırdan tam pipeline

```bash
# 1. DLC pose CSV'leri data/DLCfiltered/<group>/<subject>/<subject>.csv altında

# 2. Davranış tespiti — her hayvan için
python -m src.behavior_detection

# 3. OFT metrikleri ve trajectory görselleri (batch)
python analysis/open_field/run_kare_batch.py
# veya tek subject:
python analysis/open_field/run_analysis.py --arena 397 777 156 535

# 4. Anxiety feature matrix + PCA + univariate KW
python -m src.anxiety.profile
#   → data/anxiety_features.csv
#   → reports/anxiety_pca.png + anxiety_boxplots.png + anxiety_stats.csv

# 5. Composite index (negatif bulgu, dürüstlük için)
python -m src.anxiety.composite_index
#   → reports/composite_anxiety_*.csv

# 6 + 7. Spatial rearing + retrain (tek komut):
bash scripts/retrain_anxiety_rearonly.sh
#   = python -m src.anxiety.spatial_rearing
#   + python -m src.anxiety.classifier --csv data/anxiety_features_extended.csv \
#         --features 'rear|pct_periphery|pct_freeze|spatial_entropy|comfort' --tag rearonly

# 8. PC1-axis regression
python -m src.anxiety.regression

# 9. Tek subject inference (CLI)
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

## 18. Methods bölümü için hazır cümleler

### 18.1. Pipeline

> "Pose was estimated using a top-down 9-keypoint DeepLabCut model (PyTorch backend, RTX 3060). Rearing and grooming bouts were detected from filtered pose CSVs using a rule-based classifier (postural thresholds on inter-paw distance, paw-vertical offset, and body velocity; merge gap ≤ 15 frames, minimum bout duration ≥ 10 frames). OFT metrics (thigmotaxis, locomotion, freeze, spatial entropy) and spatial rearing classification (center vs wall, using arena 397–777 × 156–535 px and a literature-standard 20% margin inner zone, ≈36% of arena area) were computed downstream. A sensitivity analysis with a wider center definition (~7% margin, ~85% of arena classified as center) confirmed that the metric measures the thigmotaxis-ring versus inner-area distinction rather than wall contact per se."

### 18.2. Model

> "A logistic regression classifier (L2, C=1.0, `class_weight='balanced'`) was trained on 12 rearing-focused features (regex pattern `rear|pct_periphery|pct_freeze|spatial_entropy|comfort`, after dropping six highly correlated spatial-rearing center/wall variants identified as a source of LOOCV variance during iterative refinement) to discriminate Control (n=5) vs Treated (n=24, pooling Aspartame / Grapefruit / Aspartame+Grapefruit). Leave-one-out cross-validation was used owing to sample size constraints; no held-out test set was retained. Feature space and zone definition were iteratively refined through LOOCV; the final configuration corresponds to LR LOOCV AUC = 0.683 (balanced accuracy = 0.738; RF AUC = 0.525, SVM-RBF AUC = 0.358). The 'Treated' label is heterogeneous, pooling anxiogenic (aspartame) and anxiolytic (grapefruit) interventions; the model is therefore best interpreted as a **treatment-induced behavioral shift detector** rather than an anxiety classifier."

### 18.3. Inference

> "The final classifier is deployed via two equivalent interfaces: a command-line tool (`scripts/predict_anxiety_v2.py`) and a Streamlit web GUI (`scripts/predict_anxiety_v3_gui.py`). Both compute the 12-feature vector from a single DLC pose CSV, apply the saved median imputer + standard scaler + LR pickle, and report (a) predicted class with probability, (b) top-5 contributing features with their standardized z-values and coefficient-weighted contributions (`push = coef × z`), (c) an arena overlay PNG showing trajectory and per-bout rearing locations colored by center / wall classification."

### 18.4. Sınırlılıklar

> "Effect sizes for spatial rearing (Cohen's d = −0.92 for `rear_center_frac`, Treated vs Control) reach uncorrected significance (Mann-Whitney U p = 0.030) but do not survive Bonferroni correction across the seven spatial metrics tested (corrected p = 0.21). Power analysis indicates ≥20–22 animals per group are required to detect this effect size at 80% power; the present n = 29 study should therefore be interpreted as **pilot evidence**, not a definitive treatment effect."

### 18.5. Tezdeki tek-cümle mesaj

> "29 hayvan üzerinde DeepLabCut tabanlı uçtan uca yeniden üretilebilir bir davranış-analizi pipeline'ı geliştirildi; pipeline, rearing davranışının mekansal dağılımı (center vs wall) üzerinden Control vs Treated için LOOCV AUC=0.683 (balanced accuracy=0.738) elde etti ve uncorrected anlamlılıkta (`rear_center_frac` MW p=0.030, Cohen's d=−0.92) pilot kanıt sağladı; Bonferroni sonrası bu eşik geçilmemekle birlikte, etki büyüklüğü ve yön literatürdeki aspartam-anksiyojenik / flavonoid-anksiyolitik hipoteziyle uyumludur ve ≥20 hayvan/grup ile %80 güçte replikasyon için a-priori temel oluşturmaktadır."

---

## 19. Dosya haritası

```
scripts/
├── predict_anxiety_v2.py              ← CLI entrypoint + saf fonksiyonlar
├── predict_anxiety_v3_gui.py          ← Streamlit GUI (v2'yi import eder)
├── retrain_anxiety_rearonly.sh        ← config değişikliği sonrası tek-komut retrain
└── generate_test.py                   ← test data üretici

src/
├── behavior_detection.py              ← rule-based rearing/grooming detector
│                                        thresholds + frame→bout dönüşümü
└── anxiety/
    ├── config.py                      ← ARENA + INNER_ZONE + INNER_MARGIN (tek otorite)
    ├── spatial_rearing.py             ← bout center/wall sınıflayıcı + CSV üretici
    ├── profile.py                     ← feature matrix builder (eğitim için)
    ├── classifier.py                  ← LR/RF/SVM LOOCV training + pickle yazıcı
    ├── regression.py                  ← PC1 axis regression
    └── composite_index.py             ← AI_v1/v2/v3 (negatif bulgu)

analysis/open_field/
├── oft_metrics.py                     ← thigmotaxis · freeze · entropi · locomotion
├── run_analysis.py                    ← GUI'nin subprocess olarak çağırdığı script
│                                        (timeline · orbit · KDE · bodypart)
├── run_kare_batch.py                  ← tüm subject'ler için batch wrapper
└── behavior_analysis.py               ← downstream davranış analizi

models/anxiety_classifier/
├── lr_rearonly.pkl                    ← eğitilmiş LR (12-feature)
├── rf_rearonly.pkl, svm_rearonly.pkl
└── scaler_rearonly.pkl                ← {imputer, scaler, features} bundle

data/
├── DLCfiltered/<group>/<subject>/<subject>.csv   ← pipeline input
├── anxiety_features.csv                          ← ham 24-feature matrix
├── anxiety_features_extended.csv                 ← + 6 spatial sütun (classifier input'u)
└── spatial_rearing.csv                           ← bout-level center/wall CSV

reports/
├── anxiety_predictions_v2/<subject>_*            ← per-subject inference output
├── anxiety_classifier_metrics_rearonly.csv       ← model × metric tablosu
├── anxiety_classifier_predictions_rearonly.csv   ← fold başına tahmin
├── anxiety_classifier_importance_rearonly.csv   ← RF importance + LR coef
├── anxiety_stats.csv                             ← univariate KW
├── spatial_rearing_stats.csv                     ← univariate spatial istatistikler
├── composite_anxiety_*.csv                       ← AI_v1/v2/v3 (negatif bulgu)
└── figures/                                      ← CM, ROC, spatial PNG'leri

docs/
├── full_pipeline_overview.md          ← bu doküman (tam pipeline referansı)
├── anxiety_report.md                  ← konsolide bilimsel rapor (bulgular ağırlıklı)
├── REPO_MAP.md                        ← repo organization
├── DEEPLABCUT_PIPELINE.md             ← pose tracking tarafı (upstream)
├── OUTPUTS.md                         ← üretilen tüm CSV/PNG çıktılarının listesi
├── anxiety_findings_report.md         ← 2026-05-10 baseline (tarihsel kayıt)
└── anxiety_progress_2026-05-10.md     ← 2026-05-10 spatial rearing + inference (tarihsel)

archive/                               ← anxiety hot path'inde değil
├── scripts/predict_anxiety.py         ← v1 (v2 ile değişti)
├── src/{train_baseline,features,...}.py
└── models/classifier/                 ← eski 4-target baseline (25 .pkl)
```

---

## Appendix — Bir sıçan, kaç saniyede ne üretiliyor?

GUI progress adımları aşağı yukarı gerçek zamanı yansıtır (~178 sn'lik bir seans için):

| Adım | Tipik süre | Çıktı |
|---|---|---|
| 1. Davranış bout'ları | <1 sn | `bouts_df` (rearing/grooming) |
| 2. OFT metrikleri | <1 sn | 8 alan |
| 3. Spatial rearing | <0.5 sn | 7 alan + bout_zones |
| 4. Model yükleme + tahmin | <0.2 sn | pred + proba + top5 push |
| 5. Rapor + PNG | ~1 sn | TXT + JSON + arena overlay |
| (opsiyonel) `run_analysis.py` subprocess | 10–60 sn | timeline + KDE + bodypart PNG'ler |

CLI'de toplam ~3 sn (analiz adımı çalıştırılmaz). GUI'de tam analiz açıkken ~30–60 sn.

---

## Appendix — Hızlı referans

**Anxiety üzerinde çalışırken bakılacak yerler:**

```
src/anxiety/                              # tüm anxiety modülleri
src/behavior_detection.py                 # rule-based detector + thresholds
analysis/open_field/oft_metrics.py        # OFT metrik hesabı (anxiety'nin import ettiği)
scripts/predict_anxiety_v2.py             # CLI inference
scripts/predict_anxiety_v3_gui.py         # Streamlit GUI
scripts/retrain_anxiety_rearonly.sh       # config sonrası tek-komut retrain

models/anxiety_classifier/                # eğitilmiş modeller
reports/anxiety_predictions_v2/           # per-subject inference output
reports/figures/                          # CM, ROC, spatial rearing PNG'leri
data/anxiety_features.csv                 # ham 24-feature matrix
data/anxiety_features_extended.csv        # + 6 spatial sütun (classifier'ın input'u)

docs/full_pipeline_overview.md            # bu doküman
docs/anxiety_report.md                    # konsolide bilimsel rapor
docs/REPO_MAP.md                          # repo organization
```
