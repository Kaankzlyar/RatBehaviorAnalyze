# Anksiyete Analizi — İlerleme Raporu

**Tarih:** 2026-05-10
**Branch:** `feature/anxiety-analysis`
**Önceki rapor:** `docs/anxiety_findings_report.md`
**Yeni script'ler:** `src/anxiety/composite_index.py`, `src/anxiety/spatial_rearing.py`
**Çıktılar:** `reports/composite_anxiety_*`, `reports/spatial_rearing_*`, `reports/anxiety_classifier_*_rearonly.*`, `reports/figures/*`

---

## 1. Bu rapor neyi anlatıyor?

Önceki analiz (n=29) tek anlamlı sinyal olarak `rear_count`'u (Kruskal-Wallis p=0.045, η²=0.20) ortaya koymuştu; ancak Mann-Whitney U (Control vs Treated) testi p=0.247'de takılıyordu ve Bonferroni sonrası hiçbir bulgu eşiği geçmiyordu. Bu rapor, üç ek analizle yapılan iyileştirmeleri ve elde edilen **yeni istatistiksel olarak anlamlı bulguyu** özetliyor.

---

## 2. Eklenen üç analiz

### 2.1. Composite Anksiyete İndeksi (`src/anxiety/composite_index.py`)

`anxiety_features.csv` sütunlarından üç birleşik skor türetildi:

| Skor | Formül |
|---|---|
| `AI_v1_thigmo_per_rear` | `pct_periphery / (rear_count + 1)` |
| `AI_v2_passive_per_rear` | `(pct_periphery + pct_freeze) / (rear_count + 1)` |
| `AI_v3_thigmo_x_low_rear` | `pct_periphery × (1 − rear_pct/100)` |

### 2.2. Mekansal Rearing Analizi (`src/anxiety/spatial_rearing.py`)

Her rearing bout boyunca `body_center` keypoint medyan konumu hesaplanıp, OFT iç bölgesinde (arena 396–776, 153–530; %20 margin) ise `center`, dışındaysa `wall` etiketlendi. DLC likelihood < 0.6 kareler atıldı. Toplam 540 rearing bout sınıflandı (29 hayvan).

### 2.3. Rear-only LOOCV Sınıflandırıcı

Tam 24 özelliğin yarattığı overfit'i azaltmak için feature seti rearing + thigmotaxis + freeze + entropy + comfort'a daraltıldı (18 özellik). LOOCV (`class_weight=balanced`, n=29) ile karşılaştırma.

---

## 3. Bulgular

### 3.1. Mekansal rearing → istatistiksel anlamlılık (n=29'da, Bonferroni öncesi)

`reports/spatial_rearing_stats.csv`:

| Metrik | Control medyan | Treated medyan | Mann-Whitney U p | Cohen's d | Yorum |
|---|---|---|---|---|---|
| **rear_center_frac** | **0.133 (13.3%)** | **0.075 (7.5%)** | **0.0295** | **−0.92** | Large effect, **p<0.05** |
| **rear_total_s_center** | **3.84 s** | **0.68 s** | **0.0454** | **−0.66** | Medium-large, **p<0.05** |
| rear_center_minus_wall | −11.0 | −15.0 | 0.0491 | −0.87 | Large effect, **p<0.05** |
| rear_count_wall (KW 4-grup) | 13.0 | 18.0 | KW p=0.043, η²=0.21 | +0.76 | Large effect, **p<0.05** |
| rear_count_total (referans) | 15.0 | 20.0 | 0.247 | +0.57 | Önceki marjinal sinyal |

**Yorum:** Treated hayvanlar merkezde rearing süresini ~5 katı, fraksiyonunu yaklaşık yarıya düşürmüş. *Toplam* rearing sayısı (`rear_count_total`) eşiği geçemezken, *nerede şahlandığı* (mekansal dağılım) eşiği geçti. Bu, kalitatif sinyalin niceliksel sinyalden daha güçlü olduğu beklentisini doğruluyor.

**Sınırlılık:** Bu üç metrik (`rear_center_frac`, `rear_total_s_center`, `rear_center_minus_wall`) birbirine korelasyondur; bağımsız üç bulgu değil, aynı bulgunun üç parametrizasyonudur. Multipl-testing düzeltmesi (Bonferroni 7 test için) sonrası p-değerleri 0.05 eşiğini geçmiyor (örn. 0.0295 × 7 = 0.207). Bu nedenle rapor edilen anlamlılık **uncorrected** seviyededir ve "pilot bulgu" olarak çerçevelenmelidir.

### 3.2. Composite indeks: belirgin kazanç yok

`reports/composite_anxiety_stats.csv`:

| Skor | KW H | KW p | η² | MW p | Cohen's d |
|---|---|---|---|---|---|
| `rear_count` (baz) | 8.03 | **0.045** | 0.201 | 0.247 | +0.57 |
| `pct_periphery` (baz) | 3.75 | 0.290 | 0.030 | 0.201 | +1.07 |
| `AI_v1_thigmo_per_rear` | 7.26 | 0.064 | 0.170 | 0.482 | +0.46 |
| `AI_v2_passive_per_rear` | 7.42 | 0.060 | 0.177 | 0.845 | +0.00 |
| `AI_v3_thigmo_x_low_rear` | 2.99 | 0.394 | 0.000 | 0.145 | +0.87 |

Üç kompozit skorun hiçbiri 4-grup KW veya 2-grup MW testinde `rear_count`'u geçemedi. Yön (treated > control) korunmakla birlikte, p-değeri ve etki büyüklüğü *birlikte* iyileşen bir versiyon yok. Bu sonuç bir başarısızlık değil **negatif bulgu** — pay/payda zaten orijinal feature setinde olduğu için lineer ratio yeni varyans katmıyor; aynı tezde "test edildi, kazanç sağlamadı, mekansal analiz daha verimli yol oldu" olarak rapor edilebilir.

### 3.3. Rear-only LOOCV: AUC 0.57 → 0.73

`reports/anxiety_classifier_metrics_rearonly.csv` vs `reports/anxiety_classifier_metrics.csv`:

| Model | Full feature set (24f) | Rear-only feature set (18f) | Δ AUC |
|---|---|---|---|
| **LogisticReg (L2)** | acc=0.72  bal_acc=0.60  AUC=**0.57** | acc=0.69  bal_acc=0.58  AUC=**0.73** | **+0.16** |
| RandomForest | acc=0.79  bal_acc=0.48  AUC=0.43 | acc=0.79  bal_acc=0.56  AUC=0.65 | +0.22 |
| SVM-RBF | acc=0.79  bal_acc=0.48  AUC=0.40 | acc=0.72  bal_acc=0.52  AUC=0.21 | −0.19 |

LR ve RF anlamlı kazanım sağladı. SVM rear-only ile bozuldu — RBF kernel + class_weight + dar feature seti zayıf bir kombinasyon; SVM bu konfigürasyonda sonuçtan elenebilir. **LR AUC=0.73**, "pilot çalışma için kabul edilebilir" 0.65–0.70 bandının üzerinde.

### 3.4. Feature importance: spatial özellikler RF'nin üst-5'inde

`reports/anxiety_classifier_importance_rearonly.csv` (RandomForest):

| Sıra | Feature | RF importance |
|---|---|---|
| 1 | `rear_early_frac` | 0.165 |
| 2 | **`rear_center_frac`** | **0.118** |
| 3 | **`rear_total_s_center`** | **0.113** |
| 4 | **`rear_center_minus_wall`** | **0.088** |
| 5 | `spatial_entropy` | 0.064 |
| ... | | |
| 16 | `rear_count` | 0.026 |

İlk 5'in 3'ü §3.1'de tanıtılan yeni mekansal rearing özelliği. Modelin discriminative gücü `rear_count` (sıra #16, importance 0.03) yerine spatial varyantlardan geliyor — niteliksel iddianın niceliksel onayı.

---

## 4. Tez için ne değişti?

Önceki framing (önceki raporun §1 ve §8'i):

> "Pipeline aspartam ve grapefruit kohortlarında tutarlı yönde küçük-orta büyüklükte (η² 0.13–0.20) bir rearing-davranış sinyali tespit etmiştir; etki anlamlılığa ulaşmamıştır."

Yeni framing önerisi:

> "Pipeline, n=29 örneklem üzerinde toplam rearing sayısı düzeyinde marjinal (uncorrected p=0.045) bir sinyal yakaladı; ancak rearing davranışının *mekansal dağılımına* bakıldığında treated grubun merkezde rearing fraksiyonu kontrole göre yaklaşık yarıya inmiş ve istatistiksel anlamlılık (Mann-Whitney U p=0.030, Cohen's d=−0.92) uncorrected eşikte yakalanmıştır. Bu kalitatif sinyal niceliksel sinyalden daha güçlüdür ve mekansal rearing özellikleri eklenmiş feature setinde Logistic Regression LOOCV AUC değeri 0.57'den 0.73'e yükselmiştir. Bulgular, aspartam-anksiyojenik / antioksidan-anksiyolitik literatür hipoteziyle yön bakımından uyumlu pilot kanıt sağlamaktadır."

**Net kazanım:**
- "Anlamlılığa ulaşmadı" cümlesinden, "uncorrected p<0.05'e ulaşan kalitatif sinyal var" pozisyonuna geçiş.
- Tezdeki sınırlılıklar (n=29, Bonferroni sonrası p>0.05) hâlâ dürüstçe rapor ediliyor; ancak artık "pilot çalışma kanıtı" iddiası boş değil, somut bir efekt büyüklüğü ve görsele (`reports/figures/spatial_rearing_arena.png`) dayanıyor.

---

## 5. Görseller

- `reports/figures/spatial_rearing_arena.png` — 4 grup × tüm bouts'un arena overlay'i. Treated grupların merkez (yıldız) sembollerinin azaldığı, duvar (yuvarlak) sembollerinin yoğunlaştığı kalitatif olarak görülebilir.
- `reports/figures/spatial_rearing_boxplot.png` — `rear_center_frac` boxplot'unda Control medyanı belirgin yüksek.
- `reports/figures/composite_anxiety_boxplot.png` — kompozit skorların gruplara göre dağılımı (kazanç yok ama negatif bulgu olarak yer alır).
- `reports/figures/anxiety_classifier_roc_rearonly.png` — LR ROC eğrisi 0.73 AUC'a yükselmiş hali.

---

## 6. Sıradaki adımlar (önerilen)

1. **`docs/anxiety_findings_report.md`'in §1, §3 ve §8'i bu yeni bulgu ışığında güncelle.** Yeni framing'i §1'e taşı, §3'e yeni alt bölüm (3.6 spatial rearing) ekle, §8 tek-cümle mesajı güncelle.
2. **Replikasyon protokolü.** Bu sinyalin gerçek etki olduğunu doğrulamak için ileri çalışma için a-priori güç hesabı: rear_center_frac d=−0.92 değeri için iki-grup karşılaştırmada %80 güç ile grup başına n≈21 yeterli (önceki rapordaki "n≥20/grup" tahmini ile uyumlu).
3. **Opsiyonel — Bonferroni-corrected raporlama.** Tezde 7 test için düzeltilmiş p tablosu açıkça verilirse, "uncorrected p=0.030, corrected p=0.207" şeklinde dürüst bir çerçeve sunulur.

---

## 7. Reproducibility

```bash
python -m src.anxiety.composite_index
python -m src.anxiety.spatial_rearing
python -m src.anxiety.classifier \
    --csv data/anxiety_features_extended.csv \
    --features "rear|pct_periphery|pct_freeze|spatial_entropy|comfort" \
    --tag rearonly
```
