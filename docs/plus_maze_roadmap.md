# Plus Maze Analizi — Adım Adım Yol Haritası

**Tarih:** 2026-05-11  
**Kapsam:** `docs/plus_maze_documentation.md` §8.4'te listelenen eksik analizler

---

## Gerçek Durum (Kontrol Edilmiş)

Bazı adımlar belgede "yapılmadı" görünse de aslında tamamlanmış:

| # | Adım | Durum | Çıktı |
|---|------|-------|-------|
| 1 | KW + Dunn + PERMANOVA istatistik | **✅ Tamam** | `reports/cohort_epm_kw.csv` + `dunn.csv` + `permanova.csv` |
| 2 | Box/strip grafikleri | **✅ Tamam** | `reports/figures/epm_open_arm_by_cohort.png` + 2 grafik daha |
| 3 | OFT + Plus Maze feature birleştirmesi | **✅ Tamam** | `data/features/features_combined.csv` (46 özellik) |
| 4 | Cohen's d + Radar + bar chart | **✅ Tamam** | `reports/effect_analysis/` klasörü |
| 5 | ML (combined features) yeniden eğitimi | **⚠️ Belirsiz** | `scaler_combined.pkl` var ama model dosyaları combined mi? |
| 6 | Markov geçiş matrisi | **✅ Tamam** | `reports/markov_transition_matrices.csv` + `markov_heatmaps.png` + `markov_perseveration.png` |
| 7 | Etolojik feature'lar (grooming, SAP) | **✅ Tamam** | `reports/ethological_metrics_epm.csv` + 3 PNG — Rearing kapsam dışı (üstten kamera kısıtı) |

---

## Adım 5 — ML Yeniden Eğitimi (Combined Features)

### Neden gerekli?

Mevcut ML modelleri (`reports/model_comparison_all.csv`) yalnızca **OFT özelliklerini** kullanıyor. `features_combined.csv` içinde 16 adet `pm_*` Plus Maze özelliği var. Bunları eklersek:

- SHAP analizinde **hangi maze özelliğinin grup ayrımına katkı sağladığı** görünür.
- Özellikle `pm_successive_alternation_pct` ve `pm_perseveration_rate_pct` (çalışma belleği) OFT'de olmayan bilgi ekler.
- Tez için "OFT + Plus Maze birleşik pipeline" iddiasını destekler.

### Ne üretir?

| Çıktı | Açıklama |
|-------|----------|
| `reports/model_comparison_combined.csv` | 6 model × 4 hedef × LOOCV/LOGOCV F1 |
| `reports/figures/shap_combined_group.png` | Combined SHAP — `pm_*` özelliklerinin katkısı |
| `reports/figures/table_combined_comparison.png` | OFT-only vs combined F1 karşılaştırması |

### Nasıl çalışır?

`src/train_baseline.py` zaten LOOCV + LOGOCV + SHAP yapıyor. Sadece girdi dosyasını değiştirmek yeterli:

**Şu an:** `data/features/features_raw.csv` (26 özellik, OFT)  
**Olması gereken:** `data/features/features_combined.csv` (42 özellik, OFT+PM)

```bash
# Proje kökünden
python src/train_baseline.py --features data/features/features_combined.csv \
                              --out-prefix combined
```

> **Not:** `train_baseline.py` bir `--features` argümanı almıyorsa, scriptin içindeki `FEATURES_FILE` sabitini değiştir ya da kopyalayıp `train_combined.py` olarak oluştur.

### Dürüst beklenti

n=12 ile F1'in anlamlı iyileşmesi beklenmez. Amaç iyileşme değil, **SHAP'ta Plus Maze özelliklerinin görünmesi**.

---

## Adım 6 — Markov Geçiş Matrisi

### Neden gerekli?

`entry_sequence` sütunu ham kol-giriş dizisini tutuyor (örn. `B->T->L->R->B->B->T`). Bu seriyi sayısal geçiş olasılıklarına dönüştürmek:

- Hangi kohortun **hangi koldan hangi kola geçmeyi tercih ettiğini** gösterir.
- Persiverasyon (B→B) ve alternasyon (B→T→L→R) örüntülerini görsel hale getirir.
- Grup bazında davranış stratejileri karşılaştırılabilir.
- Cruz (1994) iki faktör modelini destekleyen bir **strateji farklılaşması** kanıtı olur.

### Ne üretir?

| Çıktı | Açıklama |
|-------|----------|
| `reports/markov_transition_matrices.csv` | 4 kohort × 4×4 geçiş olasılığı tablosu |
| `reports/figures/markov_heatmaps.png` | 4 adet 4×4 ısı haritası (kohort başına bir) |

### Nasıl çalışır?

**Yöntem:**
1. Her sıçan için `entry_sequence` dizisini `["B", "T", "L", "R"]` tokenlarına ayır.
2. Her ardışık çift `(a, b)` için geçiş sayısını bir 4×4 matrise ekle.
3. Satır-normalize et → satırlar olasılığa dönüşür (toplam = 1).
4. Kohort bazında (3 sıçan birleştirilerek) veya sıçan bazında hesapla.
5. Seaborn `heatmap()` ile görselleştir; diagonal = persiverasyon.

**Oluşturulacak script:** `analysis/plus_maze/markov_analysis.py`

```python
# Pseudokod — mantığı gösterir
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

ARMS  = ["B", "T", "L", "R"]   # bottom, top, left, right
df    = pd.read_csv("data/plus_maze_metrics_all.csv")

for cohort, grp in df.groupby("cohort"):
    mat = np.zeros((4, 4))
    for seq_str in grp["entry_sequence"].dropna():
        tokens = seq_str.split("->")
        for a, b in zip(tokens, tokens[1:]):
            if a in ARMS and b in ARMS:
                mat[ARMS.index(a), ARMS.index(b)] += 1
    # Satır normalize et
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, row_sums, where=row_sums > 0)
    # Görselleştir
    sns.heatmap(mat_norm, annot=True, xticklabels=ARMS, yticklabels=ARMS,
                cmap="Blues", vmin=0, vmax=1)
    plt.title(f"{cohort} — Geçiş Olasılıkları")
```

### Dürüst beklenti

n=3/grup ile geçiş sayıları çok düşük olacak. Bunu metodlar bölümünde sınırlılık olarak belirt. Yine de **görsel örüntü** (diagonal baskın = persiverasyon) tartışma için değerlidir.

---

## Adım 7 — Etolojik Feature'lar (Opsiyonel / İleri Seviye)

### Neden gerekli?

Mevcut Plus Maze metrikleri **yalnızca nerede** sorusunu cevaplar (zaman, giriş). EPM'nin altın standardı ek etolojik gözlemleri kapsar:

| Feature | Ne ölçer | Anksiyete yönü |
|---------|----------|---------------|
| **Korumasız head-dip** | Açık kol kenarından aşağı bakış | ↑ = anksiyolitik |
| **Stretch-attend posture (SAP)** | Riski değerlendirirken gerilen duruş | ↓ = anksiyolitik |
| **Risk assessment** | Açık kola yarım girip geri dönme | ↓ = anksiyolitik |

Bu feature'lar olmadan **saf anksiyolitik vs motor etki** ayrımı yapılamaz (Cruz 1994).

### Önkoşul

DLC modelinin 10 keypointi var (nose, head, body_center, left/right_ear, forepaws, hindpaws, tail_base). Bu noktalara erişim var — sadece hesaplama scripti yazılmamış.

### Nasıl çalışır?

**Head-dip tespiti:**
```python
# Açık kol (sol veya sağ) içindeyken nose'un arena tabanına yaklaşması
# body_center açık koldaki junction yakınındayken nose_y > eşik → head-dip
is_open_arm = zone_label in ("left_arm", "right_arm")
near_edge   = nose_y > (junction_y + threshold_px)
head_dip    = is_open_arm and near_edge
```

**SAP tespiti:**
```python
# Vücut uzunluğu = nose - tail_base mesafesi
# Hız düşük + vücut uzunluğu artışı → stretch-attend
body_len = dist(nose, tail_base)
sap = (body_len > mean_body_len * 1.2) and (speed < low_speed_thresh)
```

**Oluşturulacak script:** `analysis/plus_maze/ethological_features.py`

### Dürüst beklenti

Bu adım tez için **zorunlu değil**, ama sunulursa güçlü bir metodolojik katkıdır. EPM literatüründe head-dip ve SAP bulunmayan çalışmalar yine de kabul görmekte — özellikle otomatik/DLC tabanlı çalışmalarda (Sturman 2020).

**Öneri:** Tez takvimi sıkışıksa bu adımı "gelecek çalışma" olarak sınırlılıklar bölümüne ekle.

---

## Önerilen Uygulama Sırası

```
1. Adım 5 — ML (combined) yeniden eğitimi       │ ~1 saat │ Tez için değerli
2. Adım 6 — Markov geçiş matrisi (yeni script)  │ ~2 saat │ Tez için değerli
3. Adım 7 — Etolojik feature'lar                │ ~1 gün  │ Opsiyonel / güçlü
```

---

## Tez Takvimi Kılavuzu

| Tezde bölüm | Hangi adımlar gerekli |
|-------------|----------------------|
| **Yöntemler** | Adım 5 (combined pipeline kurgusu) |
| **Bulgular** | Adım 5 (SHAP görsel), Adım 6 (Markov ısı haritaları) |
| **Tartışma** | Adım 6 (strateji karşılaştırması), Adım 7 (gelecek çalışma olarak) |

---

## Dosya Haritası (Mevcut ve Hedef)

```
Thesis/RatBehaviorAnalyze/
│
├── data/features/
│   ├── features_raw.csv            ✅ OFT (26 özellik)
│   ├── features_combined.csv       ✅ OFT + PM (46 özellik, pm_* kolonları var)
│   └── features_combined_normalized.csv  ✅
│
├── reports/
│   ├── cohort_epm_kw.csv           ✅ KW istatistik sonuçları
│   ├── cohort_epm_dunn.csv         ✅ Dunn post-hoc
│   ├── cohort_epm_permanova.csv    ✅ PERMANOVA
│   ├── model_comparison_all.csv    ✅ OFT modelleri
│   ├── model_comparison_combined.csv  ✅ Tamam — Adım 5
│   └── markov_transition_matrices.csv ✅ Tamam — Adım 6
│
├── reports/figures/
│   ├── epm_open_arm_by_cohort.png  ✅ Box+strip grafikleri
│   ├── epm_arm_distribution.png    ✅ Stacked bar kol dağılımı
│   ├── epm_locomotor_covariate.png ✅ Scatter lokomotor kovaryat
│   ├── shap_combined_group.png     ✅ Tamam — Adım 5
│   └── markov_heatmaps.png         ✅ Tamam — Adım 6
│
├── reports/effect_analysis/
│   ├── behavioral_effect_table.csv ✅ Cohen's d tablosu
│   ├── behavioral_radar.png        ✅ Radar grafiği
│   └── behavioral_effect_bars.png  ✅ Bar chart
│
├── analysis/plus_maze/
│   ├── cohort_stats_epm.py         ✅ KW + Dunn + PERMANOVA
│   ├── plot_open_arm.py            ✅ Box/strip + stacked bar + scatter
│   ├── behavioral_effect_analysis.py  ✅ Cohen's d + radar + bar
│   ├── markov_analysis.py          ✅ Tamam — Adım 6
│   └── ethological_features.py     ✅ Tamam — Adım 7 (grooming + SAP)
│
└── src/
    ├── features.py                 ✅ OFT features (Plus Maze ekleme scripti ayrı)
    ├── train_baseline.py           ✅ OFT-only modeller
    └── train_combined.py           ✅ Tamam — Adım 5
```

---

## Sık Sorulan Sorular

**S: Adım 1-4 gerçekten bitti mi?**  
E: Evet — `reports/` klasöründe CSV ve PNG çıktıları doğrulandı (2026-05-11).

**S: features_combined.csv hangi `pm_` kolonlarına sahip?**  
A: 16 adet: `pm_pct_open_arm`, `pm_pct_closed_arm`, `pm_pct_open_arm_entries`, `pm_pct_closed_arm_entries`, `pm_anxiety_index_epm`, `pm_total_entries`, `pm_mean_speed_px_s`, `pm_total_distance_px`, `pm_arm_preference_index`, `pm_successive_alternation_pct`, `pm_perseveration_rate_pct`, `pm_pct_time_left`, `pm_pct_time_right`, `pm_pct_time_top`, `pm_pct_time_bottom`, `pm_pct_time_junction`.

**S: n=12 ile Markov matrisi güvenilir mi?**  
A: İstatistiksel olarak değil, ama görsel örüntü (örn. diagonal baskınlık = persiverasyon tercihi) biyolojik anlamlıdır. Metodlar bölümünde açıkla.

**S: ML F1 combined features ile iyileşir mi?**  
A: Muhtemelen az veya hiç iyileşmez (n=12). Ama SHAP görüntüsü `pm_*` özelliklerini gösterir — bu tez için yeterli.
