# Stage 05 — Feature Engineering
## Plan & Roadmap

---

## Objective

OFT (Stage 03A), T-maze (Stage 04) ve davranış sınıflandırma (Stage 02B) metriklerini birleştirerek birleşik özellik veri seti oluştur. Türetilmiş özellikler üret, seans ve hayvan bazında topla, ML model eğitimi için nihai tabloyu hazırla. Gruplar: **Kontrol, ASP, Greyfurt, ASP & Greyfurt**.

---

## Status

- [ ] Başlanmadı — Stage 02B, 03A, 03B, 04 çıktıları gerektirir

---

## Girdi Kaynakları

| Dosya | Kaynak Stage | Açıklama |
|-------|-------------|----------|
| `data/features/oft_metrics.csv` | Stage 03A | Seans başına OFT + davranış metrikleri |
| `data/features/tmaze_metrics.csv` | Stage 04 | Deneme başına T-maze metrikleri |
| `data/features/tmaze_zone_metrics.csv` | Stage 03B | Seans başına zon × davranış metrikleri |
| `data/behavior/group_stats.xlsx` | Stage 02B | Grup bazında bout istatistikleri |
| `data/metadata.xlsx` | Stage 01 | Hayvan ID, grup, seans bilgisi |

---

## Görevler

### 5.1 — T-Maze Denemelerinden Seans Düzeyi Toplama

T-maze metrikleri deneme başına; OFT ile birleştirmek için seans başına topla.

- [ ] Her deneme metriği için seans başına ortalama ve SD hesapla
- [ ] Doğru kol seçim oranı (eğer doğru kol tanımlanmışsa)
- [ ] Denemeler içi öğrenme eğimi (turn_bias değişimi)

```python
import pandas as pd

tmaze = pd.read_csv("data/features/tmaze_metrics.csv")

agg = tmaze.groupby(["rat_id", "session"]).agg(
    turn_bias                  = ("turn_choice", lambda x: (x == "sol").mean()),
    path_efficiency_mean       = ("path_efficiency", "mean"),
    path_efficiency_std        = ("path_efficiency", "std"),
    decision_latency_mean      = ("decision_latency_s", "mean"),
    velocity_govde_mean        = ("velocity_govde", "mean"),
    backtrack_rate             = ("backtrack_rate", "first"),
    heading_var_mean           = ("heading_angle_var", "mean"),
    # Davranış metrikleri (Stage 02B entegrasyonu)
    rearing_pct_mean           = ("rearing_pct_of_trial", "mean"),
    rearing_in_secim_pct_mean  = ("rearing_in_secim_pct", "mean"),
    grooming_pct_mean          = ("grooming_pct_of_trial", "mean"),
    rearing_before_decision_mean = ("rearing_before_decision_s", "mean"),
    trial_count                = ("turn_choice", "count"),
).reset_index()
```

### 5.2 — OFT + T-Maze + Zon Metriklerini Birleştir

- [ ] `rat_id` + `session` üzerinden sol-birleştir
- [ ] Hiçbir seans kaybolmadığını doğrula (4 hayvan × 3 seans = 12 satır)
- [ ] Metadata'dan `group` sütununu ekle

```python
oft       = pd.read_csv("data/features/oft_metrics.csv")
zone      = pd.read_csv("data/features/tmaze_zone_metrics.csv")
meta      = pd.read_excel("data/metadata.xlsx")[["rat_id", "session", "group"]].drop_duplicates()

features = (
    oft
    .merge(agg,  on=["rat_id", "session"], how="outer")
    .merge(zone, on=["rat_id", "session"], how="left")
    .merge(meta, on=["rat_id", "session"], how="left")
)

assert len(features) == 12, f"Beklenen 12 satır, bulunan {len(features)}"
```

### 5.3 — Çapraz Arena Türetilmiş Özellikler

OFT ile T-maze sinyallerini birleştiren özellikler türet.

| Özellik | Formül | Biyolojik Gerekçe |
|---------|--------|-------------------|
| `anksiyete_kompozit` | `peripheral_time_ratio × decision_latency_mean` | OFT kaçınması + T-maze kararsızlığı |
| `lokomotor_delta` | `oft_total_distance - tmaze_path_efficiency_mean × 100` | Arenalar arası aktivite kayması |
| `kesifinml_verimlilik` | `oft_exploration_rate / path_efficiency_mean` | Keşif stratejisi |
| `hiz_tutarlilik` | `oft_mean_velocity / velocity_govde_mean` | Arenalar arası motor tutarlılığı |
| `davranis_kaygisi` | `(rearing_in_secim_pct_mean + peripheral_time_ratio) / 2` | Karar noktası + periferal kombinasyonu |
| `rearing_context_ratio` | `rearing_in_secim_pct_mean / rearing_pct (OFT)` | Rearing'in ne kadarı stres bağlamında? |

```python
features['anksiyete_kompozit']    = features['peripheral_time_ratio'] * features['decision_latency_mean']
features['lokomotor_delta']       = features['total_distance_cm'] - features['path_efficiency_mean'] * 100
features['kesifinml_verimlilik']  = features['exploration_rate'] / (features['path_efficiency_mean'] + 1e-9)
features['hiz_tutarlilik']        = features['mean_velocity_cms'] / (features['velocity_govde_mean'] + 1e-9)
features['davranis_kaygisi']      = (features['rearing_in_secim_pct_mean'] + features['peripheral_time_ratio']) / 2
features['rearing_context_ratio'] = features['rearing_in_secim_pct_mean'] / (features['rearing_pct'] + 1e-9)
```

### 5.4 — Seans-İçi (Rolling) Özellikler

- [ ] Her metrik için 3 seans üzerinden kayan ortalama
- [ ] Seans-seans delta (seans2 − seans1, seans3 − seans2)
- [ ] T-maze metrikleri için seans-içi varyabilite (deneme başına SD)

```python
features_sorted = features.sort_values(["rat_id", "session"])

for col in metric_cols:
    features_sorted[f'{col}_rolling3'] = (
        features_sorted.groupby("rat_id")[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    )
    features_sorted[f'{col}_delta'] = (
        features_sorted.groupby("rat_id")[col].diff()
    )
```

### 5.5 — Eksik Değer İşleme

- [ ] Sütun başına % eksik değeri raporla
- [ ] T-maze seans için geçerli deneme yoksa: toplanmış metrikler NaN
- [ ] Kalan NaN'ları sütun medyanı ile doldur (tüm imputasyonları belgele)
- [ ] % 30'dan fazla eksik değer olan sütunları düşür

### 5.6 — Kodlama & Normalizasyon

- [ ] `rat_id`'yi tamsayı faktörü olarak kodla
- [ ] `group`'u one-hot encode et (Kontrol, ASP, Greyfurt, ASP_Greyfurt)
- [ ] Tüm sürekli özellikler üzerinde StandardScaler uygula
- [ ] Ölçekleyiciyi kaydet: `models/classifier/scaler.pkl`

### 5.7 — Etiket Atama

4-sınıflı grup etiketi birincil hedef; ikincil davranışsal etiketler aşağıdaki gibi:

| Etiket | Tür | Tanım |
|--------|-----|-------|
| `group` | 4 sınıf | Kontrol / ASP / Greyfurt / ASP_Greyfurt |
| `anksiyete_seviyesi` | ikili | Yüksek vs Düşük (peripheral_time + decision_latency eşiği) |
| `uzamsal_hafiza` | sıralı | T-maze'de doğru kol seçim oranı |
| `stres_yaniti` | ikili | Kontrolden davranışsal sapma |
| `rearing_profili` | ikili | Yüksek rearing (> grup medyanı) |
| `grooming_profili` | ikili | Yüksek grooming (> grup medyanı) |

### 5.8 — Dışa Aktarım

- [ ] `data/features/features_raw.csv` — normalize edilmemiş birleşik özellikler
- [ ] `data/features/features_normalized.csv` — StandardScale edilmiş
- [ ] `data/features/labels.csv` — hedef etiket sütunları

---

## Nihai Özellik Veri Seti Yapısı

```
# Kimlik
rat_id | session | group |

# OFT Metrikleri (Stage 03A)
oft_center_time | oft_peripheral_time | oft_total_distance |
oft_mean_velocity | oft_exploration_rate | oft_immobility_bouts |
oft_first_center_latency |

# OFT Davranış Metrikleri (Stage 02B → 03A)
oft_rearing_pct | oft_rearing_bout_count | oft_rearing_total_s |
oft_rearing_in_center_pct | oft_rearing_in_peripheral_pct |
oft_grooming_pct | oft_grooming_bout_count | oft_grooming_total_s |
oft_locomotion_pct |

# T-Maze (Stage 04 — toplanmış)
tmaze_turn_bias | tmaze_path_efficiency_mean | tmaze_decision_latency_mean |
tmaze_velocity_govde | tmaze_backtrack_rate | tmaze_heading_var |

# T-Maze Davranış (Stage 02B → 04)
tmaze_rearing_pct | tmaze_rearing_in_secim_pct |
tmaze_grooming_pct | tmaze_rearing_before_decision |

# T-Maze Zon (Stage 03B)
secim_dwell_s | secim_rearing_s | secim_grooming_s | sol_bias |

# Çapraz Arena Türetilmiş
anksiyete_kompozit | lokomotor_delta | kesifinml_verimlilik |
hiz_tutarlilik | davranis_kaygisi | rearing_context_ratio |

# Seans Deltaları
oft_distance_delta | tmaze_latency_delta | rearing_pct_delta |

# Etiketler
group | anksiyete_seviyesi | uzamsal_hafiza | rearing_profili | grooming_profili
```

---

## Kabul Kriterleri

- 12 satırlı birleşik özellik veri seti (4 hayvan × 3 seans), seans kaybı yok
- Tüm özellikler birim ve türetim yöntemiyle belgelenmiş
- OFT ↔ T-maze ilişkilerini gösteren korelasyon matrisi üretilmiş
- En az `group` ve `anksiyete_seviyesi` etiket sütunları atanmış

---

## Çıktı Dosyaları

| Yol | Açıklama |
|-----|----------|
| `data/features/features_raw.csv` | Birleştirilmiş, normalize edilmemiş özellikler |
| `data/features/features_normalized.csv` | StandardScale edilmiş özellikler |
| `data/features/labels.csv` | Hedef etiketler |
| `models/classifier/scaler.pkl` | Kaydedilmiş StandardScaler |
| `src/features.py` | Özellik mühendisliği betiği |

---

## Sonraki Adım

→ **Stage 06:** `STAGE_06_MODEL_TRAINING.md`
