# Stage 03A — Open Field Test (Rectangle) Analysis
## Plan & Roadmap

---

## Objective

Dikdörtgen (Open Field) arena videolarından anksiyete, lokomotor ve keşif metriklerini çıkart. Stage 02B'den gelen **ethogram CSV'leri** ile rearing/grooming bout istatistiklerini OFT metriklerine entegre et. Dört grup karşılaştırması: **Kontrol, ASP, Greyfurt, ASP & Greyfurt**.

---

## Status

- [ ] Başlanmadı — Stage 02 (temiz DLC CSV) + Stage 02B (ethogram CSV) gerektirir

---

## Girdi Dosyaları

| Dosya | Kaynak | Açıklama |
|-------|--------|----------|
| `data/dlc_output/open_field/{grup}/{video}_clean.csv` | Stage 02 | DLC tracking CSV |
| `data/behavior/ethograms/{video}_ethogram.csv` | Stage 02B | Frame-level rearing/grooming etiketleri |
| `data/behavior/thresholds.json` | Stage 02B | Kalibrasyon eşikleri |
| `data/metadata.xlsx` | Stage 01 | Hayvan ID, grup, seans bilgisi |

---

## Grup → Hayvan Haritası (Open Field)

| Grup | Hayvan | Video'lar |
|------|--------|-----------|
| Kontrol | MA1 | MA1-1, MA1-2, MA1-3 |
| ASP | MA3 | MA3-1, MA3-2, MA3-3 |
| Greyfurt | MA5 | MA5-1, MA5-2, MA5-3 |
| ASP & Greyfurt | MA7 | MA7-1, MA7-2, MA7-3 |

---

## Arena Tanımı

```
┌─────────────────────────────┐
│  ·  ·  ·  ·  ·  ·  ·  ·  · │  ← Periferal zon (duvar hugging)
│  ·  ┌───────────────┐  ·  · │
│  ·  │               │  ·  · │
│  ·  │  Merkez Zon   │  ·  · │
│  ·  │               │  ·  · │
│  ·  └───────────────┘  ·  · │
│  ·  ·  ·  ·  ·  ·  ·  ·  · │
└─────────────────────────────┘
```

- **Merkez zon:** toplam alan'ın iç ~%25'i
- **Periferal zon:** duvarlardan ~%10 genişlikte dış çerçeve

---

## Görevler

### 3A.1 — Koordinat Sistemi Kalibrasyonu

- [ ] Referans kareden arena köşelerini (piksel) tanımla
- [ ] Piksel-to-cm ölçek faktörünü hesapla (bilinen boyuttan)
- [ ] Merkez zon poligonunu tanımla (alanın iç %25'i)
- [ ] Periferal zon poligonunu tanımla (dış %10 sınır)
- [ ] Arena konfigürasyonunu kaydet: `data/arena_config_rectangle.json`

```python
arena_config = {
    "arena_corners_px": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "px_per_cm": 4.2,
    "center_zone": [[cx1,cy1], [cx2,cy2], [cx3,cy3], [cx4,cy4]],
    "peripheral_width_px": 40
}
```

### 3A.2 — OFT Metrik Hesaplama (Lokomotor & Keşif)

`src/oft_analysis.py` içinde uygula.

- [ ] **Merkez süresi oranı** — seansın merkez zonda geçen oranı (anksiyete göstergesi)
- [ ] **Periferal süresi oranı** — duvara yakın kalma oranı (thigmotaxis indeksi)
- [ ] **Toplam mesafe (cm)** — body_center'ın kümülatif Öklid desplasman
- [ ] **Ortalama hız (cm/s)** — toplam mesafe / seans süresi
- [ ] **Keşif oranı** — dakika başına ziyaret edilen benzersiz uzaysal bin sayısı
- [ ] **İmobilite boutları** — hız < 2 cm/s olan ve > 2s süren periyotlar
- [ ] **İlk merkez girişi gecikmesi (s)** — seansın başından ilk merkez girişine kadar

```python
def compute_oft_metrics(tracking: dict, arena_config: dict, fps: int) -> dict:
    body = tracking["body_center"]
    x, y = body["x"], body["y"]

    dx = np.diff(x); dy = np.diff(y)
    total_distance_px = np.nansum(np.sqrt(dx**2 + dy**2))
    total_distance_cm = total_distance_px / arena_config["px_per_cm"]
    velocity = np.sqrt(dx**2 + dy**2) * fps / arena_config["px_per_cm"]  # cm/s

    # Zon üyeliği — Shapely ile
    from shapely.geometry import Point, Polygon
    center_poly = Polygon(arena_config["center_zone"])
    in_center = np.array([
        center_poly.contains(Point(xi, yi))
        for xi, yi in zip(x, y)
    ])
    center_time_ratio = np.nanmean(in_center)
    ...
```

### 3A.3 — Davranış Metrikleri (Stage 02B Entegrasyonu)

Ethogram CSV'sini yükle ve bout istatistiklerini hesapla.

- [ ] Seansın rearing'de geçen süre oranını hesapla
- [ ] Seansın grooming'de geçen süre oranını hesapla
- [ ] Rearing bout sayısını ve ortalama bout süresini hesapla
- [ ] Grooming bout sayısını ve ortalama bout süresini hesapla
- [ ] Rearing'in öncelikle hangi zonda gerçekleştiğini belirle (merkez vs periferal)

```python
import pandas as pd
import numpy as np
from scipy.ndimage import label

def compute_behavior_metrics(ethogram_path: str, fps: int = 25) -> dict:
    eth = pd.read_csv(ethogram_path)

    metrics = {}
    for behavior in ['rearing', 'grooming']:
        mask = eth[behavior].values
        n_frames = len(mask)

        labeled, n_bouts = label(mask)
        bout_lengths = [(labeled == i).sum() / fps for i in range(1, n_bouts + 1)]

        metrics[f'{behavior}_pct']            = mask.sum() / n_frames * 100
        metrics[f'{behavior}_bout_count']     = n_bouts
        metrics[f'{behavior}_mean_duration_s'] = np.mean(bout_lengths) if bout_lengths else 0
        metrics[f'{behavior}_total_duration_s']= sum(bout_lengths)

    # Locomotion = rearing değil + grooming değil + hız > eşik
    locomotion_mask = ~(eth['rearing'] | eth['grooming'])
    metrics['locomotion_pct'] = locomotion_mask.sum() / len(eth) * 100

    return metrics
```

#### Davranış × Zon Kesişimi

```python
def behavior_zone_breakdown(ethogram: pd.DataFrame, in_center: np.ndarray,
                             in_peripheral: np.ndarray) -> dict:
    """
    Rearing/grooming'in merkez vs periferal zonda ne kadar gerçekleştiğini döndürür.
    """
    result = {}
    for behavior in ['rearing', 'grooming']:
        mask = ethogram[behavior].values
        result[f'{behavior}_in_center_pct']     = (mask & in_center).sum() / mask.sum() * 100 if mask.sum() > 0 else 0
        result[f'{behavior}_in_peripheral_pct'] = (mask & in_peripheral).sum() / mask.sum() * 100 if mask.sum() > 0 else 0
    return result
```

### 3A.4 — Heatmap Üretimi (OFT)

Hem genel hem davranışa özel heatmap'ler üret.

- [ ] **Occupancy heatmap (tümü)** — body_center tüm frameler
- [ ] **Rearing heatmap** — yalnızca rearing frame'leri
- [ ] **Grooming heatmap** — yalnızca grooming frame'leri
- [ ] **Locomotion heatmap** — yalnızca aktif hareket frame'leri
- [ ] **Hız heatmap** — konuma göre ortalama hız
- [ ] **Renk kodlu trajectory** — davranış renkli rota çizgisi (Stage 02B format)

```
Renkler:
  Lokomotion → mavi   (#2196F3)
  Rearing    → kırmızı (#F44336)
  Grooming   → yeşil  (#4CAF50)
```

Her seans için çıktı:
```
data/figures/trajectories/open_field/{grup}/
    MA1-1_trajectory_all.png
    MA1-1_trajectory_rearing.png
    MA1-1_trajectory_grooming.png

data/figures/heatmaps/open_field/{grup}/
    MA1-1_heatmap_all.png      + .npy
    MA1-1_heatmap_rearing.png  + .npy
    MA1-1_heatmap_grooming.png + .npy
```

### 3A.5 — Grup Ortalamaları

- [ ] Her grup için OFT + davranış metriklerinin ortalama ± SD'sini hesapla
- [ ] Grup-ortalama KDE heatmap'ler üret (normalize edilmiş)
- [ ] 4 grup karşılaştırması: Kruskal-Wallis + Dunn post-hoc

```python
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

groups_data = {
    'Kontrol': df[df['group'] == 'Kontrol'][metric].values,
    'ASP': df[df['group'] == 'ASP'][metric].values,
    'Greyfurt': df[df['group'] == 'Greyfurt'][metric].values,
    'ASP_Greyfurt': df[df['group'] == 'ASP_Greyfurt'][metric].values,
}
stat, p = kruskal(*groups_data.values())
dunn = posthoc_dunn(list(groups_data.values()), p_adjust='bonferroni')
```

### 3A.6 — Dışa Aktarım

- [ ] `data/features/oft_metrics.csv` kaydet

#### Tam Sütun Listesi

| Sütun | Tür | Açıklama |
|-------|-----|----------|
| `rat_id` | str | MA1/MA3/MA5/MA7 |
| `session` | int | 1/2/3 |
| `group` | str | Kontrol/ASP/Greyfurt/ASP_Greyfurt |
| `center_time_ratio` | float | 0–1, anksiyete göstergesi |
| `peripheral_time_ratio` | float | 0–1, thigmotaxis |
| `total_distance_cm` | float | cm |
| `mean_velocity_cms` | float | cm/s |
| `exploration_rate` | float | bin/dk |
| `immobility_bouts` | int | adet |
| `first_center_latency_s` | float | s |
| `rearing_pct` | float | % |
| `rearing_bout_count` | int | adet |
| `rearing_mean_duration_s` | float | s |
| `rearing_total_duration_s` | float | s |
| `rearing_in_center_pct` | float | % |
| `rearing_in_peripheral_pct` | float | % |
| `grooming_pct` | float | % |
| `grooming_bout_count` | int | adet |
| `grooming_mean_duration_s` | float | s |
| `grooming_total_duration_s` | float | s |
| `grooming_in_center_pct` | float | % |
| `grooming_in_peripheral_pct` | float | % |
| `locomotion_pct` | float | % |

---

## Kabul Kriterleri

- Tüm 12 Open Field seans için OFT + davranış metrikleri hesaplanmış
- `oft_metrics.csv` eksik değer içermiyor
- Her seans için en az 1 genel + 1 rearing + 1 grooming heatmap üretilmiş
- 4 grup karşılaştırması tablosu (Kruskal-Wallis p-değerleri ile)

---

## Çıktı Dosyaları

| Yol | Açıklama |
|-----|----------|
| `data/arena_config_rectangle.json` | Zon tanımları |
| `data/features/oft_metrics.csv` | Seans başına OFT + davranış metrikleri |
| `data/figures/heatmaps/open_field/` | Genel + davranış heatmap'leri |
| `data/figures/trajectories/open_field/` | Davranış renkli rota çizimleri |
| `src/oft_analysis.py` | Analiz betiği |

---

## Sonraki Adım

→ **Stage 03B:** `STAGE_03B_TMAZE_HEATMAPS.md` (T-maze heatmap'leri)
→ **Stage 04:** `STAGE_04_PATH_ANALYSIS.md` (T-maze rota analizi)
