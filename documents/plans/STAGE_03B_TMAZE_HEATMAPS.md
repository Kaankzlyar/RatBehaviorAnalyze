# Stage 03B — T-Maze Heatmap Üretimi
## Plan & Roadmap

---

## Objective

T-maze tracking verilerinden (Part1) uzaysal heatmap'ler ve zon occupancy zaman serileri üret. Stage 02B ethogram verilerini kullanarak rearing/grooming'in T-maze içindeki **zona özgü** dağılımını ortaya koy. Dört grup: **Kontrol, ASP, Greyfurt, ASP & Greyfurt**.

---

## Status

- [ ] Başlanmadı — Stage 02 (temiz DLC CSV) + Stage 02B (ethogram CSV) gerektirir

---

## Girdi Dosyaları

| Dosya | Kaynak | Açıklama |
|-------|--------|----------|
| `data/dlc_output/tmaze/{grup}/{video}_clean.csv` | Stage 02 | DLC tracking CSV |
| `data/behavior/ethograms/{video}_ethogram.csv` | Stage 02B | Frame-level rearing/grooming etiketleri |
| `data/metadata.xlsx` | Stage 01 | Hayvan ID, grup, seans bilgisi |

---

## Grup → Hayvan Haritası (T-Maze)

| Grup | Hayvan | Video'lar |
|------|--------|-----------|
| Kontrol | MA1 | Part1-MA1-1, Part1-MA1-2, Part1-MA1-3 |
| ASP | MA3 | Part1-MA3-1, Part1-MA3-2, Part1-MA3-3 |
| Greyfurt | MA5 | Part1-MA5-1, Part1-MA5-2, Part1-MA5-3 |
| ASP & Greyfurt | MA7 | Part1-MA7-1, Part1-MA7-2, Part1-MA7-3 |

---

## T-Maze Zon Düzeni

```
         ┌──────┐  ┌──────┐
         │  SG  │  │  SG  │    SG = sol_goal / sag_goal
         └──┬───┘  └──┬───┘
            │  Seçim  │
            │  Noktası│
            └────┬────┘
                 │ Gövde
                 │
            ┌────┴────┐
            │  Giriş  │
            └─────────┘
```

### Adlandırılmış Zonlar

| Zon ID | İsim | Açıklama |
|--------|------|----------|
| `giris` | Giriş zonu | Başlangıç kutusu / gövdenin tabanı |
| `govde` | Gövde | Girişten seçim noktasına koridor |
| `secim` | Seçim noktası | Gövde ile kollar arasındaki kavşak |
| `sol_kol` | Sol kol | Sol koridor |
| `sag_kol` | Sağ kol | Sağ koridor |
| `sol_goal` | Sol goal | Sol kolun ucu |
| `sag_goal` | Sağ goal | Sağ kolun ucu |

---

## Görevler

### 3B.1 — Koordinat Sistemi Kalibrasyonu

- [ ] T-maze videosundan referans kare çıkar
- [ ] Zon poligon köşelerini piksel koordinatlarında manuel tanımla
- [ ] Bilinen labirent boyutlarından px-to-cm ölçek faktörü hesapla
- [ ] Kaydet: `data/arena_config_tmaze.json`

```python
arena_config = {
    "px_per_cm": 4.8,
    "zones": {
        "giris":    [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "govde":    [[...], ...],
        "secim":    [[...], ...],
        "sol_kol":  [[...], ...],
        "sag_kol":  [[...], ...],
        "sol_goal": [[...], ...],
        "sag_goal": [[...], ...],
    }
}
```

### 3B.2 — Genel Heatmap Türleri

`src/heatmap.py` içinde uygula.

| Heatmap | Uygulama | Çıktı |
|---------|----------|-------|
| Occupancy (tümü) | body_center 2D histogramı | `occupancy_{hayvan}_{seans}.png` |
| Hız | Uzaysal bin başına ortalama hız | `velocity_{hayvan}_{seans}.png` |
| Burun noktası | nose 2D histogramı — araştırma haritası | `nose_{hayvan}_{seans}.png` |
| Giriş sıklığı | Zon geçiş sayıları labirent üzerine | `entry_freq_{hayvan}_{seans}.png` |
| Yol yoğunluğu | Tüm trajectory çizgilerinin üst üste bindirilmesi | `paths_{hayvan}_{seans}.png` |

### 3B.3 — Davranışa Özel Heatmap'ler (Stage 02B Entegrasyonu)

Ethogram CSV'sinden davranış maskelerini yükle ve ayrı heatmap'ler üret.

- [ ] **Rearing heatmap** — yalnızca rearing frame'lerindeki pozisyonlar
- [ ] **Grooming heatmap** — yalnızca grooming frame'lerindeki pozisyonlar
- [ ] **Locomotion heatmap** — yalnızca aktif hareket frame'leri

```python
import pandas as pd
import numpy as np

def load_behavior_masks(ethogram_path: str) -> dict:
    eth = pd.read_csv(ethogram_path)
    return {
        'rearing':   eth['rearing'].values.astype(bool),
        'grooming':  eth['grooming'].values.astype(bool),
        'locomotion': (~eth['rearing'] & ~eth['grooming']).values,
    }

def generate_behavior_heatmaps(x: np.ndarray, y: np.ndarray,
                                masks: dict, arena_bounds: dict,
                                grid_size: int = 50, output_dir: str = None):
    for behavior, mask in masks.items():
        x_b = x[mask]; y_b = y[mask]
        if len(x_b) < 10:
            continue  # yeterli veri yok
        heatmap, _, _ = np.histogram2d(
            x_b[~np.isnan(x_b)], y_b[~np.isnan(y_b)],
            bins=grid_size,
            range=[[arena_bounds["x_min"], arena_bounds["x_max"]],
                   [arena_bounds["y_min"], arena_bounds["y_max"]]]
        )
        heatmap = heatmap / (heatmap.max() + 1e-9)
        # kaydet...
```

### 3B.4 — Zon Occupancy Süresi

- [ ] Her frame'de hayvanı Shapely `Point.within(Polygon)` ile bir zona ata
- [ ] Her zonda geçirilen süreyi hesapla (frames × 1/fps)
- [ ] Zon geçiş sayılarını hesapla (zonlar arası geçişler)
- [ ] Rearing ve grooming'in zona göre dağılımını hesapla

```python
from shapely.geometry import Point, Polygon

def assign_zone(x_frame: float, y_frame: float, zones: dict) -> str:
    pt = Point(x_frame, y_frame)
    for zone_name, coords in zones.items():
        if Polygon(coords).contains(pt):
            return zone_name
    return "disari"

def zone_behavior_breakdown(zone_sequence: list, ethogram: pd.DataFrame,
                             fps: int = 25) -> dict:
    """
    Her zon × her davranış için süre ve oranı döndürür.
    Örnek: rearing'in %40'ı secim zonunda gerçekleşiyor mu?
    """
    zone_arr = np.array(zone_sequence)
    zones = ['giris', 'govde', 'secim', 'sol_kol', 'sag_kol', 'sol_goal', 'sag_goal']
    behaviors = ['rearing', 'grooming']
    results = {}

    for zone in zones:
        in_zone = (zone_arr == zone)
        for behavior in behaviors:
            b_mask = ethogram[behavior].values
            overlap = (in_zone & b_mask).sum()
            results[f'{zone}_{behavior}_s'] = overlap / fps
            results[f'{zone}_{behavior}_pct'] = (
                overlap / b_mask.sum() * 100 if b_mask.sum() > 0 else 0
            )

    return results
```

**Hipotez:** Rearing'in `secim` zonunda yoğunlaşması → karar noktasında anksiyete göstergesi olabilir.

### 3B.5 — Seans Başına Heatmap Dışa Aktarımı

- [ ] 12 T-maze seans için tüm heatmap türlerini üret ve kaydet
- [ ] Numpy dizilerini downstream CNN girdisi için PNG'nin yanında kaydet

```
data/figures/heatmaps/tmaze/{grup}/
├── MA1_s1_occupancy_all.png    + .npy
├── MA1_s1_occupancy_rearing.png + .npy
├── MA1_s1_occupancy_grooming.png + .npy
├── MA1_s1_velocity.png
├── MA1_s1_nose.png
└── MA1_s1_paths.png
```

### 3B.6 — Grup Ortalama Heatmap'ler

- [ ] Aynı gruptaki normalize heatmap'leri üst üste katmanla ve ortalamasını al
- [ ] Her davranış için grup karşılaştırma paneli üret (4 panel yan yana)

```python
def group_average_heatmap(heatmap_list: list) -> np.ndarray:
    """Normalize edilmiş heatmap'lerin piksel bazında ortalaması."""
    stacked = np.stack(heatmap_list, axis=0)
    return stacked.mean(axis=0)
```

**Üretilecek karşılaştırma panelleri:**

| Panel | İçerik |
|-------|--------|
| `group_compare_all.png` | 4 grup × tüm frameler |
| `group_compare_rearing.png` | 4 grup × yalnızca rearing |
| `group_compare_grooming.png` | 4 grup × yalnızca grooming |
| `group_compare_locomotion.png` | 4 grup × yalnızca lokomotion |

---

## Zon Occupancy Metrik Tablosu

Her seans için aşağıdaki zon metrikleri hesaplanır ve `data/features/tmaze_zone_metrics.csv`'e eklenir:

| Sütun | Açıklama |
|-------|----------|
| `secim_dwell_s` | Seçim zonunda geçirilen toplam süre |
| `secim_rearing_s` | Seçim zonundaki rearing süresi |
| `secim_grooming_s` | Seçim zonundaki grooming süresi |
| `govde_rearing_pct` | Gövdedeki rearing'in % dağılımı |
| `goal_entry_count` | Goal zon girişi sayısı |
| `sol_bias` | sol_goal / (sol_goal + sag_goal) girişleri |

---

## Kabul Kriterleri

- 12 T-maze seans için occupancy, velocity, nose-point, rearing ve grooming heatmap'leri mevcut
- Tüm heatmap'ler normalize ve hem PNG hem `.npy` olarak kaydedilmiş
- Zon occupancy süreleri + davranış dağılımı Stage 04'e hazır
- 4 grup karşılaştırma paneli üretilmiş

---

## Çıktı Dosyaları

| Yol | Açıklama |
|-----|----------|
| `data/arena_config_tmaze.json` | Zon poligon tanımları |
| `data/figures/heatmaps/tmaze/` | Seans başına heatmap görüntüleri ve dizileri |
| `data/features/tmaze_zone_metrics.csv` | Seans başına zon × davranış metrikleri |
| `src/heatmap.py` | Heatmap üretim betiği |

---

## Sonraki Adım

→ **Stage 04:** `STAGE_04_PATH_ANALYSIS.md` (T-maze rota analizi)
