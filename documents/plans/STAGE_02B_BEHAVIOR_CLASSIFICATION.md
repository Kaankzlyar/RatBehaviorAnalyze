# Stage 02B — DeepLabCut Behavior Classification
## Trajectory, Heatmap & Ethogram Pipeline

---

## Overview

Bu plan, DeepLabCut (ResNet50) çıktılarından:
1. **Rota (trajectory) ve heatmap** üretimini
2. **Spesifik davranışların** (rearing, self-grooming vb.) otomatik tespitini ve gruplandırmasını

kapsar. Gruplar: **ASP**, **Greyfurt**, **ASP & Greyfurt**, **Kontrol**

---

## Video Envanteri

| Grup | Arena | Hayvanlar | Video |
|------|-------|-----------|-------|
| Kontrol | Open Field | MA1 | MA1-1, MA1-2, MA1-3 |
| ASP | Open Field | MA3 | MA3-1, MA3-2, MA3-3 |
| Greyfurt | Open Field | MA5 | MA5-1, MA5-2, MA5-3 |
| ASP & Greyfurt | Open Field | MA7 | MA7-1, MA7-2, MA7-3 |
| Kontrol | T-Maze | MA1 | Part1-MA1-1/2/3 |
| ASP | T-Maze | MA3 | Part1-MA3-1/2/3 |
| Greyfurt | T-Maze | MA5 | Part1-MA5-1/2/3 |
| ASP & Greyfurt | T-Maze | MA7 | Part1-MA7-1/2/3 |

---

## Bölüm 1 — Genişletilmiş Body Part Seti

Mevcut 5 noktalı set (`nose, head, neck, body_center, tail_base`) rearing ve grooming
tespiti için **yetersizdir**. Aşağıdaki genişletilmiş set önerilir.

### Önerilen Body Parts (11 nokta)

```yaml
bodyparts:
  - nose
  - head
  - neck
  - left_ear
  - right_ear
  - body_center
  - left_forepaw
  - right_forepaw
  - left_hindpaw
  - right_hindpaw
  - tail_base
```

### Skeleton

```yaml
skeleton:
  - [nose, head]
  - [head, neck]
  - [head, left_ear]
  - [head, right_ear]
  - [neck, body_center]
  - [neck, left_forepaw]
  - [neck, right_forepaw]
  - [body_center, left_hindpaw]
  - [body_center, right_hindpaw]
  - [body_center, tail_base]
```

### Neden Bu Noktalar?

| Nokta | Davranış Tespitindeki Rolü |
|-------|---------------------------|
| `left_forepaw`, `right_forepaw` | Rearing: patiler havaya kalktığında kamera açısında gövdeye yaklaşır |
| `left_ear`, `right_ear` | Grooming: kulak bölgesini tımarlarken burun bu noktaya yaklaşır |
| `left_hindpaw`, `right_hindpaw` | Hareket analizi; rearing sırasında yalnızca arka patiler zeminde |
| `tail_base` | Gövde uzunluğu proxy'si — rearing'de nose-tail_base mesafesi kısalır |

---

## Bölüm 2 — Davranış Tespiti Yaklaşımları

### 2A. Geometric Feature Engineering (Birincil Yöntem)

DLC çıktısından hesaplanan geometrik özellikler, basit eşik veya makine öğrenmesi
sınıflandırıcısıyla davranışa dönüştürülür. **Ek etiketleme gerektirmez.**

#### Rearing (Ayağa Kalkma) Tespiti

Tepeden (top-down) kamera görüntüsünde fare ayağa kalktığında:
- Gövde uzunluğu **kısalır** (önden arkaya projeksiyon küçülür)
- Forepaw'lar gövde merkezine **yaklaşır**
- Baş görece sabit, kuyruk da sabit kalır

```python
import numpy as np
import pandas as pd

def compute_rearing_features(tracking: dict, fps: int = 25) -> pd.DataFrame:
    """
    tracking: load_dlc_csv() çıktısı — her body part için x, y, likelihood array
    """
    nose   = np.stack([tracking['nose']['x'],   tracking['nose']['y']],   axis=1)
    tail   = np.stack([tracking['tail_base']['x'], tracking['tail_base']['y']], axis=1)
    lfp    = np.stack([tracking['left_forepaw']['x'],  tracking['left_forepaw']['y']],  axis=1)
    rfp    = np.stack([tracking['right_forepaw']['x'], tracking['right_forepaw']['y']], axis=1)
    center = np.stack([tracking['body_center']['x'],   tracking['body_center']['y']],   axis=1)

    body_length = np.linalg.norm(nose - tail, axis=1)          # nose-to-tail distance
    forepaw_dist = (
        np.linalg.norm(lfp - center, axis=1) +
        np.linalg.norm(rfp - center, axis=1)
    ) / 2                                                        # avg forepaw-to-center

    # Normalize by median body length to handle scale differences
    median_bl = np.nanmedian(body_length)
    body_length_norm = body_length / median_bl                   # ~1.0 at rest, <0.65 during rearing

    features = pd.DataFrame({
        'body_length_px':        body_length,
        'body_length_norm':      body_length_norm,
        'forepaw_center_dist':   forepaw_dist,
    })
    return features


def detect_rearing(features: pd.DataFrame, fps: int = 25,
                   body_length_threshold: float = 0.70,
                   min_duration_frames: int = 5) -> pd.Series:
    """
    body_length_threshold: normalized body length below this → rearing candidate
    min_duration_frames:   ignore bouts shorter than this (filter noise)
    """
    candidate = features['body_length_norm'] < body_length_threshold

    # Remove short bursts (likely tracking noise)
    from scipy.ndimage import label
    labeled, n_bouts = label(candidate.values)
    rearing = candidate.copy()
    for i in range(1, n_bouts + 1):
        bout_frames = np.where(labeled == i)[0]
        if len(bout_frames) < min_duration_frames:
            rearing.iloc[bout_frames] = False

    return rearing  # Boolean Series, True = rearing frame
```

**Eşik kalibrasyonu:** İlk kez çalıştırmadan önce 2–3 videodan manuel olarak 10–15 rearing
bout'u işaretle, `body_length_norm` dağılımını gözle ve eşiği buna göre ayarla.

#### Self-Grooming / Self-Sniffing (Kendini Tımar / Koklama) Tespiti

Fare kendini koklar/tımarlarken:
- Burun gövdeye veya kulağa yaklaşır
- Baş hareketleri sık, lokal ve tekrarlı
- Yer değiştirme (displacement) düşük

```python
def compute_grooming_features(tracking: dict, fps: int = 25,
                               window_sec: float = 0.5) -> pd.DataFrame:
    nose   = np.stack([tracking['nose']['x'],   tracking['nose']['y']],   axis=1)
    center = np.stack([tracking['body_center']['x'], tracking['body_center']['y']], axis=1)
    lear   = np.stack([tracking['left_ear']['x'],  tracking['left_ear']['y']],  axis=1)
    rear   = np.stack([tracking['right_ear']['x'], tracking['right_ear']['y']], axis=1)

    nose_to_body    = np.linalg.norm(nose - center, axis=1)
    nose_to_ear     = np.minimum(
        np.linalg.norm(nose - lear, axis=1),
        np.linalg.norm(nose - rear, axis=1)
    )

    # Head speed (pixels/frame) — low speed + proximity = grooming candidate
    head = np.stack([tracking['head']['x'], tracking['head']['y']], axis=1)
    head_speed = np.concatenate([[0], np.linalg.norm(np.diff(head, axis=0), axis=1)])

    # Rolling oscillation index: std of nose_to_body over window (high = repetitive motion)
    win = int(window_sec * fps)
    nose_body_series = pd.Series(nose_to_body)
    oscillation = nose_body_series.rolling(win, center=True).std().values

    features = pd.DataFrame({
        'nose_to_body_px':    nose_to_body,
        'nose_to_ear_px':     nose_to_ear,
        'head_speed_px':      head_speed,
        'oscillation_index':  oscillation,
    })
    return features


def detect_grooming(features: pd.DataFrame,
                    proximity_threshold_px: float = 60.0,
                    speed_threshold_px: float = 15.0,
                    oscillation_threshold: float = 5.0,
                    min_duration_frames: int = 10) -> pd.Series:
    """
    Üç koşulun birden sağlandığı frameler → grooming
    proximity_threshold_px:   nose-to-body veya nose-to-ear bu değerin altında
    speed_threshold_px:       baş hızı düşük (hayvan hareket etmiyor)
    oscillation_threshold:    burun pozisyonu salınım indeksi yüksek
    """
    candidate = (
        (features['nose_to_body_px'] < proximity_threshold_px) |
        (features['nose_to_ear_px']  < proximity_threshold_px)
    ) & (
        features['head_speed_px'] < speed_threshold_px
    ) & (
        features['oscillation_index'] > oscillation_threshold
    )

    from scipy.ndimage import label
    labeled, n_bouts = label(candidate.values)
    grooming = candidate.copy()
    for i in range(1, n_bouts + 1):
        bout_frames = np.where(labeled == i)[0]
        if len(bout_frames) < min_duration_frames:
            grooming.iloc[bout_frames] = False

    return grooming
```

### 2B. SimBA ile Supervised Sınıflandırma (Opsiyonel / Altın Standart)

SimBA, DLC koordinatlarını kullanarak davranış sınıflandırması yapan ayrı bir araçtır.
Daha yüksek doğruluk sağlar ancak **manuel etiketleme** gerektirir.

**Ne zaman kullan:** Geometric feature yaklaşımının precision/recall'u tatmin edici değilse.

```
Workflow:
1. DLC CSV'lerini SimBA'ya import et
2. Her grup için 3–5 video üzerinde ~200 frame manuel etiketle
   (rearing: 1, diğer: 0) — ethogram tablosu
3. Random Forest classifier eğit (SimBA bunu otomatik yapar)
4. Kalan tüm videolara inference uygula
5. Bout istatistiklerini export et
```

**SimBA kurulum:**
```bash
conda create -n simba python=3.6 -y
conda activate simba
pip install simba-uw-tf-dev
```

### 2C. B-SOID ile Unsupervised Davranış Keşfi (Opsiyonel)

Önceden tanımlamadığın davranış kümeleri varsa B-SOID kullan.

```python
# B-SOID: DLC koordinatlarından otomatik davranış klüsterleme
# https://github.com/YttriLab/B-SOID
# Çıktı: her frame için cluster ID → hangi cluster'ın hangi davranışa
# karşılık geldiğini manuel olarak yorumlarsın
```

---

## Bölüm 3 — Davranış Etiket Akışı (Önerilen Pipeline)

```
DLC CSV
    │
    ▼
load_dlc_csv()          # likelihood < 0.6 → NaN, interpolate
    │
    ▼
compute_rearing_features()
compute_grooming_features()
    │
    ▼
detect_rearing()        → rearing_mask  (bool Series)
detect_grooming()       → grooming_mask (bool Series)
    │
    ▼
build_ethogram_df()     # frame-level DataFrame
    │
    ├──► bout_statistics()    # boutsayısı, ortalama süre, toplam süre/grup
    └──► behavior_heatmaps()  # sadece rearing/grooming framelerinden heatmap
```

### Ethogram DataFrame Yapısı

```
frame | x_center | y_center | rearing | grooming | locomotion | group   | animal
------|----------|----------|---------|----------|------------|---------|-------
0     | 312.4    | 210.1    | False   | False    | True       | ASP     | MA3
1     | 314.1    | 211.3    | False   | False    | True       | ASP     | MA3
...
542   | 290.0    | 198.7    | True    | False    | False      | ASP     | MA3
```

`locomotion = not rearing and not grooming and speed > threshold`

---

## Bölüm 4 — Trajectory & Heatmap Üretimi

### 4A. Trajectory (Rota)

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_trajectory(ethogram: pd.DataFrame, animal_id: str, group: str,
                    arena_px: tuple = (800, 600), output_path: str = None):
    """
    Rearing/grooming/locomotion davranışlarını renk kodlu çizer.
    """
    colors = {
        'locomotion': '#2196F3',   # mavi
        'rearing':    '#F44336',   # kırmızı
        'grooming':   '#4CAF50',   # yeşil
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, arena_px[0])
    ax.set_ylim(0, arena_px[1])
    ax.set_aspect('equal')
    ax.set_title(f'{group} — {animal_id}')

    for behavior, color in colors.items():
        mask = ethogram[behavior]
        ax.scatter(
            ethogram.loc[mask, 'x_center'],
            ethogram.loc[mask, 'y_center'],
            c=color, s=1, alpha=0.4, label=behavior
        )

    # Hareket çizgisi (locomotion only, ince)
    loco = ethogram[ethogram['locomotion']]
    ax.plot(loco['x_center'].values, loco['y_center'].values,
            color='#90CAF9', linewidth=0.5, alpha=0.3, zorder=0)

    ax.legend(markerscale=6)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
```

### 4B. Heatmap — Tüm Davranışlar

```python
from scipy.stats import gaussian_kde
import numpy as np

def compute_kde_heatmap(x: np.ndarray, y: np.ndarray,
                         arena_px: tuple = (800, 600),
                         resolution: int = 200) -> np.ndarray:
    """Kernel Density Estimation heatmap."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    xi = np.linspace(0, arena_px[0], resolution)
    yi = np.linspace(0, arena_px[1], resolution)
    XX, YY = np.meshgrid(xi, yi)

    positions = np.vstack([XX.ravel(), YY.ravel()])
    values    = np.vstack([x, y])
    kernel    = gaussian_kde(values, bw_method='scott')
    Z = kernel(positions).reshape(resolution, resolution)
    return Z


def plot_group_heatmaps(group_data: dict, behavior: str = 'all',
                         arena_px: tuple = (800, 600)):
    """
    group_data: {'Kontrol': [ethogram_df, ...], 'ASP': [...], ...}
    behavior: 'all' | 'rearing' | 'grooming' | 'locomotion'
    """
    groups = list(group_data.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5))

    for ax, group in zip(axes, groups):
        dfs = group_data[group]
        all_x, all_y = [], []
        for df in dfs:
            if behavior == 'all':
                mask = pd.Series([True] * len(df))
            else:
                mask = df[behavior]
            all_x.append(df.loc[mask, 'x_center'].values)
            all_y.append(df.loc[mask, 'y_center'].values)

        x = np.concatenate(all_x)
        y = np.concatenate(all_y)

        Z = compute_kde_heatmap(x, y, arena_px)
        im = ax.imshow(Z, origin='lower', cmap='hot',
                       extent=[0, arena_px[0], 0, arena_px[1]])
        ax.set_title(f'{group}\n({behavior})')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'Grup Karşılaştırması — {behavior}', fontsize=14)
    return fig
```

### 4C. Davranışa Özel Heatmap Katmanları

Her grup için üç ayrı heatmap üret:

| Heatmap | İçerik | Yorum |
|---------|--------|-------|
| `heatmap_all` | Tüm pozisyonlar | Genel alan kullanımı |
| `heatmap_rearing` | Yalnızca rearing frame'leri | Rearing'in tercih edilen bölgeleri |
| `heatmap_grooming` | Yalnızca grooming frame'leri | Grooming'in tercih edilen bölgeleri |
| `heatmap_locomotion` | Yalnızca hareket frame'leri | Aktif hareket bölgeleri |

---

## Bölüm 5 — Bout İstatistikleri & Grup Karşılaştırması

### 5A. Bout Analizi

```python
def compute_bout_stats(rearing_mask: pd.Series, grooming_mask: pd.Series,
                        fps: int = 25) -> dict:
    """
    Her davranış için:
    - Bout sayısı
    - Ortalama bout süresi (saniye)
    - Toplam süre (saniye)
    - Toplam video süresinin yüzdesi
    """
    from scipy.ndimage import label

    stats = {}
    for behavior_name, mask in [('rearing', rearing_mask), ('grooming', grooming_mask)]:
        labeled, n_bouts = label(mask.values)
        bout_lengths = [
            (labeled == i).sum() / fps
            for i in range(1, n_bouts + 1)
        ]
        total_frames = len(mask)
        stats[behavior_name] = {
            'bout_count':        n_bouts,
            'mean_duration_s':   np.mean(bout_lengths) if bout_lengths else 0,
            'total_duration_s':  sum(bout_lengths),
            'pct_of_session':    (mask.sum() / total_frames) * 100,
        }
    return stats
```

### 5B. Grup Karşılaştırma Tablosu

Son çıktı tablosu:

```
Grup         | Rearing Bouts | Rearing Süre (s) | Grooming Bouts | Grooming Süre (s) | Hareket %
-------------|---------------|------------------|----------------|-------------------|----------
Kontrol      | 12.3 ± 2.1   | 45.2 ± 8.3      | 8.7 ± 1.4     | 32.1 ± 6.2       | 58.4
ASP          | 8.1 ± 1.8   | 28.4 ± 5.7      | 12.3 ± 2.1    | 41.5 ± 7.8       | 49.2
Greyfurt     | ...          | ...              | ...            | ...               | ...
ASP&Greyfurt | ...          | ...              | ...            | ...               | ...
```

### 5C. İstatistiksel Testler

```python
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

# Kruskal-Wallis (non-parametric, 4 grup karşılaştırması)
stat, p = kruskal(kontrol_values, asp_values, greyfurt_values, asp_grey_values)

# Post-hoc: Dunn testi (Bonferroni düzeltmeli)
dunn_results = posthoc_dunn([kontrol_values, asp_values, greyfurt_values, asp_grey_values],
                              p_adjust='bonferroni')
```

---

## Bölüm 6 — Etiket Kalibrasyon Protokolü

Geometric feature eşiklerini doğrulamadan kullanma.

### Adım Adım Kalibrasyon

1. **Manuel referans oluştur**: Her gruptan 1 video seç (4 video toplam).
   - Her videoda rearing ve grooming bout'larını gözle tespit et.
   - Frame aralıklarını kaydet: `{video_id: [(start, end, 'rearing'), ...]}`

2. **Feature dağılımını gözlemle**:
   ```python
   feats = compute_rearing_features(tracking)
   # Rearing framelerinde body_length_norm dağılımı
   rearing_frames = [f for s, e, b in manual_labels if b == 'rearing'
                     for f in range(s, e)]
   print(feats.loc[rearing_frames, 'body_length_norm'].describe())
   ```

3. **ROC eğrisi ile eşik seç**:
   ```python
   from sklearn.metrics import roc_curve, f1_score

   y_true = build_ground_truth(manual_labels, n_frames)
   y_score = 1 - feats['body_length_norm'].values  # high score = rearing

   fpr, tpr, thresholds = roc_curve(y_true, y_score)
   # En iyi F1 eşiğini bul
   f1s = [f1_score(y_true, y_score > t) for t in thresholds]
   best_threshold = 1 - thresholds[np.argmax(f1s)]
   ```

4. **Hedef metrikler**:
   - Precision > 0.80
   - Recall > 0.80
   - F1 > 0.80

---

## Bölüm 7 — Çıktı Dizin Yapısı

```
data/
├── dlc_output/
│   ├── open_field/
│   │   ├── ASP/
│   │   ├── Greyfurt/
│   │   ├── ASP_Greyfurt/
│   │   └── Kontrol/
│   └── tmaze/
│       ├── ASP/
│       ├── Greyfurt/
│       ├── ASP_Greyfurt/
│       └── Kontrol/
│
├── behavior/
│   ├── ethograms/                  # Per-session CSV (frame-level labels)
│   │   └── MA3-1_ethogram.csv
│   ├── bout_stats/                 # Per-session bout statistics
│   │   └── MA3-1_bouts.csv
│   ├── group_stats.xlsx            # Grup karşılaştırma tablosu
│   └── thresholds.json             # Kullanılan eşik değerleri
│
└── figures/
    ├── trajectories/
    │   ├── open_field/
    │   │   ├── ASP_MA3-1_trajectory.png
    │   │   └── ...
    │   └── tmaze/
    ├── heatmaps/
    │   ├── group_all/
    │   ├── group_rearing/
    │   └── group_grooming/
    └── stats/
        ├── bout_counts_boxplot.png
        └── duration_comparison.png
```

---

## Bölüm 8 — Uygulama Sırası

```
[ ] 1. config.yaml'ı 11-noktalı body part seti ile güncelle
[ ] 2. Frame extraction yeniden çalıştır (mevcut labellar varsa merge et)
[ ] 3. Yeni body partları etiketle (özellikle forepaw, hindpaw, ears)
[ ] 4. Modeli yeniden eğit (maxiters=50000)
[ ] 5. Tüm 24 videoya inference uygula
[ ] 6. src/behavior_detection.py yaz (compute + detect fonksiyonları)
[ ] 7. 4 referans videodan manuel kalibrasyon yap → thresholds.json kaydet
[ ] 8. Tüm videolara detection pipeline uygula → ethograms/ üret
[ ] 9. Bout istatistiklerini hesapla → group_stats.xlsx
[ ] 10. Trajectory ve heatmap figürlerini üret → figures/
[ ] 11. Kruskal-Wallis + Dunn testleri → istatistiksel raporlama
```

---

## Bölüm 9 — Sık Karşılaşılan Sorunlar

| Sorun | Sebep | Çözüm |
|-------|-------|-------|
| Rearing yanlış pozitif | Hızlı dönüş sırasında da body_length kısalıyor | Min duration filtresi artır (≥8 frame), forepaw koşulu ekle |
| Grooming yanlış negatif | Burun hızı eşiği çok düşük | `speed_threshold_px` artır |
| Çok az rearing bout | Eşik çok katı | `body_length_threshold` 0.70 → 0.75 yükselt |
| NaN frame'lerde detection | İnterpolasyon yapılmamış | Önce `interpolate_tracking()` çalıştır |
| Forepaw kaybı | Küçük nokta, occlusion sık | Likelihood eşiğini forepaw için 0.5'e indir |

---

## İlgili Dosyalar

| Dosya | Rol |
|-------|-----|
| `src/behavior_detection.py` | Feature engineering + davranış tespiti |
| `src/heatmap_generator.py` | KDE heatmap üretimi |
| `src/trajectory_plotter.py` | Renk kodlu rota çizimi |
| `src/bout_statistics.py` | Bout analizi + grup karşılaştırması |
| `data/behavior/thresholds.json` | Kalibrasyon eşik değerleri |

---

## Sonraki Adım

→ **Stage 03A:** `STAGE_03A_OFT_ANALYSIS.md` — Open Field Testi metrikleri
→ **Stage 03B:** `STAGE_03B_TMAZE_HEATMAPS.md` — T-maze heatmapları
