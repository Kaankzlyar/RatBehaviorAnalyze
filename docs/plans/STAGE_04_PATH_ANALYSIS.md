# Stage 04 — T-Maze Rota (Path) Analizi
## Plan & Roadmap

---

## Objective

T-maze trajectory'lerinden geometrik, kinematik ve karar-düzeyi metrikler çıkar. Stage 02B davranış etiketlerini her deneme (trial) ile hizala: rearing/grooming'in seçim zonunda gerçekleşmesi anksiyete göstergesi olarak ele alınır. Dört grup: **Kontrol, ASP, Greyfurt, ASP & Greyfurt**.

---

## Status

- [ ] Başlanmadı — Stage 02 (temiz DLC CSV), Stage 02B (ethogram), Stage 03B (zon konfigürasyonu) gerektirir

---

## Girdi Dosyaları

| Dosya | Kaynak |
|-------|--------|
| `data/dlc_output/tmaze/{grup}/{video}_clean.csv` | Stage 02 |
| `data/behavior/ethograms/{video}_ethogram.csv` | Stage 02B |
| `data/arena_config_tmaze.json` | Stage 03B |

---

## Referans Çizgiler & Koordinat Sistemi

```
         ┌──────┐  ┌──────┐
         │  SG  │  │  SG  │    SG = sol_goal / sag_goal
         └──┬───┘  └──┬───┘
            │    SP    │        SP = secim_noktasi zonu
            └────┬─────┘
                 │
              [GOVDE]
                 │
              [GIRIS]
```

**Labirent ekseni:** giriş orta noktasından seçim noktasına dikey çizgi — yön açısı hesaplamalarında kullanılır.

---

## Görevler

### 4.1 — Deneme (Trial) Segmentasyonu

- [ ] Her seansı bireysel denemelere böl (giriş → goal)
- [ ] Deneme başlangıcı: fare `govde` zonuna `giris`'ten geçince
- [ ] Deneme sonu: fare `sol_goal` veya `sag_goal`'a ulaşınca
- [ ] Zaman aşımlı veya anormal denemeleri çıkar (timeout > 60s)
- [ ] Deneme indeksini başlangıç/bitiş frame numaraları ile dışa aktar

```python
def segment_trials(zone_sequence: list, fps: int, timeout_s: int = 60) -> list:
    """
    zone_sequence: her frame için zon etiketi listesi
    Returns: [(start_frame, end_frame, secenek)] tuple'ları
    """
    trials = []
    in_trial = False
    for i, zone in enumerate(zone_sequence):
        if zone == "govde" and not in_trial:
            start = i
            in_trial = True
        if in_trial and zone in ("sol_goal", "sag_goal"):
            trials.append((start, i, zone))
            in_trial = False
        if in_trial and (i - start) / fps > timeout_s:
            in_trial = False  # zaman aşımı — denemeyi çıkar
    return trials
```

### 4.2 — Dönüş Yanlılığı (Turn Bias)

- [ ] Seans başına sol vs sağ goal seçimlerini say
- [ ] `turn_bias = sol_secimler / (sol_secimler + sag_secimler)` hesapla
- [ ] Perseverasyon bayrağı: > 3 ardışık aynı taraf seçimi

### 4.3 — Rota Verimliliği (Path Efficiency)

- [ ] Gerçek rota uzunluğunu hesapla (trajectory boyunca Öklid toplamı)
- [ ] Optimal (düz çizgi) rota uzunluğunu hesapla (giriş → goal)
- [ ] `path_efficiency = optimal / actual` (1.0 = mükemmel düz çizgi)

```python
def path_efficiency(x: np.ndarray, y: np.ndarray,
                    start_frame: int, end_frame: int) -> float:
    traj_x = x[start_frame:end_frame]
    traj_y = y[start_frame:end_frame]
    dx = np.diff(traj_x); dy = np.diff(traj_y)
    actual  = np.nansum(np.sqrt(dx**2 + dy**2))
    optimal = np.sqrt((traj_x[-1] - traj_x[0])**2 + (traj_y[-1] - traj_y[0])**2)
    return optimal / actual if actual > 0 else np.nan
```

### 4.4 — Karar Gecikmesi (Decision Latency)

- [ ] Deneme başına `secim` zonunda geçirilen süre (frames × 1/fps)
- [ ] Deneme başından goal'a toplam gecikmeyi de hesapla
- [ ] Yüksek gecikmeli denemeleri bayrakla (rat ortalamasının > 2 SD üstü)

### 4.5 — Hız Profili

- [ ] body_center'dan anlık hız hesapla (cm/s)
- [ ] Deneme ve zon başına ortalama hız hesapla:
  - `velocity_govde`, `velocity_secim`, `velocity_kol`
- [ ] Donma olaylarını tespit et ve say (hız < 2 cm/s ve > 1s)

### 4.6 — Yön Açısı (Heading Angle)

- [ ] Yön açısı = burun-kuyruk vektörü ile labirent ekseni arasındaki açı
- [ ] `secim` zonundaki ortalama yön açısı: yönsel kararlılık göstergesi
- [ ] Yön varyansı: yüksek varyans = kararsız / anksiyöz

```python
def heading_angle(nose_x: np.ndarray, nose_y: np.ndarray,
                  tail_x: np.ndarray, tail_y: np.ndarray,
                  maze_axis_angle: float = 90.0) -> np.ndarray:
    dx = nose_x - tail_x
    dy = nose_y - tail_y
    angle = np.degrees(np.arctan2(dy, dx))
    return angle - maze_axis_angle  # labirent eksenine göre
```

### 4.7 — Geri Dönüş Oranı (Backtrack Rate)

- [ ] Zon geri dönüşlerini tespit et: `govde → secim → govde` aynı denemede
- [ ] `backtrack_rate = geri_donus / toplam_deneme` hesapla

### 4.8 — Rota Pürüzsüzlüğü (Trajectory Smoothness)

- [ ] Her noktada eğriliği hesapla: `κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)`
- [ ] Deneme başına ortalama eğrilik → düşük = pürüzsüz, yüksek = düzensiz

### 4.9 — Davranış-Deneme Entegrasyonu (Stage 02B)

Her deneme için ethogram verilerini hizala ve davranış özelliklerini çıkar.

- [ ] Deneme başına rearing frame sayısını say
- [ ] Deneme başına grooming frame sayısını say
- [ ] Zon başına rearing: seçim, gövde, sol kol, sağ kol
- [ ] Deneme içinde rearing/grooming'in başlangıç zamanlaması

```python
def trial_behavior_features(ethogram: pd.DataFrame,
                             start_frame: int, end_frame: int,
                             zone_sequence: list, fps: int = 25) -> dict:
    """
    Tek bir deneme için davranış özelliklerini hesapla.
    """
    trial_eth = ethogram.iloc[start_frame:end_frame]
    trial_zones = zone_sequence[start_frame:end_frame]

    n_frames = end_frame - start_frame

    # Temel sayılar
    rearing_frames  = trial_eth['rearing'].sum()
    grooming_frames = trial_eth['grooming'].sum()

    # Zon × davranış kesişimi
    in_secim = np.array(trial_zones) == 'secim'
    rearing_in_secim = (trial_eth['rearing'].values & in_secim).sum()

    return {
        'rearing_frames_total':    rearing_frames,
        'rearing_pct_of_trial':    rearing_frames / n_frames * 100,
        'grooming_frames_total':   grooming_frames,
        'grooming_pct_of_trial':   grooming_frames / n_frames * 100,
        'rearing_in_secim_frames': rearing_in_secim,
        'rearing_in_secim_pct':    rearing_in_secim / rearing_frames * 100 if rearing_frames > 0 else 0,
        # rearing başladıktan sonra karar ne kadar sürer?
        'rearing_before_decision': (
            trial_eth['rearing'].values[:in_secim.argmax()].sum() / fps
            if in_secim.any() else np.nan
        ),
    }
```

**Hipotez:** `rearing_in_secim_pct` → ASP ve Greyfurt gruplarında Kontrol'e kıyasla daha yüksek bekleniyor (seçim noktasında anksiyete kaynaklı rearing).

### 4.10 — Görselleştirme

- [ ] Deneme başına hız renk kodlu trajectory çiz (colormap: soğuk→sıcak)
- [ ] Bir seansın tüm denemelerini üst üste bindir (rota yoğunluğu)
- [ ] Davranış renkli rota: rearing = kırmızı, grooming = yeşil, lokomotion = mavi

---

## Metrik Özet Tablosu

| Metrik | Birim | Düzey | Betik |
|--------|-------|-------|-------|
| `turn_choice` | sol/sag | deneme | path_analysis.py |
| `turn_bias` | 0–1 | seans | path_analysis.py |
| `path_efficiency` | 0–1 | deneme | path_analysis.py |
| `decision_latency_s` | s | deneme | path_analysis.py |
| `total_latency_s` | s | deneme | path_analysis.py |
| `velocity_govde` | cm/s | deneme | path_analysis.py |
| `velocity_secim` | cm/s | deneme | path_analysis.py |
| `velocity_kol` | cm/s | deneme | path_analysis.py |
| `heading_angle_mean` | derece | deneme | path_analysis.py |
| `heading_angle_var` | derece² | deneme | path_analysis.py |
| `backtrack_rate` | 0–1 | seans | path_analysis.py |
| `trajectory_smoothness` | 1/px | deneme | path_analysis.py |
| `rearing_pct_of_trial` | % | deneme | path_analysis.py |
| `rearing_in_secim_pct` | % | deneme | path_analysis.py |
| `grooming_pct_of_trial` | % | deneme | path_analysis.py |
| `rearing_before_decision_s` | s | deneme | path_analysis.py |

---

## Kabul Kriterleri

- 12 T-maze seans için deneme başına metrikler dışa aktarılmış
- `tmaze_metrics.csv` eksik seans içermiyor
- Davranış-deneme metrikleri `rearing_in_secim_pct` dahil doldurulmuş
- En az 2 örnek seans için görselleştirme kaydedilmiş

---

## Çıktı Dosyaları

| Yol | Açıklama |
|-----|----------|
| `data/features/tmaze_metrics.csv` | Deneme başına T-maze metrikleri (davranış dahil) |
| `data/figures/trajectories/tmaze/` | Trajectory görselleştirme grafikleri |
| `src/path_analysis.py` | Analiz betiği |

---

## Sonraki Adım

→ **Stage 05:** `STAGE_05_FEATURE_ENGINEERING.md`
