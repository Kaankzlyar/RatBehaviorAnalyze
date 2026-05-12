# analysis/plus_maze — Script Cikti Haritasi

Her script, ne alir, ne uretir ve nereye kaydeder.

---

## 0. `show_frame_coords.py`

**Calistirma:** `python show_frame_coords.py --video <video.avi>`  
**Dosya ciktisi:** Yok  
**Terminal ciktisi:** Arm koordinatlari — diger scriptlere arguman olarak girilir

```
--bottom-arm xmin xmax ymin ymax
--left-arm   xmin xmax ymin ymax
--right-arm  xmin xmax ymin ymax
--top-arm    xmin xmax ymin ymax
```

---

## 1. `tmaze_metrics.py`

**Calistirma:** Her sican icin ayri, arm koordinatlariyla  
**Girdi:** `data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>.csv`

**Ciktilar (subject klasorune kaydedilir):**

```
data/DLCfiltered/<kohort>/PlusMaze<ID>/
  PlusMaze<ID>_plus_maze_metrics.csv   <- zon/giris/alternasyon/lokomotor metrikleri (1 satir)
  PlusMaze<ID>_arm_coords.json         <- arm koordinatlari (ethological_features icin)
```

---

## 2. `batch_metrics.py`

**Calistirma:** Tek calistirma, tum subject'ler  
**Girdi:** `data/DLCfiltered/` altindaki tum `*_plus_maze_metrics.csv`

**Cikti:**

```
data/
  plus_maze_metrics_all.csv            <- 12 satir, 35+ sutun (tum kohortlar + turetilmis EPM sutunlar)
```

---

## 3. `orbit_plot.py`

**Calistirma:** Her sican icin ayri, arm koordinatlariyla  
**Girdi:** `data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>.csv`

**Ciktilar (subject klasorune kaydedilir):**

```
data/DLCfiltered/<kohort>/PlusMaze<ID>/
  PlusMaze<ID>_plus_maze_orbit.png     <- body_center trajektoru, zone'lara gore renkli
  PlusMaze<ID>_plus_maze_bodyparts.png <- 4 panel: nose / head / body_center / tail_base
```

---

## 4. `activity_heatmap.py`

**Calistirma:** Her sican icin ayri, arm koordinatlariyla  
**Girdi:** `data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>.csv`

**Ciktilar (subject klasorune kaydedilir):**

```
data/DLCfiltered/<kohort>/PlusMaze<ID>/
  PlusMaze<ID>_heatmap_kde.png         <- KDE (Gaussian blur) yogunluk haritasi
  PlusMaze<ID>_heatmap_histogram.png   <- 2D histogram haritasi
```

---

## 5. `ethological_features.py`

**Calistirma:** Tek calistirma, tum subject'ler  
**Girdi:** DLC CSV + `_arm_coords.json` (rearing icin; yoksa rearing atlanir)

**Ciktilar:**

```
reports/
  ethological_metrics_epm.csv          <- grooming / rearing / SAP metrikleri (tum kohortlar)

reports/figures/plus_maze/
  ethological_grooming.png             <- kohort bazinda grooming % suresi
  ethological_rearing.png              <- kohort bazinda rearing % suresi (arm_coords mevcutsa)
  ethological_sap.png                  <- kohort bazinda SAP % suresi
  ethological_summary.png              <- 2-3 panel ozet
```

**Grooming 4 dal:** tight / loose / upright / occluded (OFT ile esit hassasiyet)  
**Rearing 6 kural:** compact / top-arm / bottom-arm / side-arm / wall-press (arm siniri gerekli)  
**Son adim:** `grooming = grooming_posture AND NOT rearing`

---

## 6. `markov_analysis.py`

**Calistirma:** Tek calistirma  
**Girdi:** `data/plus_maze_metrics_all.csv` — `entry_sequence` sutunu

**Ciktilar:**

```
reports/
  markov_transition_matrices.csv       <- gecis olasiliklari uzun format (cohort/from/to/prob)
  markov_transition_counts.csv         <- ham gecis sayilari

reports/figures/plus_maze/
  markov_heatmaps.png                  <- 4 kohort x 4x4 Markov isi haritasi
  markov_perseveration.png             <- perseverasyon / acik kol / kapali kol bar chart
```

---

## 7. `cohort_stats_epm.py`

**Calistirma:** Tek calistirma  
**Girdi:** `data/plus_maze_metrics_all.csv`

**Ciktilar:**

```
reports/
  cohort_epm_kw.csv                    <- Kruskal-Wallis test sonuclari
  cohort_epm_dunn.csv                  <- Dunn post-hoc karsilastirma
  cohort_epm_permanova.csv             <- PERMANOVA sonuclari
```

---

## 8. `plot_open_arm.py`

**Calistirma:** Tek calistirma  
**Girdi:** `data/plus_maze_metrics_all.csv` + `reports/cohort_epm_kw.csv`

**Ciktilar:**

```
reports/figures/plus_maze/
  epm_open_arm_by_cohort.png           <- acik kol % sure kohort karsilastirmasi
  epm_arm_distribution.png             <- 4 kol dagilim grafigi
  epm_locomotor_covariate.png          <- lokomotor kovaryat analizi
```

---

## 9. `behavioral_effect_analysis.py`

**Calistirma:** Tek calistirma  
**Girdi:** `data/plus_maze_metrics_all.csv` + `data/oft_metrics_all.csv`

**Ciktilar:**

```
reports/effect_analysis/
  behavioral_effect_table.csv          <- etki buyuklugu tablosu (Cohen d / eta-sq)
  behavioral_radar.png                 <- radar chart kohort karsilastirmasi
  behavioral_effect_bars.png           <- etki buyuklugu bar chart
```

---

## 10. `run_analysis.py`

**Pipeline yoneticisi** — tek bir subject icin asagidakileri sirayla cagiriror:

```
tmaze_metrics.py -> orbit_plot.py -> activity_heatmap.py -> speed_analysis.py
```

Kendine ait dosya ciktisi yok; her alt script kendi dosyasini uretir.

---

## Calistirma Sirasi (tam pipeline)

```
1.  show_frame_coords.py          # koordinat al (her video icin, bir kere)
          |
2.  tmaze_metrics.py              # bireysel metrik + arm_coords.json  (her sican icin)
          |
3.  batch_metrics.py              # data/plus_maze_metrics_all.csv
          |
4.  cohort_stats_epm.py           # istatistik CSV'leri
          |
     +----+----+----+
     |         |    |
5a. plot_open_arm.py
5b. markov_analysis.py            # (paralel calistirilabilir)
5c. ethological_features.py
          |
6.  behavioral_effect_analysis.py # EPM + OFT birlestik etki analizi
```

`orbit_plot.py` ve `activity_heatmap.py` subject bazli olup istedigi zaman calistirilabilir;
ana pipeline akisini bloklamamalar.

---

## Iliskili Dosyalar

| Dosya | Aciklama |
|---|---|
| `data/plus_maze_metrics_all.md` | `plus_maze_metrics_all.csv` uretim sureci ve sutun aciklamalari |
| `reports/figures/plus_maze/markov_heatmaps.md` | `markov_heatmaps.png` okuma rehberi |
| `reports/figures/plus_maze/markov_perseveration.md` | `markov_perseveration.png` okuma rehberi |
