# data/plus_maze_metrics_all.csv — Uretim Sureci

**Satir sayisi:** 12  (3 sican x 4 kohort)  
**Sutun sayisi:** 35  
**Son guncelleme:** `analysis/plus_maze/batch_metrics.py` calistirildiktan sonra otomatik uretilir

---

## Ozet

Bu dosya dogrudan elle doldurulmaz. Iki asama halinde otomatik uretilir:

```
Adim 1 — tmaze_metrics.py    (her sican icin ayri calistirilir)
            -> data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>_plus_maze_metrics.csv

Adim 2 — batch_metrics.py    (bir kere calistirilir)
            -> data/plus_maze_metrics_all.csv
```

---

## Adim 0 — On Kosul: Koordinat Secimi

`tmaze_metrics.py` calistirilmadan once her video icin dort kolun piksel sinirlarinin bilinmesi
gerekir. Bu sinirlar interaktif olarak su sekilde elde edilir:

```
python analysis/plus_maze/show_frame_coords.py --video <video.avi>
```

Acilan pencerede her kol icin 4 kose tiklenir; script terminal ciktisinda su formati verir:

```
--bottom-arm xmin xmax ymin ymax
--left-arm   xmin xmax ymin ymax
--right-arm  xmin xmax ymin ymax
--top-arm    xmin xmax ymin ymax
```

Bu degerler bir sonraki adima arguman olarak aktarilir.

---

## Adim 1 — tmaze_metrics.py: Bireysel Metrik Hesaplama

**Script:** `analysis/plus_maze/tmaze_metrics.py`  
**Girdi:** `data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>.csv`  (DLC ciktisi — kare bazinda vucüt parcasi koordinatlari)  
**Cikti:** `data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>_plus_maze_metrics.csv`

### Veri on isleme

| Adim | Islem | Parametre |
|---|---|---|
| 1 | DLC CSV'den `body_center` x, y, likelihood okunur | — |
| 2 | Dusuk guvenilirlikli kareler NaN yapilir | likelihood < 0.60 |
| 3 | Anlik ziplama hareketleri NaN yapilir | >60 px / kare |
| 4 | Kayan pencere medyan uygulama (gurultu giderme) | pencere = 5 kare |

### Zon atamasi

Her kare, koordinatlarina gore dort koldan birine veya "junction" (kavsaklik) bolgesine atanir.
Zon sinirlarini belirleyen degerler `--bottom-arm / --left-arm / --right-arm / --top-arm`
argumanlariyla girilir.

### Kol girisi tespiti

Ardisik kare serisi analiz edilerek kollar arasi gecisler belirlenir:
- Hayvanin bir kola "giris" yapmasi sayilmasi icin o kolda **en az 3 ardisik kare** (30fps'te ~0.10s) kalmasi gerekir
- Kisa titreme seklindeki yanlis tespitler bu esikle bastirilir
- Her giris `entry_sequence` sutununa kaydedilir: ornek → `B->T->B->L->B->R->...`

### Hesaplanan metrikler

**Zon suresi:**

| Sutun | Aciklama |
|---|---|
| `pct_time_bottom/left/right/top` | Her kolda gecirilen sure (%) |
| `pct_time_junction` | Kavsaklikta gecirilen sure (%) |
| `time_bottom/left/right/top_s` | Her kolda gecirilen sure (saniye) |

**Kol girisleri:**

| Sutun | Aciklama |
|---|---|
| `total_entries` | Toplam kol girisi |
| `bottom/left/right/top_entries` | Her kola ayri ayri giris sayisi |
| `most_visited_arm` | En cok ziyaret edilen kol |
| `arm_preference_index` | (max_giris - min_giris) / toplam  (0=esit, 1=tek kol) |

**Alternasyon:**

| Sutun | Aciklama |
|---|---|
| `successive_alternation_pct` | Ardisik girislerin farkli olma yuzdesi |
| `tetrad_alternation_pct` | Ardisik 4 giris setinin 4 farkli kolu kapsama yuzdesi |
| `perseveration_count` | Ayni kola arka arkaya giris sayisi |
| `perseveration_rate_pct` | Perseverasyon yuzdesi |
| `entry_sequence` | Tum giris siralamasi (B/T/L/R harfleriyle) |

**Lokomotor:**

| Sutun | Aciklama |
|---|---|
| `mean_speed_px_s` | Ortalama hareket hizi (piksel/saniye) |
| `total_distance_px` | Toplam hareket mesafesi (piksel) |

### Cikti

Her sican icin kendi DLC klasorune kaydedilir:
```
data/DLCfiltered/control/PlusMazeMA1_1/PlusMazeMA1_1_plus_maze_metrics.csv   (1 satir)
data/DLCfiltered/control/PlusMazeMA1_2/PlusMazeMA1_2_plus_maze_metrics.csv   (1 satir)
...
```

---

## Adim 2 — batch_metrics.py: Birlestirme ve Turetilmis Sutunlar

**Script:** `analysis/plus_maze/batch_metrics.py`  
**Girdi:** `data/DLCfiltered/` altindaki tum `*_plus_maze_metrics.csv` dosyalari  
**Cikti:** `data/plus_maze_metrics_all.csv`

### Tarama mantigi

Script `data/DLCfiltered/` altindaki dort kohort klasorunu sirasiyla tarar:

| Klasor | Kohort etiketi |
|---|---|
| `control/` | Control |
| `ASP/` | Aspartame |
| `Greyfurt/` | Grapefruit |
| `ASP ve Greyfurt/` | ASP+Greyfurt |

Her klasor altinda `PlusMaze` ile baslayan alt dizinler bulunur. Her birinde
`*_plus_maze_metrics.csv` dosyasi aranir ve bulunursa yuklenir.

### Turetilmis sutunlar (batch_metrics tarafindan eklenir)

`tmaze_metrics.py` bu sutunlari uretmez; birlesim sirasinda hesaplanir:

| Sutun | Formul | Anlami |
|---|---|---|
| `pct_open_arm` | `pct_time_left + pct_time_right` | Acik kolda gecirilen toplam sure (%) |
| `pct_closed_arm` | `pct_time_top + pct_time_bottom` | Kapali kolda gecirilen toplam sure (%) |
| `pct_open_arm_entries` | `(left + right) / total * 100` | Girislerin acik kola dusen yuzdesi |
| `pct_closed_arm_entries` | `(top + bottom) / total * 100` | Girislerin kapali kola dusen yuzdesi |
| `anxiety_index_epm` | `(pct_open_arm + pct_open_arm_entries) / 2` | EPM anksiyete indeksi — yuksek = dusuk kaygi |

### Sutun siralaması

Birlesik tabloda meta sutunlar once, ham metrikler ortada, turetilmis EPM sutunlari en sonda gelir:

```
[Meta]      subject_id, cohort_id, cohort, session, n_frames, n_valid_frames, session_duration_s
[Ham]       pct_time_*, time_*_s, *_entries, alternation, perseveration, locomotion, entry_sequence
[Turetilmis] pct_open_arm, pct_closed_arm, pct_open_arm_entries, pct_closed_arm_entries, anxiety_index_epm
```

---

## Yeniden Uretmek Icin

Koordinatlar mevcut ve DLC CSV'leri yerli yerindeyse yalnizca Adim 2 yeterlidir:

```bash
python analysis/plus_maze/batch_metrics.py
```

Koordinatlar degismisse veya yeni bir sican eklenirse once Adim 1 tekrar calistirilmalidir:

```bash
python analysis/plus_maze/show_frame_coords.py --video <video.avi>

python analysis/plus_maze/tmaze_metrics.py \
    --csv data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>.csv \
    --bottom-arm ... --left-arm ... --right-arm ... --top-arm ...

python analysis/plus_maze/batch_metrics.py
```

---

## Iliskili Dosyalar

- `analysis/plus_maze/show_frame_coords.py` — koordinat secici (on kosul)
- `analysis/plus_maze/tmaze_metrics.py` — bireysel metrik hesaplayici
- `analysis/plus_maze/batch_metrics.py` — birlestirici ve turetilmis sutun ekleyici
- `analysis/plus_maze/run_analysis.py` — tmaze_metrics dahil tam pipeline yoneticisi
- `reports/markov_transition_matrices.csv` — bu CSV'den turetilmis Markov gecis verileri
