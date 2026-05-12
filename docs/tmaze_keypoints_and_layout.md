# T-Maze Keypoint Setu ve Dosya Organizasyonu

**Tarih:** 2026-05-05
**Kapsam:** T-maze arm'ı için DeepLabCut etiket setinin tanımı + repo içinde T-maze verisinin/çıktılarının nereye konacağının kararlaştırılması.
**İlişkili:** `documents/RAT_TMAZE_PROJECT.md` (eski plan — tek DLC modeli varsayıyordu, bu doküman onu geçersiz kılıyor).

---

## 1. Bağlam ve karar

Open-field arm'ında 9 noktalı bir DLC modeli kullanıldı (bkz. `docs/behavior_detection_documentation.md`):
`nose, head, left_forepaw, right_forepaw, left_hindpaw, right_hindpaw, tail_base` ve sırt/omuz noktaları.

T-maze kayıtlarında üç farkı durum 9 noktayı verimsiz kılıyor:

1. **Subject daha küçük görünüyor** — kameranın daha geniş bir alanı kapsaması gerektiğinden fare başına düşen piksel sayısı düşük; ince ayrımlı noktaların (hip vs mid-back, left vs right hindpaw) likelihood'u kötüleşiyor ve etiketleme süresi maliyeti faydadan büyük oluyor.
2. **Köşe oklüzyonu** — koridor duvarları köşelerde gövde noktalarını kapatıyor; hip / shoulder / hindpaw'lar dönüşler sırasında sürekli swap ediyor.
3. **Sadece üstten kamera** — patiler, özellikle ön patiler, grooming sırasında çenenin altında kaldığından top-down görüntüde likelihood'ları sürekli düşük kalıyor; modeli yormak dışında bilgi katmıyorlar.

**Karar:** T-maze için ayrı bir DLC projesi, **5 nokta**:

```
nose, ear_L, ear_R, mid_back, tail_base
```

OFT ile bu nedenle DLC modeli paylaşılmıyor — `documents/RAT_TMAZE_PROJECT.md` bölüm 1'deki "Single DeepLabCut model … both arenas" varsayımı bu kararla geçersiz oldu.

---

## 2. Davranış metriklerine eşleme

| Davranış | Metrik | Kullanılan noktalar |
|---|---|---|
| **T-maze navigasyon** (kol seçimi, karar noktasında geçirilen süre, dönüş yarıçapı) | `(mid_back, tail_base)` ile gövde merkezi izleme; `(nose − mid_back)` ile heading açısı | `nose, mid_back, tail_base` |
| **Rearing** (top-down) | `dist(nose, tail_base)` 2D mesafesinin ani düşüşü ("teleskoplaşma") + bbox alanı küçülmesi | `nose, mid_back, tail_base` |
| **Grooming** (top-down) | gövde hızı düşük (`mid_back, tail_base` sabit) **+** `nose` osilasyonu **+** `(ear_L − ear_R)` kafa-ekseninin rotasyonu | `nose, ear_L, ear_R, mid_back` |

Notlar:
- **Kulaklar kritik:** top-down kayıtta kafa yönelimini en stabil veren çift onlar. Burun tek başına çok titreşimli, heading için yeterli değil.
- **Patiler dahil edilmiyor:** top-down grooming sırasında zaten görünmüyorlar; ileride lateralden ek kamera takılırsa o görüntüye eklenir.
- **Hip / shoulder yok:** küçük gövdede `mid_back` ile sürekli swap ediyor, atlamak daha sağlıklı.
- **Tail mid / tail tip yok:** T-maze'de gövde kıvrım analizi yapılmıyor; ileride istenirse tail mid eklenebilir (uç değil).

---

## 3. Reddedilen alternatifler

| Alternatif | Neden reddedildi |
|---|---|
| 9 noktayı koru, OFT modeli kullan | Küçük subject + duvar oklüzyonu → likelihood kayıpları, etiketleme yükü, swap'lar |
| Tek DLC modeli her iki arenaya | Yukarı bkz. — keypoint setlerinin ayrışması bunu zorunlu kıldı |
| 3 nokta minimum (`nose, mid_back, tail_base`) | Grooming için heading/rotasyon sinyali kayboluyor; kulaklar olmadan grooming-vs-immobil ayrımı düşük SNR |
| Patiler dahil 7 nokta | Top-down grooming'de likelihood çok düşük, gürültü kaynağı |

---

## 4. Dosya organizasyonu

OFT için kullanılan `data/DLCfiltered/<treatment_group>/<subject>/` örüntüsünü aynen kullanıyoruz; subject prefix'i (`OpenField` vs `TMaze`) zaten arenaları ayırıyor, ek bir arena katmanına gerek yok.

### 4.1 Veri dizinleri

```
data/
└── DLCfiltered/
    ├── control/
    │   ├── OpenFieldMA1_1/                      # mevcut
    │   ├── ...
    │   ├── TMazeMA1_1/                          # YENİ
    │   │   ├── TMazeMA1_1.csv                   # 5 nokta × (x, y, lik) DLC çıktısı
    │   │   ├── TMazeMA1_1_arena_geometry.json   # koridor/karar noktası/kol koordinatları
    │   │   ├── TMazeMA1_1_behavior_bouts.csv    # rearing + grooming aralıkları
    │   │   ├── TMazeMA1_1_behavior_frames.csv
    │   │   ├── TMazeMA1_1_behavior_timeline.png
    │   │   ├── TMazeMA1_1_tmaze_metrics.csv     # turn bias, path efficiency, zone dwell, decision time
    │   │   ├── TMazeMA1_1_path_overlay.png      # arena üstüne çizilen yörünge
    │   │   ├── TMazeMA1_1_zone_heatmap.png      # koridor/kol tabanlı occupancy
    │   │   ├── TMazeMA1_1_speed.csv
    │   │   └── TMazeMA1_1_speed.png
    │   ├── TMazeMA1_2/
    │   └── TMazeMA1_3/
    ├── ASP/                                     # MA3 — 3 OFT + 3 TMaze
    ├── Greyfurt/                                # MA5
    ├── ASP ve Greyfurt/                         # MA7
    └── Kare/                                    # OFT-only validation reference (Kutu_v1)
```

OFT'ye özgü çıktılar (`*_orbit_grid.png`, `*_thigmotaxis.png`, `*_kutu_validation.png`, `*_oft_metrics.csv`) **T-maze klasöründe üretilmez**. Onların T-maze'deki karşılığı `*_tmaze_metrics.csv` + `*_path_overlay.png` + `*_zone_heatmap.png`.

### 4.2 Kod dizinleri

```
src/
├── behavior_detection.py            # MEVCUT — keypoint config'i parametrize edilecek
│                                       (OFT 9-pt vs T-maze 5-pt için ayrı profil)
├── dlc/
│   ├── dlc_setup.py                 # MEVCUT — OFT projesi
│   ├── dlc_train.py
│   ├── dlc_inference.py
│   └── tmaze/                       # YENİ — T-maze için ayrı DLC projesi
│       ├── dlc_setup_tmaze.py
│       ├── dlc_train_tmaze.py
│       └── dlc_inference_tmaze.py
└── tmaze/                           # YENİ — T-maze'e özgü modüller
    ├── arena_geometry.py            # koridor + karar noktası + sol/sağ kol kalibrasyonu
    ├── tmaze_metrics.py             # turn bias, path efficiency, dwell, decision latency
    └── zone_classifier.py           # her frame'i {start, corridor, decision, arm_L, arm_R}'a etiketler

analysis/
├── run_analysis.py                  # MEVCUT — arena flag eklenecek (--arena oft|tmaze)
├── tmaze_path_plot.py               # YENİ — yörünge overlay
└── tmaze_zone_heatmap.py            # YENİ — zone-tabanlı occupancy

dlc-projects/                        # YENİ — DLC config.yaml'ları repo dışı tutuluyor mu?
└── (gitignore ile kontrol et — büyük modeller commit'lenmemeli)

docs/
├── behavior_detection_documentation.md   # MEVCUT — OFT
├── tmaze_keypoints_and_layout.md         # BU DOSYA
└── tmaze_metrics_documentation.md        # YENİ — turn bias / dwell tanımları (metrikler kararlaşınca)

reports/
├── figures/                         # MEVCUT
│   ├── oft/                         # mevcut figürler buraya taşınabilir
│   └── tmaze/                       # YENİ
└── tmaze_summary.csv                # YENİ — cohort × turn-bias / efficiency tablosu (analiz başlayınca)
```

### 4.3 Adlandırma kuralları

- **Subject:** `TMazeMA<cohort>_<run>` — README'de zaten belirtilmiş, koruyoruz.
- **DLC CSV:** `<subject>.csv` (filtered), 3 satırlı header (`scorer / bodypart / coord`) — OFT ile aynı format, sadece 5 keypoint.
- **Per-subject çıktılar:** `<subject>_<artifact>.<ext>` — OFT konvansiyonu.
- **Cohort sınıflandırması:** `data/DLCfiltered/<treatment_group>/` altında, mevcut Türkçe klasör adları korunuyor (`control`, `ASP`, `Greyfurt`, `ASP ve Greyfurt`).

---

## 5. Geçiş ve bir sonraki adımlar

1. **DLC projesi:** `src/dlc/tmaze/dlc_setup_tmaze.py` ile yeni proje yarat, bodypart listesi yukarıdaki 5 nokta.
2. **Etiketleme:** 12 video × ~40-60 frame ≈ 500-700 etiketlenmiş frame hedefle (OFT'de 9 nokta için 12×120 idi; nokta sayısı azaldığı için frame başına süre düşüyor, video başına frame sayısını biraz artırabilirsin).
3. **`behavior_detection.py` refaktörü:** keypoint isimleri hard-coded; bir `KeypointProfile` (oft / tmaze) sözlüğü ekle, her metriğin hangi keypoint'i istediğini bu profilden çek. Aynı dosya iki arenaya da çalışsın.
4. **Arena geometrisi:** T-maze için her video'ya bir kez koridor / karar / sol kol / sağ kol köşe noktaları işaretlenmeli (`tmaze/arena_geometry.py`); piksel kalibrasyonu OFT'deki gibi config'e yazılır.
5. **Hafıza notu:** `memory/project_tmaze_postponed.md` "indefinitely deferred" diyordu — T-maze tekrar gündemdeyse onu güncelle (ya da kaldır).

---

## 6. Karar özeti

- T-maze için **5 nokta**: `nose, ear_L, ear_R, mid_back, tail_base`.
- **Ayrı DLC projesi**, OFT modeli paylaşılmıyor.
- Veri layout'u OFT örüntüsünü aynen kullanıyor: `data/DLCfiltered/<treatment_group>/TMazeMA<cohort>_<run>/`.
- T-maze'e özgü kod `src/tmaze/`, `src/dlc/tmaze/`, `analysis/tmaze_*.py` altında; ortak kod (behavior_detection, run_analysis) keypoint profili ile parametrize edilerek paylaşılır.
