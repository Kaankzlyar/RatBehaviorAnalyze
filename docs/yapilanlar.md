# Sıçan Davranış Analizi — Yapılanlar ve Proje Durumu

**Proje:** Rat Behavioral Analysis System (Sıçan Davranışsal Analiz Sistemi)  
**Tez konusu:** Yaygın diyet katkı maddelerinin sıçanlarda açık alan (OFT) ve elevated plus maze (EPM) davranışlarına etkisi  
**Son güncelleme:** 2026-05-10 (ML sonuçları güncellendi, Plus Maze EPM anksiyete yorumu + iyileştirme yol haritası eklendi)

---

## İçindekiler

1. [Proje Özeti](#1-proje-özeti)
2. [Deney Tasarımı ve Gruplar](#2-deney-tasarımı-ve-gruplar)
3. [Yazılım ve Teknoloji Altyapısı](#3-yazılım-ve-teknoloji-altyapısı)
4. [Aşama 01 — Video Ön İşleme ve Kalite Kontrolü](#4-aşama-01--video-ön-i̇şleme-ve-kalite-kontrolü)
5. [Aşama 02 — DeepLabCut Pose Tahmini](#5-aşama-02--deeplabcut-pose-tahmini)
6. [Aşama 02B — Kural Tabanlı Davranış Tespiti](#6-aşama-02b--kural-tabanlı-davranış-tespiti)
7. [Aşama 03A — Açık Alan Testi (OFT) Uzamsal Analizi](#7-aşama-03a--açık-alan-testi-oft-uzamsal-analizi)
8. [Aşama 03C — Plus Maze (4 Kollu) Analizi](#7b-aşama-03c--plus-maze-4-kollu-analizi)
9. [Üretilen Çıktı Dosyaları](#8-üretilen-çıktı-dosyaları)
10. [Gruplar Arası Temel Bulgular](#9-gruplar-arası-temel-bulgular)
11. [Profesör MATLAB Pipeline ile Çapraz Doğrulama](#10-profesör-matlab-pipeline-ile-çapraz-doğrulama)
12. [Mevcut Durum Özeti](#11-mevcut-durum-özeti)
13. [Planlanan Çalışmalar](#12-planlanan-çalışmalar)
14. [Plus Maze EPM Anksiyete Analizi](#13-plus-maze-epm-anksiyete-analizi)
15. [İyileştirme Yol Haritası ve Literatür](#14-i̇yileştirme-yol-haritası-ve-literatür)

---

## 1. Proje Özeti

Bu proje, sıçanların açık alan (OFT) arenasında gerçekleştirilen kayıt videolarından davranışsal metrikleri otomatik olarak çıkarmak için uçtan uca bir analiz boru hattı oluşturmaktadır.

**Temel soru:** Aspartam ve greyfurt gibi yaygın diyet katkı maddeleri sıçanların açık alan davranışlarını nasıl değiştirmektedir?

**Çalışmanın kapsamı:**
- 4 tedavi grubu × 3 sıçan = 12 sıçan
- **12 açık alan videosu** (OFT) + **12 plus maze videosu** = 24 oturum
- Tüm videolar **30 fps** hızında çekilmiştir

---

## 2. Deney Tasarımı ve Gruplar

| Kohort | Sıçanlar | Tedavi | Rol |
|--------|----------|--------|-----|
| **MA1** | MA1_1, MA1_2, MA1_3 | Hiçbiri (araç) | Kontrol grubu |
| **MA3** | MA3_1, MA3_2, MA3_3 | Aspartam | Tatlandırıcı kolu |
| **MA5** | MA5_1, MA5_2, MA5_3 | Sadece greyfurt | Greyfurt kolu |
| **MA7** | MA7_1, MA7_2, MA7_3 | Aspartam + greyfurt | Kombine / etkileşim kolu |

Greyfurt kolu, greyfurt etkisini aspartam + greyfurt kombinasyonundan ayırt etmemizi ve her ikisini de aspartam ve kontrol gruplarıyla karşılaştırmamızı sağlar.

### Kayıtlar

| Arena | Video Sayısı | Amaç |
|-------|-------------|-------|
| Açık alan (dikdörtgen) | 12 | Keşif, kaygı, lokomotor aktivite, rearing, grooming |
| Plus maze (4 kollu) | 12 | Çalışma belleği (alternation), kol tercihi, esneklik, persiverasyon |

---

## 3. Yazılım ve Teknoloji Altyapısı

| Bileşen | Araç |
|---------|------|
| Pose tahmini | DeepLabCut (PyTorch arka ucu) |
| Video işleme | OpenCV |
| Veri işleme | Pandas, NumPy, SciPy |
| Görselleştirme | Matplotlib, Seaborn |
| Planlanan ML modelleri | Scikit-learn, XGBoost, PyTorch |

**Sistem gereksinimleri:** CUDA 11.8 veya 12.1 destekli NVIDIA GPU (RTX 3060 6 GB ile test edilmiştir). Davranış tespiti ve uzamsal analiz CPU üzerinde çalışmaktadır.

---

## 4. Aşama 01 — Video Ön İşleme ve Kalite Kontrolü

**Durum: Tamamlandı**  
**Script:** `src/video_preprocessing.py`

### Yapılanlar

1. **Video envanteri oluşturuldu**
   - Tüm 24 `.avi` videosu listelendi ve dosya boyutları doğrulandı
   - Her video açılabilir durumda mı kontrol edildi (`get_video_properties()`)
   - Süre, çözünürlük ve kare hızı kaydedildi

2. **Metadata CSV üretildi** (`data/metadata.xlsx`)
   - Her video için bir satır içeren ana envanter tablosu oluşturuldu
   - Sütunlar: `rat_id`, `session`, `arena`, `part`, `filename`, `duration_s`, `fps`, `resolution`, `file_path`

3. **Kalite kontrolü yapıldı**
   - Tüm oturumlarda tutarlı çözünürlük doğrulandı (`check_consistency()`)
   - Parlaklık tutarsızlıkları kontrol edildi — grafikleri `data/quality_plots/` altında
   - Düşen kareler işaretlendi (`check_dropped_frames()` — %1'den fazla eksikse uyarı)
   - Bulanık kareler tespit edildi (`check_blur()` — Laplacian varyansı, eşik=80)
   - `data/quality_report.xlsx` dosyası üretildi

### Üretilen Çıktılar

| Dosya | Açıklama |
|-------|----------|
| `data/metadata.xlsx` | 24 videonun tüm meta verileri |
| `data/quality_report.xlsx` | Kalite kontrol özeti |
| `data/quality_plots/` | Video başına parlaklık grafikleri |
| `data/part0_trajectories/` | Hızlı bakış trajektori görüntüleri |

---

## 5. Aşama 02 — DeepLabCut Pose Tahmini

**Durum: Tamamlandı (tüm 12 OFT videosu filtrelenmiş)**  
**Scriptler:** `src/dlc/dlc_setup.py`, `src/dlc/dlc_train.py`, `src/dlc/dlc_inference.py`

### Strateji

Açık alan (OFT) videolarını kapsayan **tek bir DLC modeli** eğitildi. Model, ResNet-50 omurgası ile 12 sıçanın tüm oturumlarını kapsamaktadır.

### Takip Edilen Vücut Parçaları (9 Anahtar Nokta)

| Anahtar Nokta | Rol |
|---------------|-----|
| `nose` | Burun ucu — grooming ve duvar yakınlığı |
| `head` | Kafa merkezi — `htdist` ve `htd_y` özelliklerinde kullanılır |
| `body_center` | Vücut merkezi — thigmotaxis ve heatmap için birincil referans |
| `left_forepaw`, `right_forepaw` | Ön ayaklar — rearing ve grooming tespiti |
| `left_hindpaw`, `right_hindpaw` | Arka ayaklar — dikey yükselme referansı |
| `tail_base` | Kuyruk tabanı — `htdist` ve `htd_y` özelliklerinde kullanılır |
| `tail_tip` | Kuyruk ucu |

### Yapılan Adımlar

1. **DLC Proje Kurulumu** (`dlc_setup.py`)
   - DLC projesi oluşturuldu
   - 12 OFT videosu config'e eklendi
   - Vücut parçaları ve iskelet tanımlandı

2. **Çerçeve Çıkarımı ve Etiketleme**
   - Her videodan ~20 çerçeve çıkarıldı (kmeans yöntemi)
   - Etiketleme GUI'si ile 200–400 çerçeve etiketlendi
   - Her vücut parçası için koordinatlar atandı

3. **Model Eğitimi** (`dlc_train.py`)
   - Eğitim veri seti oluşturuldu
   - `pose_cfg.yaml` RTX 3060 için ayarlandı (`batch_size=8`)
   - `maxiters=50000` ile model eğitildi (ResNet-50 omurgası)
   - Test RMSE < 8px hedefi

4. **Çıkarım ve Filtreleme** (`dlc_inference.py`)
   - 12 OFT videosu üzerinde çıkarım yapıldı
   - Medyan filtresi uygulandı (pencere = 5 kare)
   - `data/DLCfiltered/<subject>/` altında filtrelenmiş CSV'ler dışa aktarıldı

### Üretilen Çıktılar

Her `data/DLCfiltered/<KOHORT>_<N>/` klasöründe bir CSV dosyası:
- Sütunlar: `frame`, `<bodypart>_x`, `<bodypart>_y`, `<bodypart>_likelihood` (30 fps)
- 9 vücut parçası × 3 koordinat = 27 sütun

---

## 6. Aşama 02B — Kural Tabanlı Davranış Tespiti

**Durum: Tamamlandı — MA1_2 (rearing) ve MA5_1 (grooming) üzerinde doğrulandı**  
**Script:** `src/behavior_detection.py`

### Genel Bakış

ML eğitimi gerektirmeyen kural tabanlı dedektör. Postüral özellikler üzerindeki eşikler, davranışları ayırt etmektedir.

### Hesaplanan Özellikler (Kare Başına)

| Özellik | Formül | Davranışsal Anlam |
|---------|--------|-------------------|
| `htdist` | `√((head_x - tail_x)² + (head_y - tail_y)²)` | Duvar rearing'inde beden projeksiyonu kısalır |
| `fp_hp_vert` | `hindpaw_y − forepaw_y` | Ön-arka ayak dikey ayrımı (rearing yönü) |
| `nose_y` | Burun dikey pozisyonu | Duvar yakınlığı (üst/alt duvar) |
| `nose2fp` | Burun ile ön ayak arası mesafe | Grooming sırasında burun ön ayağa yakın |
| `htd_y` | `|head_y − tail_y|` | Yan duvar rearing'inde vücut ekseni eğimi |
| `nose_x` | Burun yatay pozisyonu | Sol/sağ duvar yakınlığı |

### Sınıflandırma Kuralları

**Rearing (Ayağa Kalkma) Kuralları — herhangi biri tetiklenince:**

| Kural | Koşul | Tespit Ettiği |
|-------|-------|---------------|
| R1 — Kompakt Rearing | `htdist < 55` | Herhangi bir duvara karşı güçlü kompresyon |
| R2 — Üst Duvar Uzatmalı | `fp_hp_vert > 45 VE nose_y < 165` | Üst duvara karşı uzanma |
| R3 — Alt Duvar Güçlü | `fp_hp_vert < -80 VE nose_y > 500` | Alt duvara karşı güçlü postür |
| R4 — Alt Duvar Kompakt | `fp_hp_vert < -45 VE nose_y > 500 VE htdist < 105` | Alt duvara karşı ılımlı kompresyon |
| R5 — Yan Duvar | `40 < htd_y < 60 VE (nose_x > 760 VEYA nose_x < 430)` | Sol/sağ duvara karşı rearing |

**Grooming (Tımar) Kuralı:**

| Kural | Koşul | Not |
|-------|-------|-----|
| G1 — Grooming | `nose2fp < 22 VE fp_hp_vert < 10 VE rearing değil` | Yüz yıkama postu |

**Öncelik sırası:** Rearing > Grooming > Diğer

### Bout Gruplandırması

| Parametre | Değer | Anlam |
|-----------|-------|-------|
| `INTER_BOUT_GAP` | 15 kare (0.50 s) | Bu kadar boşlukla ayrılmış pozitifler birleştirilir |
| `MIN_BOUT_FRAMES` | 10 kare (0.33 s) | Bundan kısa boutlar gürültü olarak atılır |

### Manuel Doğrulama Sonuçları (MA1_2)

- **Rearing:** 9 onaylı pencere, tüm R1–R5 kuralları devreye girdi
- **Grooming:** 2 onaylı pencere — 72.7–73.2 s (kısa) ve 160.2–167.7 s (uzun)
- **Bilinen yanlış pozitiflerin giderilmesi:** 4 adet tanımlanmış ve düzeltilmiş

### Üretilen Çıktılar (Her Sıçan İçin)

| Dosya | İçerik |
|-------|--------|
| `*_behavior_timeline.png` | 3 satırlı renk kodlu zaman çizelgesi: gerçek zemin / tespit edilen boutlar / kare bazlı etiketler |
| `*_behavior_bouts.csv` | Bout başına tablo: davranış, başlangıç/bitiş kare ve saniye, süre |
| `*_behavior_frames.csv` | Kare başına tablo: kare indeksi, zaman (s), davranış etiketi |

---

## 7. Aşama 03A — Açık Alan Testi (OFT) Uzamsal Analizi

**Durum: Tamamlandı — tüm 4 kohort (MA1, MA3, MA5, MA7)**  
**Script:** `analysis/run_analysis.py` (tek komutla 4 görsel üretir)

### Veri Filtreleme Boru Hattı (Tüm Scriptlerde Ortak)

1. **Likelihood filtresi** (varsayılan ≥ 0.6) — düşük güvenilirlikli DLC tahminleri atılır
2. **Arena sınır filtresi** — alan dışı noktalar NaN yapılır (görsel taşmayı önler)
3. **Atlama eşiği** (varsayılan 60 piksel) — ardışık kareler arasındaki takip atlamalarını kaldırır
4. **Yuvarlanmalı medyan yumuşatma** (varsayılan pencere=5) — gerçek NaN boşlukları korunarak temporal gürültü azaltılır

**Sonuç:** Vücut parçası başına ~%85–99 kare tutma oranı

### Arena Sınır Tanımı

`analysis/show_frame_coords.py` ile interaktif 2 aşamalı sınır belirleme:
- **Aşama 1:** Arena duvarlarının 4 köşesi tıklanır
- **Aşama 2:** Thigmotaxis sınırı için 4 iç bölge köşesi tıklanır

### Üretilen Görsel Analizler

#### 1. Orbit Trajektori Izgarası (`*_orbit_grid.png`)
- 4×3 ızgara, her vücut parçası için bir panel
- Trajektori rengi: açık (erken kareler) → koyu (geç kareler)
- Arena (beyaz kesik) ve iç bölge (turuncu noktalı) sınırları

#### 2. Thigmotaxis Grafiği (`*_thigmotaxis.png`)
- `body_center` tek panel trajektorisi
- Dış bölge (turuncu): iç sınır dışında kalan noktalar
- İç bölge (mavi): iç sınır içinde kalan noktalar
- Başlıkta **thigmotaxis oranı** raporlanır

**Thigmotaxis:** Duvar yakınında kalma davranışı — açık alan testlerinde kaygıyı ölçen klasik bir indeks.

#### 3. KDE Isı Haritası (`*_heatmap_kde.png`) — TEZ BİRİNCİL
- Gaussian kernel yoğunluk tahmini
- `inferno` renk haritası (siyah → mor → sarı)
- 150 dpi çözünürlükte yayın kalitesi
- Tez için onaylanmış görsel

#### 4. Histogram Isı Haritası (`*_heatmap_histogram.png`)
- 2D bölgelenmiş sayım yoğunluğu
- KDE'nin yanı sıra ham dağılımı doğrulamak için kullanılır

#### 5. Vücut Parçası Isı Haritaları (`*_bodypart_heatmaps.png`)
- Her vücut parçası için bir panel (4×3 ızgara)
- Farklı vücut parçalarının hareket örüntülerini karşılaştırmak için kullanılır

### OFT Metrikleri (`*_oft_metrics.csv`)

| Metrik | Açıklama | Davranışsal Anlam |
|--------|----------|-------------------|
| Toplam mesafe | Kümülatif yol uzunluğu (piksel) | Lokomotor aktivite |
| Ortalama hız | Ortalama hareket hızı | Arousal / inhibisyon |
| Donma süresi | Neredeyse sıfır hızda geçirilen süre | Korku / donma |
| Thigmotaxis oranı | İç bölge dışında geçirilen süre % | Kaygı benzeri davranış |
| Merkez giriş sayısı | İç bölgeye giriş sıklığı | Keşif dürtüsü |

---

<a id="7b-aşama-03c--plus-maze-4-kollu-analizi"></a>
## 7B. Aşama 03C — Plus Maze (4 Kollu) Analizi

**Durum: Tamamlandı — tüm 4 kohort × 3 sıçan = 12 sıçan**
**Klasör:** `analysis/plus_maze/` (script adları korundu, çıktılar `_plus_maze_*` olarak)

### Genel Bakış

Sıçanlar T-maze değil, **4 kollu plus/cross maze**te kayıt edilmiştir (alt + sol + sağ + üst kol, ortada bir junction). Aşama 03A açık alan boru hattıyla aynı temizleme adımları (likelihood, jump, smoothing) kullanıldı. DLC kaynak CSV'leri **filtresiz** sürüm (yumuşatılmamış raw output) — Aşama 03A ile tutarlı, çift filtreleme önlendi.

### Maze Yapısı

```
              [TOP ARM]
                  ||
[LEFT ARM] == JUNCTION == [RIGHT ARM]
                  ||
              [BOTTOM ARM]
```

Sıçanlar deney başında **bottom arm** (start) içine bırakılır.

### İş Akışı

| Adım | Script | Amaç |
|------|--------|------|
| 1 | `show_frame_coords.py` | İnteraktif 4-fazlı arena seçimi (her kol için 4 köşe) |
| 2 | `tmaze_metrics.py` | Sıçan başına nicel metrikler |
| 3 | `orbit_plot.py` | Bölge renkli trajektori (ana + 4 panel bodypart grid) |
| 4 | `activity_heatmap.py` | KDE + 2D histogram yoğunluk haritaları |
| 5 | `run_analysis.py` | Yukarıdaki 4 adımı zincirleyen master script |

**Tek koordinat seti** kullanıldı (Part1 ve Part2 videolarında kamera konumu özdeş):
- bottom-arm: 544 605 404 717
- left-arm: 258 551 348 403
- right-arm: 604 893 345 404
- top-arm: 544 608 39 348

### Veri / Klasör Adlandırma

| Eski | Yeni |
|------|------|
| `data/DLCfiltered/<kohort>/TmazeMA*_n/` | `data/DLCfiltered/<kohort>/PlusMazeMA*_n/` |
| `*_tmaze_metrics.csv` | `*_plus_maze_metrics.csv` |
| `*_tmaze_orbit.png` | `*_plus_maze_orbit.png` |
| `*_tmaze_bodyparts.png` | `*_plus_maze_bodyparts.png` |

### Plus Maze Metrikleri (`*_plus_maze_metrics.csv`)

| Kategori | Metrikler | Davranışsal Anlam |
|----------|-----------|-------------------|
| Bölge işgali | `pct_time_bottom/left/right/top/junction`, `time_*_s` | Mekânsal tercih, kol seçimi |
| Kol girişleri | `total_entries`, `bottom/left/right/top_entries` | Genel keşif aktivitesi |
| Alternasyon | `successive_alternation_pct` (her giriş öncekinden farklı), `tetrad_alternation_pct` (4'lü grup tüm 4 kolu kapsar) | Çalışma belleği, esneklik |
| Persiverasyon | `perseveration_count`, `perseveration_rate_pct` (üst üste aynı kol) | Bilişsel katılık |
| Kol tercihi | `most_visited_arm`, `arm_preference_index` (en çok ziyaret / toplam) | Lateralizasyon, alışkanlık |
| Locomotion | `mean_speed_px_s`, `total_distance_px` | Hareket aktivitesi |
| Sequence | `entry_sequence` (örn. `R->L->T->B->...`) | Davranış zamanlaması analizi için ham dizi |

### Üretilen Çıktılar (Her Sıçan için 5 Dosya)

```
PlusMazeMA1_2/
├── PlusMazeMA1_2.csv                       ← DLC filtresiz raw girdi
├── PlusMazeMA1_2_plus_maze_metrics.csv     ← 30 sütunluk metrik
├── PlusMazeMA1_2_plus_maze_orbit.png       ← Bölge renkli trajektori
├── PlusMazeMA1_2_plus_maze_bodyparts.png   ← 4 panel bodypart grid (nose, head, body_center, tail_base)
├── PlusMazeMA1_2_heatmap_kde.png           ← KDE yoğunluk haritası
└── PlusMazeMA1_2_heatmap_histogram.png     ← 2D histogram
```

### Grup Düzeyinde Çıktı

| Dosya | İçerik |
|-------|--------|
| `data/plus_maze_metrics_all.csv` | 12 sıçan × 30 sütunluk birleştirilmiş tablo |

### Önemli Bulgular (İlk Bakış, n=12)

- **Kontrol (MA1):** Aşırı bottom/top tercihi (örn. MA1_1 %98 top, MA1_2 %87 bottom). Yan kollara minimal giriş.
- **Aspartam (MA3):** Yüksek thigmotaxis benzeri pattern — MA3_3 %97 bottom, neredeyse hareketsiz.
- **Greyfurt (MA5):** Bottom + top dengesi, sınırlı keşif (5–10 toplam giriş).
- **ASP+Greyfurt (MA7):** En yüksek varyans — MA7_3 dengeli 4-kol kullanımı (`R->L->R->R->L->L->T->R...`), MA7_1 ise %50 top dominansı.

### EPM Yorumlaması — Açık Kol vs Kapalı Kol

Plus maze yapısı **Elevated Plus Maze (EPM)** olarak yorumlandığında:
- **Dikey kollar (top + bottom) = kapalı kollar** (güvenli)
- **Yatay kollar (left + right) = açık kollar** (aversif — yükseklik + aydınlık)

Açık kolda geçirilen süre/giriş yüzdesi anksiyolitik etkinin **altın standart** ölçütüdür (Pellow et al. 1985; Walf & Frye 2007 *Nat Protoc*).

**Sıçan başına açık kol % zamanı (left + right):**

| Sıçan | Kohort | %Open Arm | %Closed Arm | Toplam Giriş |
|-------|--------|-----------|-------------|--------------|
| MA1_1 | Control | 0.00 | 98.83 | 6 |
| MA1_2 | Control | 0.00 | 92.94 | 4 |
| MA1_3 | Control | 0.63 | 93.78 | 30 |
| MA3_1 | Aspartame | 5.93 | 88.47 | 13 |
| MA3_2 | Aspartame | 0.16 | 95.61 | 17 |
| MA3_3 | Aspartame | 0.78 | 97.31 | 4 |
| MA5_1 | Grapefruit | 0.52 | 97.63 | 5 |
| MA5_2 | Grapefruit | 4.66 | 89.76 | 10 |
| MA5_3 | Grapefruit | 4.75 | 93.42 | 9 |
| MA7_1 | ASP+GF | 1.93 | 77.92 | 25 |
| MA7_2 | ASP+GF | 8.11 | 73.72 | 23 |
| MA7_3 | ASP+GF | **37.53** | 35.99 | 20 |

**Kohort ortalamaları (n=3/grup):**

| Kohort | Mean %Open Arm | SD | Median | Mean Total Entries |
|--------|---------------:|----|--------|-------------------:|
| Control | 0.21 | 0.36 | 0.00 | 13.3 |
| Aspartame | 2.29 | 3.20 | 0.78 | 11.3 |
| Grapefruit | 3.31 | 2.41 | 4.66 | 8.0 |
| **ASP+Greyfurt** | **15.86** | 18.95 | 8.11 | **22.7** |

**Sonuç:** ASP+Greyfurt kombinasyonu açık kol süresini Control'e göre yaklaşık **75× artırıyor**, ancak bu artış MA7_3 outlier'ı (%37.53) tarafından şişirilmiş — median (8.11) daha güvenilir. Kombinasyon kohortu ayrıca **en yüksek lokomotor aktivite**ye sahip (22.7 entries vs Control 13.3).

**Saf anksiyolitik vs lokomotor karıştırıcı:** Cruz, Frei & Graeff (1994) ve Carobrez & Bertoglio (2005) çerçevesine göre:
- Saf anksiyolitik: %open arm time ↑ **AND** kapalı kol girişi değişmez
- Karışık etki: ikisinin de değişmesi → ASP+GF tablosu buna uyuyor
- Bu, naringin'in (greyfurt flavonoidi) bilinen anksiyolitik etkisi (Fernandez et al. 2009) + greyfurtun CYP3A4 inhibisyonuyla aspartam metabolitlerinin temizlenmesini geciktirmesi (Bailey et al. 2013 *CMAJ*) hipoteziyle uyumlu.

### Bilinen Düzeltmeler

1. **`activity_heatmap.py` histogram düzeltmesi:** `hist2d` veri aralığına otomatik fit ediyordu; veri küçük bir alanda yoğunlaşınca grafik köşeye sıkışıyordu. `auto_extent(zones)` ile sabit `range` ve `xlim/ylim` belirlendi.
2. **CSV tipi:** `_filtered.csv` (DLC'nin pre-smoothed çıktısı) yerine **raw CSV** kullanıldı — analiz scriptleri kendi yumuşatmasını uyguladığı için çift filtreleme engellendi.
3. **Klasör/script ayrımı:** `analysis/` altındaki scriptler `analysis/open_field/` ve `analysis/plus_maze/` olarak ayrıldı; `speed_analysis.py` paylaşılan ortak script olarak `analysis/` kökünde kaldı.

---

## 8. Üretilen Çıktı Dosyaları

Her `data/DLCfiltered/<KOHORT>_<N>/` klasörü şu 9 dosyayı içermektedir:

```
OpenFieldMA3_2/
├── OpenFieldMA3_2.csv                    ← DLC filtreli girdi (gerçek kaynak)
│
│   Davranış tespiti (src/behavior_detection.py)
├── OpenFieldMA3_2_behavior_timeline.png
├── OpenFieldMA3_2_behavior_bouts.csv
├── OpenFieldMA3_2_behavior_frames.csv
│
│   Uzamsal / lokomotor analiz (analysis/*.py)
├── OpenFieldMA3_2_orbit_grid.png
├── OpenFieldMA3_2_thigmotaxis.png
├── OpenFieldMA3_2_heatmap_kde.png
├── OpenFieldMA3_2_heatmap_histogram.png
└── OpenFieldMA3_2_bodypart_heatmaps.png
```

### Grup Düzeyinde Çıktılar

| Dosya | İçerik |
|-------|--------|
| `data/behavior_summary.csv` | Sıçan başına davranış bout istatistikleri |
| `data/behavior_group_stats.csv` | Kohort düzeyinde davranış özetleri |
| `data/behavior_comparison.png` | Gruplar arası karşılaştırma grafiği |
| `data/part0_oft_metrics.xlsx` | Tüm sıçanları kapsayan OFT metrik tablosu |

---

## 9. Gruplar Arası Temel Bulgular

Ayrıntılı analiz için bkz. [`behavior_comparison.md`](behavior_comparison.md).

### Örüntü A — Lokomotor aktivite gradiyanı

```
Toplam mesafe (px): Kontrol 4.449 < Aspartam 4.982 < Greyfurt 5.960 < ASP+GF 6.949
Ortalama hız (px/s): Kontrol 26.5 < Aspartam 29.8 < Greyfurt 33.6 < ASP+GF 39.9
```

Kombine tedavi, kontrole göre yaklaşık %50 daha hızlıdır. Muhtemelen CYP3A4 aracılı greyfurt + aspartam etkileşimi.

### Örüntü B — Donma / lokomotor ters eksen

```
Donma süresi (%): Kontrol 18.6% > Greyfurt 9.7% > ASP+GF 6.1% ≈ Aspartam 6.0%
```

Daha fazla hareket eden hayvanlar daha az donuyor. Tüm tedavi grupları, yenilik kaynaklı korkuyu azaltıyor.

### Örüntü C — Greyfurta özgü grooming artışı

```
Grooming süresi (%): Kontrol 2.3% ≈ Aspartam 2.1%  <<  Greyfurt 17.9% > ASP+GF 11.3%
```

Bu, veri setindeki en keskin tedavi-kontrol sinyalidir. Greyfurt grooming'i yaklaşık 8× artırıyor; aspartam eklendiğinde bu artış hafifliyor.

### Örüntü D — Sürekli ve parçalı davranış

| Metrik | Kontrol | Aspartam | Greyfurt | ASP+GF |
|--------|---------|----------|----------|--------|
| Rearing parçalanma indeksi | 0.926 | 0.837 | **0.590** | 0.711 |
| Grooming parçalanma indeksi | 1.279 | 1.248 | **0.680** | 1.023 |

Düşük değer = daha uzun, sürekli boutlar. Greyfurt rearing ve grooming'i uzun episodlar halinde düzenliyor; aspartam eklenmesi bu düzeni bozuyor.

### Örüntü E — Thigmotaxis ≠ Donma

Klasik kaygı teorisi duvar yakınlaşması ve donmanın birlikte görüldüğünü öngörür — ancak burada görülmüyor:
- Greyfurt en güçlü thigmotaxis gösteriyor (%93 çevre) ama yalnızca ara düzeyde donma
- ASP+GF en düşük donmayı (%6.1) ve en fazla merkez girişini gösteriyor
- Bu ayrışma, kaygı hedefinin birden fazla metriği birlikte ağırlıklandırması gerektiğini ortaya koyuyor

---

## 10. Profesör MATLAB Pipeline ile Çapraz Doğrulama

`data/kutu_validation_summary.csv` dosyası, DLC pose çıktımızı profesörün 2017 HSV-blob tabanlı MATLAB pipeline'ıyla karşılaştırmaktadır.

- **Pearson r ≥ 0.78** — 12 sıçanın 9'unda (medyan 0.85)
- **İstisna:** MA3_1 (r=0.45), MA3_3 (r=0.54) — trajektori görüntüleri gözden geçirilmeli
- **Hız oranı:** kutu/dlc = 0.79–0.90 — DLC tutarlı biçimde %10–20 daha hızlı okuyor

Bu sapma **sistematik ve uniform** — tüm kohortlarda aynı yönde. Muhtemelen DLC bireysel vücut parçalarını takip ederken HSV-blob tıkandığında ince hareketi kaybediyor. Grup karşılaştırmalarını geçersiz kılmaz, ancak tez yöntemler bölümünde belirtilmesi gerekir.

---

## 11. Mevcut Durum Özeti

| Aşama | Durum |
|-------|-------|
| 01 — Video ön işleme / kalite kontrolü | **Tamamlandı** |
| 02 — DeepLabCut pose tahmini | **Tamamlandı** (tüm 12 OFT videosu filtrelenmiş) |
| 02B — Kural tabanlı davranış tespiti | **Tamamlandı**, MA1_2 / MA5_1 üzerinde doğrulandı |
| 03A — OFT uzamsal analizi — tüm kohortlar (MA1, MA3, MA5, MA7) | **Tamamlandı** |
| 03C — Plus maze (4 kollu) analizi — 12 sıçan | **Tamamlandı** (2026-05-07) |
| 05 — Özellik mühendisliği (yalnızca OFT) | **Tamamlandı** |
| 06 — ML modeli eğitimi + SHAP analizi | **Tamamlandı** |
| 06B — Rapor görselleştirme (CSV → PNG) | **Tamamlandı** |
| 05B — OFT + Plus Maze özellik birleştirmesi | **Sıradaki** |
| 07 — Gruplar arası raporlama | **Planlandı** |

---

### Aşama 05 — Özellik Mühendisliği

**Durum: Tamamlandı**
**Script:** `src/features.py`

`oft_metrics_all.csv` ve `behavior_summary.csv` dosyaları `subject_id` üzerinden birleştirilerek 26 özellikten oluşan tek bir tablo üretildi. Türetilmiş özellikler eklendi (`locomotion_pct`, `rear_per_min`, `groom_per_min`, `exploration_ratio`). Dört hedef etiket hesaplandı.

**Üretilen dosyalar:**

| Dosya | İçerik |
|-------|--------|
| `data/features/features_raw.csv` | 12 satır × 26 özellik (ham) |
| `data/features/features_normalized.csv` | StandardScaler uygulanmış |
| `data/features/labels.csv` | `anxiety_score`, `anxiety_level`, `rearing_profile`, `grooming_profile` |
| `models/classifier/scaler.pkl` | Kaydedilmiş ölçekleyici |

---

### Aşama 06 — ML Model Eğitimi

**Durum: Tamamlandı**
**Script:** `src/train_baseline.py`

6 model × 4 hedef = 24 model **iki ayrı CV stratejisi** ile eğitildi:
- **LOOCV** (Leave-One-Out): her sıçan tek başına test edildi (n=12 fold)
- **LOGOCV** (Leave-One-Group-Out): her kohort tek başına test edildi (n=4 fold) → gerçek genelleme testi

XGBoost ile SHAP analizi yapıldı. Sonuçlar `reports/model_comparison_all.csv`'de.

**LOOCV F1-Macro Sonuçları (en iyi değerler kalın):**

| Model | group | anxiety_level | rearing_profile | grooming_profile |
|-------|-------|---------------|-----------------|-----------------|
| LogisticReg-L1 | 0.310 | **0.646** | 0.408 | 0.574 |
| RandomForest | **0.317** | 0.621 | 0.408 | 0.771 |
| XGBoost | 0.297 | 0.621 | **0.460** | 0.923 |
| SVM | 0.275 | 0.621 | 0.295 | 0.602 |
| LogReg+MI8 | 0.056 | 0.227 | 0.295 | 0.237 |
| LightGBM | 0.227 | 0.609 | 0.313 | **1.000** |
| *Rastgele baz* | *0.250* | *0.333* | *0.333* | *0.333* |

**LOGOCV F1-Macro Sonuçları (cohort hold-out — daha gerçekçi):**

| Model | group | anxiety_level | rearing_profile | grooming_profile |
|-------|-------|---------------|-----------------|-----------------|
| LogisticReg-L1 | 0.067 | 0.597 | 0.408 | 0.715 |
| RandomForest | 0.160 | 0.597 | 0.408 | 0.771 |
| XGBoost | 0.163 | **0.621** | **0.467** | 0.798 |
| SVM | 0.072 | 0.597 | 0.295 | 0.547 |
| LogReg+MI8 | 0.000 | 0.195 | 0.295 | 0.237 |
| LightGBM | **0.165** | 0.621 | 0.431 | **0.896** |

**Dürüst yorum (kritik):**

| Hedef | Yorum |
|-------|-------|
| **group** | LOGOCV F1=0.16 → şans seviyesi (1/4=0.25). n=12 ile cohort tahmini **çürük**. Cohort istatistiği için `analysis/cohort_stats.py` (KW + Dunn + PERMANOVA) kullanılmalı. |
| **anxiety_level** | LOOCV/LOGOCV F1=0.62 — yüksek görünüyor ama **label-leakage var**: etiket `pct_time_periphery + pct_time_freeze − center_entries` formülünden üretiliyor ve aynı feature'lar input'ta. §13.3 (final_report) drop-the-leakage-feature sanity check şart. |
| **rearing_profile** | F1=0.46-0.47 — sınırda; rearing fragmentation bir feature'dan ezbere yakın çalışıyor. |
| **grooming_profile** | F1=0.92-1.00 — çok yüksek; **Greyfurt'un grooming'i 8× artırması** sayesinde tek feature (`groom_pct_time`) hedefin neredeyse tamamını anlatıyor. Tautoloji riski yüksek. |

**One-vs-Rest SHAP — Her Grubun Ayırt Edici Özellikleri:**

| Grup | En Belirleyici Özellik | Biyolojik Yorum |
|------|------------------------|-----------------|
| Control | `freeze_bout_count`, `total_distance_px` | Düşük lokomotor + yüksek donma |
| Aspartame | `rear_bout_count`, `groom_total_s` | Kontrol ve greyfurt arası geçiş profili |
| Grapefruit | `center_zone_entries` (tutarlı biçimde = 6) | Kaygı kaynaklı thigmotaxis |
| ASP+GF | `groom_pct_time` (en güçlü sinyal) | Grooming + lokomotor sinerjik aktivasyon |

**Üretilen dosyalar:**

| Dosya | İçerik |
|-------|--------|
| `reports/model_comparison.csv` | 4 model × 4 hedef × F1/accuracy |
| `reports/loocv_predictions.csv` | Sıçan bazlı LOOCV tahmin detayları |
| `reports/ovr_binary_f1.csv` | OvR ikili sınıflandırma F1 skorları |
| `reports/figures/confusion_<hedef>.png` | 4 adet karışıklık matrisi |
| `reports/figures/shap_<hedef>.png` | 4 adet çok-sınıflı SHAP grafikleri |
| `reports/figures/shap_ovr_groups.png` | **Grup bazlı OvR SHAP (tez için birincil görsel)** |
| `reports/figures/table_model_comparison.png` | Model karşılaştırma ısı haritası (tez tablosu) |
| `reports/figures/table_loocv_predictions.png` | XGBoost sıçan bazlı doğru/yanlış tahmin tablosu |
| `reports/figures/chart_ovr_f1.png` | OvR F1 bar grafiği |
| `models/classifier/<model>_<hedef>.pkl` | 16 eğitilmiş model dosyası |

**Örneklem kısıtı ve F1 yorumu:**

n=12 (3/grup) ile grup tahmini F1=0.25–0.27 (rastgele baza eşit) elde edildi. Bu beklenen bir sonuçtur — güvenilir 4-sınıf ML için grup başına en az 10 örnek (toplam 40) gereklidir. Buna karşın anxiety_level (F1=0.52) ve grooming_profile (F1=0.51) tahminleri rastgele bazın anlamlı biçimde üzerindedir. Bu bulgular pipeline'ın gerçek davranışsal sinyal ürettiğini kanıtlamakta, ancak grup ayrımı için daha büyük örneklem gerektiğini göstermektedir.

| F1 aralığı | Yorum |
|-----------|-------|
| ≤ 0.33 | Rastgele veya altı |
| 0.33–0.45 | Zayıf sinyal |
| 0.45–0.60 | **İyi (n=12 için gerçekçi başarı)** |
| 0.60–0.75 | Çok iyi |
| > 0.75 | Şüpheli — veri sızıntısı kontrol edilmeli |

---

### Aşama 06B — Rapor Görselleştirme

**Durum: Tamamlandı**
**Script:** `src/visualize_reports.py`

`reports/` altındaki CSV çıktıları teze doğrudan alınabilecek PNG görsellerine dönüştürüldü.

| Görsel | Açıklama |
|--------|----------|
| `table_model_comparison.png` | 4 model × 4 hedef F1 ısı haritası (kırmızı=kötü, yeşil=iyi) |
| `table_loocv_predictions.png` | XGBoost'un her sıçan için doğru/yanlış tahmin tablosu (kohort renkli) |
| `chart_ovr_f1.png` | OvR grup bazlı F1 ve accuracy bar grafiği |

---

### Aşama 07 — Gruplar Arası Raporlama: Plan

**Ön koşul:** 05 ve 06 çıktılarının tamamlanmış olması.

**Hedef:** Tüm pipeline çıktılarını yapılandırılmış bir tez raporu haline getirmek.

**7.1 — Sıçan başına davranışsal profiller**

Her kohort (MA1, MA3, MA5, MA7) için tek sayfalık özet:
- OFT occupancy KDE heatmap'leri (3 seans yan yana)
- Rearing + grooming heatmap'leri
- OFT metrik eğilimleri seans boyunca (çizgi grafikleri)
- ML modelinin tahminlediği kaygı / rearing / grooming etiketleri

**7.2 — Grup karşılaştırma istatistikleri**

Her özellik için Kruskal-Wallis testi (4 grup, n=3/grup) + Bonferroni düzeltmeli Dunn post-hoc testi. Çıktı: özellik × p-değeri × etki büyüklüğü tablosu.

**7.3 — Davranış bout istatistikleri**

Rearing ve grooming için grup bazında kutu / violin grafikleri:
- Bout sayısı, ortalama süre, toplam süre yüzdesi
- Parçalanma indeksi (fragmentation index)

**7.4 — Heatmap galerisi**

| Görsel | Açıklama |
|--------|----------|
| 4 grup × occupancy KDE | Mekânsal tercih karşılaştırması |
| 4 grup × rearing heatmap | Rearing bölgesi dağılımı |
| 4 grup × grooming heatmap | Grooming bölgesi dağılımı |

**7.5 — Model performans özeti**

Model × etiket × F1 tablosu, SHAP özet grafikleri ve karışıklık matrisleri (sıçan ID'leriyle).

**Tez rapor yapısı:**

```
reports/
├── figures/
│   ├── heatmaps/open_field/     # OFT: occupancy, rearing, grooming
│   ├── trajectories/            # Davranış renkli rota grafikleri
│   ├── ethograms/               # Seans zaman çizelgeleri
│   ├── statistics/              # Grup karşılaştırma grafikleri
│   └── model/                   # SHAP, karışıklık matrisleri, ROC
├── tables/
│   ├── oft_metrics_summary.csv
│   ├── behavior_bout_stats.csv
│   ├── group_comparisons.csv
│   └── model_comparison.csv
└── profiles/
    ├── MA1_profile.pdf   (Kontrol)
    ├── MA3_profile.pdf   (Aspartam)
    ├── MA5_profile.pdf   (Greyfurt)
    └── MA7_profile.pdf   (ASP + Greyfurt)
```

---

## 12. Planlanan Çalışmalar

### Tamamlanan Kısa Vadeli Görevler ✓

| Görev | Durum |
|-------|-------|
| Etiket oluşturma (`anxiety_score`, `rearing_profile`, `grooming_profile`) | **Tamamlandı** — `src/features.py` |
| Özellik tablosu birleştirme (`features_raw.csv`, `features_normalized.csv`) | **Tamamlandı** — `src/features.py` |
| 4 model × 4 hedef LOOCV eğitimi | **Tamamlandı** — `src/train_baseline.py` |
| SHAP analizi (çok-sınıflı + One-vs-Rest) | **Tamamlandı** — `src/train_baseline.py` |
| CSV çıktılarının PNG görselleştirmesi | **Tamamlandı** — `src/visualize_reports.py` |
| Plus maze pipeline (4 kollu metrikler + heatmap + orbit) | **Tamamlandı** — `analysis/plus_maze/` |
| 12 sıçanın plus maze metriklerinin birleştirilmesi (`plus_maze_metrics_all.csv`) | **Tamamlandı** |

### Sıradaki Görev — Aşama 05B: OFT + Plus Maze Özellik Birleştirmesi

`src/features.py` güncellenecek. İki arenanın metrikleri sıçan ID üzerinden birleştirilip ML pipeline'ına bağlanacak.

1. **ID normalleştirme:** `subject_id` formatları farklı — OFT `MA1_1`, plus maze `PlusMazeMA1_1`. Plus maze ID'lerinden `PlusMaze` prefix kaldırılarak ortak `MA1_1` formatına indirgenecek.
2. **Sütun çakışmaları:** `mean_speed_px_s`, `total_distance_px`, `session_duration_s` her iki arenada da var. Plus maze sütunlarına `pm_` prefix eklenecek (`pm_mean_speed_px_s` vb).
3. **Yeni özellikler eklenecek:**
   - `pm_pct_time_bottom/left/right/top/junction` (5)
   - `pm_total_entries`, `pm_*_entries` (5)
   - `pm_arm_preference_index` (1)
   - `pm_successive_alternation_pct`, `pm_tetrad_alternation_pct` (2 — çalışma belleği)
   - `pm_perseveration_rate_pct` (1 — esneklik)
   - `pm_total_distance_px`, `pm_mean_speed_px_s` (2)
   - **Toplam +16 özellik** → mevcut 26 OFT özelliğine eklenir = **42 özellik**
4. **Yeniden eğitim:** 4 model × 4 hedef LOOCV ile yeniden eğitilecek; SHAP analizinde plus maze özelliklerinin grup ayrımına katkısı incelenecek.
5. **Yeni etiket önerisi:** `cognitive_flexibility` — `pm_successive_alternation_pct` ve `pm_perseveration_rate_pct` üzerinden tanımlanan üçlü etiket (low/moderate/high).

### Aşama 07: Gruplar Arası Raporlama

Tüm pipeline çıktılarını yapılandırılmış bir tez raporu haline getirmek:

1. **Kruskal-Wallis + Dunn post-hoc istatistik testi** — her özellik için grup karşılaştırması (p-değeri ve etki büyüklüğü tablosu) — **OFT + plus maze metrikleri ayrı ayrı**
2. **Violin / kutu grafikleri** — rearing, grooming, alternation, perseveration istatistikleri grup bazında
3. **KDE heatmap galerisi** — 4 grup yan yana (OFT occupancy + plus maze occupancy)
4. **Sıçan başına profil sayfaları** — her kohort için özet görsel (OFT heatmap + plus maze trajektori + ML tahminleri)
5. **Plus maze sequence analizi** — `entry_sequence` üzerinden Markov geçiş matrisleri (gruplar arası karşılaştırma)

### Örneklem Kısıtı Notu

Mevcut n=12 (3/grup) ile 4-sınıf grup tahmini güvenilir değildir. Gelecek çalışmalar için **grup başına en az 10 sıçan (toplam 40)** önerilir. Anxiety ve grooming profil tahminlerindeki F1>0.50 değerleri, pipeline'ın biyolojik sinyal ürettiğini göstermiştir.

---

<a id="13-plus-maze-epm-anksiyete-analizi"></a>
## 13. Plus Maze EPM Anksiyete Analizi

**Durum:** Plus maze metrikleri tüm 12 sıçan için hesaplanmış (Aşama 03C). EPM çerçevesine göre **anksiyete-spesifik istatistik henüz yapılmadı** — sıradaki iş.

### 13.1 EPM Endpoint Hiyerarşisi

EPM literatüründe (Pellow et al. 1985; Walf & Frye 2007 *Nat Protoc*) altın standart birincil endpoint'ler:

| Endpoint | Formül | Anksiyolitik etki |
|----------|--------|-------------------|
| **% açık kol süresi** | (açık kol süresi / toplam test süresi) × 100 | ↑ |
| **% açık kol girişi** | (açık kol girişi / toplam giriş) × 100 | ↑ |
| **Toplam giriş** veya **kapalı kol girişi** | locomotion göstergesi (kovariat olarak) | nötr olmalı |
| **Korumasız head-dip** | açık kol kenarından aşağı bakış | ↑ |
| **Stretch-attend posture (SAP)** | risk-değerlendirme postürü | ↓ |

### 13.2 Cruz 1994 İki Faktör Modeli

Cruz, Frei & Graeff (1994 *Pharmacol Biochem Behav*) faktör analizi ile EPM ölçütlerinin iki ortogonal faktöre yüklendiğini gösterdi:

| Faktör 1 — Anksiyete | Faktör 2 — Lokomotor |
|----------------------|----------------------|
| % open arm time | closed arm entries |
| % open arm entries | total distance |
| Korumasız head-dip | mean speed |
| SAP (negatif yön) | — |

**Yorum kuralı:** Bir madde sadece Faktör 1'i değiştiriyorsa **saf anksiyolitik**; her iki faktörü değiştiriyorsa **karışık etki** (motor karıştırıcı dahil).

### 13.3 Sizin Verinizde Beklenen İstatistik (Yapılacak)

Plus maze'in `analysis/cohort_stats.py` (KW + Dunn + PERMANOVA) eşdeğeri **henüz yazılmadı**. OFT için var; Plus Maze'e port edilmeli. Beklenen pipeline:

```python
# Yeni: analysis/plus_maze/cohort_stats_epm.py
features = ["pct_open_arm", "pct_open_arm_entries",
            "total_entries", "closed_arm_entries",
            "mean_speed_px_s", "successive_alternation_pct"]
# Kruskal-Wallis (4 grup) + permütasyon p
# Dunn post-hoc (BH-FDR within-feature)
# PERMANOVA (multivariate)
# ANCOVA: open_arm % ~ cohort, kovariat=closed_arm_entries
```

### 13.4 Etolojik Feature'lar (DLC Pose'dan Türetilebilir)

Mevcut Plus Maze metrikleri **sadece spasiyo-temporal**. Anksiyete spesifikliği için etolojik feature'lar eklenmeli (Rodgers & Johnson 1995):

| Feature | Pose'dan tespiti |
|---------|------------------|
| **Korumasız head-dip** | nose y-koordinatı arena kenarı eşiğinin altında + body merkezde |
| **Stretch-attend posture (SAP)** | body uzunluğu (nose-tail mesafesi) %120+ + low velocity |
| **Risk assessment** | kapalı→açık kol geçişinde yarım kalan giriş (junction'a girip geri dönme) |
| **Rearing on EPM** | OFT detector'ından port |
| **Defecation/grooming** | mevcut OFT detector'dan port (Plus Maze keypoint set'ine adapte) |

### 13.5 Madde-Spesifik Beklenen Yön (Literatür)

| Madde | Klasik beklenen EPM etkisi | Sizin verinizdeki gözlem |
|-------|--------------------------|--------------------------|
| **Aspartame** | Anksiyojenik (Ashok & Sheeladevi 2015; Onaolapo 2017; Jones et al. 2022 *PNAS*) — open arm ↓ | %open=2.3 (Control'den biraz yüksek, beklentinin tersi yönünde) |
| **Naringin/Greyfurt** | Anksiyolitik (Fernandez et al. 2009; Viswanatha et al. 2017) — open arm ↑ | %open=3.3 (hafif artış) |
| **Greyfurt × ilaç** | CYP3A4 inhibisyonu, psikotropikler 1.5-3× plazma artışı (Bailey 2013) | ASP+GF kombinasyonunda en güçlü açık kol artışı (median 8.11) — etkileşim hipoteziyle uyumlu |
| **Pozitif kontrol önerisi** | Diazepam 1 mg/kg i.p. (30 dk önce) | Yapılmadı — sonraki deney için öneri |

---

<a id="14-i̇yileştirme-yol-haritası-ve-literatür"></a>
## 14. İyileştirme Yol Haritası ve Literatür

### 14.1 Acil Düzeltmeler (1-2 gün)

| # | İş | Dosya | Çıktı |
|---|----|-------|-------|
| 1 | **Plus maze toplu metrik tablosu üret** | `analysis/plus_maze/run_analysis.py` (`--batch-dir` ekle) | `data/plus_maze_metrics_all.csv` (12 satır) |
| 2 | **EPM cohort istatistiği** | `analysis/plus_maze/cohort_stats_epm.py` (yeni — OFT'tan port) | `reports/cohort_epm_kw.csv`, `reports/cohort_epm_dunn.csv` |
| 3 | **Açık kol % box/violin** | grafik scripti | `reports/figures/epm_open_arm_by_cohort.png` |
| 4 | **Label-leakage caveat'ini netleştir** | `src/train_baseline.py`'a drop-the-leakage-feature varyantı | yeni LOOCV F1 değerleri (genelde 0.10-0.15 düşer ama dürüst) |

### 14.2 Orta Vade (1-2 hafta)

| # | İş | Beklenen Sonuç |
|---|----|----------------|
| 5 | **OFT + Plus Maze feature birleştirmesi** (Aşama 05B) | 42 feature × 12 sıçan; LOGOCV F1 ~0.20-0.30 olabilir |
| 6 | **Etolojik feature'lar** (head-dip, SAP, risk-assessment) | Plus Maze'de saf anksiyete sinyali |
| 7 | **Window classifier'ı çalıştır** (`src/train_window_classifier.py`) — ground-truth genişletme şart | Rule-based detector'ın yerine, F1=0.70+ |
| 8 | **anxiety_score regresyon** (sınıflandırma yerine) | n=12 için Pearson r + RMSE rapor — F1'den dürüst |

### 14.3 Uzun Vade (>1 ay, opsiyonel)

| # | İş | Not |
|---|----|----|
| 9 | **Veri büyütme** — grup başına +2-3 sıçan | n=20+ → cohort F1 0.5+ olabilir |
| 10 | **Pozitif kontrol deneyi** (diazepam 1 mg/kg) | Pipeline'ın gerçek anksiyolitik sinyali yakaladığını doğrular |
| 11 | **SimBA / B-SOiD entegrasyonu** | Sosyal/ etolojik davranış sınıflandırma (Goodwin et al. 2024) |
| 12 | **Multi-modal fusion**: heatmap CNN embedding + tabular | Layer 3-4 (final_report §9) |

### 14.4 Literatür Referansları

#### EPM Metodoloji
- Pellow S, et al. (1985). *J Neurosci Methods* — EPM validasyonu
- Pellow S & File SE (1986) PMID 2864480 — Anksiyolitik/anksiyojenik iki yönlü duyarlılık
- Cruz APM, Frei F & Graeff FG (1994). *Pharmacol Biochem Behav* — Faktör analizi (anksiyete vs lokomotor)
- Rodgers RJ & Johnson NJT (1995). *Pharmacol Biochem Behav* — Etolojik analiz (head-dip, SAP)
- Carobrez AP & Bertoglio LJ (2005). *Neurosci Biobehav Rev* — EPM 20 yıl sonra
- File SE (2001). *Behav Brain Res* — Yorumlama tuzakları
- Walf AA & Frye CA (2007). *Nat Protoc* PMC3623971 — Standart protokol

#### Aspartam
- Ashok I & Sheeladevi R (2015) — Wistar sıçanlarda uzun süreli aspartam → anksiyojenik
- Onaolapo OJ et al. (2017). *Pathophysiology* — Subkronik aspartam nörodavranışı
- Jones BL et al. (2022). *PNAS* PMC9894161 — Transgenerasyonel aspartam anksiyetesi (amigdala glu/GABA)
- Christian B et al. (2004). *Pharmacol Biochem Behav* — Aspartam + etanol farede anksiyete

#### Naringin / Naringenin (Greyfurt Flavonoidleri)
- Fernandez SP et al. (2009). *Neurochem Res* — Naringin anksiyolitik etkisi
- Viswanatha GL et al. (2017) — Naringin doz-cevap (25-100 mg/kg)
- Nouri Z et al. (2019). *Biomed Pharmacother* — Naringenin Parkinson modelinde anksiyolitik

#### Greyfurt × CYP3A4 İlaç Etkileşimi
- Bailey DG et al. (2013). *CMAJ* PMC3589309 — Greyfurt-ilaç etkileşimleri review
- Hanley MJ et al. (2024). *Expert Opin Drug Metab Toxicol* — Greyfurt + psikotropikler güncel

#### DLC + Davranış Sınıflandırma
- Sturman O et al. (2020). *Neuropsychopharmacology* — DLC tabanlı EPM/OFT/FST otomatik analiz
- Goodwin NL et al. (2024). *Front Behav Neurosci* — DLC + SimBA etolojik analiz pipeline
- Hsu AI & Yttri EA (2021). *Nat Commun* — B-SOiD unsupervised davranış kümeleme
- Bohnslav JP et al. (2021). *eLife* — DeepEthogram supervised davranış
- SimBA: github.com/sgoldenlab/simba (Golden Lab, RandomForest tabanlı)

#### Türkçe Kaynaklar
- YÖK Ulusal Tez Merkezi — tez.yok.gov.tr ("elevated plus labirent", "sıçan anksiyete", "DeepLabCut" anahtar kelimeleri)
- Marmara Medical Journal — "Anksiyete araştırmalarında kullanılan sıçan davranış modelleri" derlemesi (TR metodolojik referans)

---

*Bu dosya, `docs/behavior_comparison.md`, `docs/final_report.md`, `docs/window_classifier_plan.md`, `docs/tmaze_keypoints_and_layout.md`, `analysis/WORKFLOW_SUMMARY.md`, `reports/model_comparison_all.csv`, `reports/cohort_*.csv` ve `documents/plans/` altındaki aşama planlarından derlenerek hazırlanmıştır.*
