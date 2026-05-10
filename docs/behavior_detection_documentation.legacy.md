# Behavior Detection Algorithm Documentation
## `src/behavior_detection.py`

**Proje:** Open-Field Test — Rat Behavior Analysis  
**Model:** DLC_Resnet50_rat_behavior_openfieldApr1shuffle1_snapshot_best-180  
**Kalibrasyon verisi:** `OpenFieldMA1_2.csv` (5068 kare, 30 fps, 168.9 saniye)

---

## 1. Genel Bakış

Script, DeepLabCut (DLC) filtrelenmiş CSV çıktısından kare bazında **rearing** (ayağa kalkma) ve **grooming** (kendini yalama) davranışlarını kurala dayalı (rule-based) olarak tespit eder. Makine öğrenmesi eğitimi gerektirmez; bunun yerine vücudun eklem noktaları arasındaki geometrik ilişkileri eşik değerleriyle karşılaştırır.

### Tespit edilen davranışlar

| Davranış | Türkçe | Tanım |
|----------|--------|-------|
| `rearing` | Ayağa kalkma | Sıçan arka ayakları üzerinde dik durur |
| `grooming` | Kendini yalama | Sıçan ön patilerini yüzüne götürür |
| `other` | Diğer | Yürüme, koklama, hareketsiz bekleme vb. |

---

## 2. Girdi Verisi

### 2.1 CSV Formatı

DLC, üç satırlı başlıklı bir CSV üretir:

```
Row 0 (scorer)    : model adı
Row 1 (bodyparts) : eklem noktası adı
Row 2 (coords)    : x | y | likelihood
Row 3+            : kare verileri (frame 0, 1, 2, ...)
```

### 2.2 Eklem Noktaları (Keypoints)

Script 10 eklem noktasını kullanır:

| Keypoint | Sütun prefix | Kullanım |
|----------|-------------|----------|
| `nose` | `nose_x`, `nose_y` | Burun konumu; duvar yakınlığı ve grooming |
| `head` | `head_x`, `head_y` | Kafa merkezi; htdist hesabı |
| `left_ear` | — | Bu scriptte doğrudan kullanılmaz |
| `right_ear` | — | Bu scriptte doğrudan kullanılmaz |
| `body_center` | — | Bu scriptte doğrudan kullanılmaz |
| `left_forepaw` | `left_forepaw_x/y/likelihood` | Ön pati yüksekliği ve grooming |
| `right_forepaw` | `right_forepaw_x/y/likelihood` | Ön pati yüksekliği ve grooming |
| `left_hindpaw` | `left_hindpaw_y/likelihood` | Arka pati referans yüksekliği |
| `right_hindpaw` | `right_hindpaw_y/likelihood` | Arka pati referans yüksekliği |
| `tail_base` | `tail_base_x`, `tail_base_y` | htdist hesabı |

### 2.3 Koordinat Sistemi

Görüntü koordinat sistemi kullanılır: **y ekseni aşağı doğru artar.**

```
(0,0) ─────────────────────────── x →
  │
  │    ARENA ÜST DUVARI  (küçük y ~ 160 px)
  │
  │    [arena içi: x ≈ 397–775, y ≈ 158–532]
  │
  │    ARENA ALT DUVARI  (büyük y ~ 530 px)
  y ↓
```

Sıçan yukarı kalktığında (rearing, üst duvar) → `nose_y` **küçülür**.  
Sıçan alt duvara kalktığında (rearing, alt duvar) → `nose_y` **büyür**.

---

## 3. Ön İşleme

### 3.1 `load_dlc_csv(path)`

DLC'nin üç satırlı başlığını okur ve sütun adlarını düzleştirir.

```python
df.columns = ["_".join(c[-2:]).strip() for c in df.columns.values]
# Örnek: "nose_x", "nose_y", "nose_likelihood"
```

### 3.2 `mask_low_likelihood(df, thresh=0.6)`

DLC'nin güven skoru (`likelihood`) düşük olan eklem noktalarının koordinatlarını `NaN` ile değiştirir. Bu, hatalı takip tahminlerinin özellik hesaplamalarını bozmasını engeller.

```
likelihood < LIKELIHOOD_THRESH  →  x = NaN, y = NaN
```

**Önemli istisna:** `htdist` özelliği maskelenmemiş (ham) veriyle hesaplanır. Bunun nedeni, sıçan duvara karşı ayağa kalktığında `tail_base` kuyruğun duvarın arkasında kalması ve likelihood'un düşmesidir. Bu durumda maskeleme yapılsaydı `htdist = NaN` olacak ve rearing tespiti başarısız olacaktı.

| Özellik | Kaynak DataFrame |
|---------|-----------------|
| `htdist` | `raw_df` (maskelenmemiş) |
| `fp_hp_vert` | `masked_df` (nanmean ile) |
| `nose_y` | `masked_df` |
| `nose2fp` | `masked_df` |

---

## 4. Özellik Çıkarımı

### 4.1 `compute_features(raw_df, masked_df)`

Her kare için 4 skalar özellik hesaplanır.

---

#### Özellik 1: `htdist` — Kafa-Kuyruk Mesafesi

```
htdist = sqrt( (head_x - tail_base_x)² + (head_y - tail_base_y)² )
```

**Fiziksel anlamı:** Sıçanın 2B yansımasındaki vücut uzunluğu. Normal yürüyüşte vücut yatay uzanır ve bu mesafe büyüktür (~110–145 px). Sıçan duvara yaslanarak ayağa kalktığında vücut kameradan bakan açıdan sıkışır ve mesafe dramatik biçimde düşer (~17–55 px).

| Durum | htdist değer aralığı |
|-------|---------------------|
| Normal yürüyüş | 95 – 155 px |
| Üst/alt duvara karşı rearing (kompakt) | 17 – 55 px |
| Serbest rearing (üst duvar, genişlemiş) | 110 – 125 px |

---

#### Özellik 2: `fp_hp_vert` — Ön Pati / Arka Pati Düşey Farkı

```python
fp_y = mean(left_forepaw_y, right_forepaw_y)   # NaN olanı atla
hp_y = mean(left_hindpaw_y, right_hindpaw_y)   # NaN olanı atla

fp_hp_vert = hp_y - fp_y
```

**Fiziksel anlamı:**  
- `fp_hp_vert > 0`: Ön patiler arka patilerden **yukarıda** (küçük y) → Sıçan ön patileri kaldırmış  
- `fp_hp_vert < 0`: Ön patiler arka patilerden **aşağıda** (büyük y) → Ön patiler zemine veya duvara basıyor  

Koordinat sisteminde y aşağı doğru arttığı için:

```
Üst duvara rearing:  hp_y (zemin) >> fp_y (havada)  →  fp_hp_vert > 45
Alt duvara rearing:  fp_y (zemin) >> hp_y (havada)  →  fp_hp_vert < -45
Grooming:            fp_y ≈ nose_y (yüze yakın)     →  fp_hp_vert < 0, küçük
Normal yürüyüş:      fp_hp_vert ≈ -50 ile +40 arası (değişken)
```

**nanmean kullanım gerekçesi:** Şiddetli rearing sırasında bir ön pati duvara yaslanıp kamera tarafından görülemeyebilir (`likelihood < 0.6`). Tek patiyle hesaplama yapmak yerine `NaN` döndürmek tespiti kesebilirdi. `pandas.DataFrame.mean(axis=1)` NaN değerleri atlayarak mevcut patiyi kullanır.

| Durum | fp_hp_vert değer aralığı |
|-------|--------------------------|
| Normal yürüyüş (üst yarı) | +45 ile +80 |
| Normal yürüyüş (alt yarı) | -80 ile -30 |
| Üst duvara rearing | +45 ile +92 |
| Alt duvara rearing (kompakt) | -45 ile -70 |
| Alt duvara rearing (şiddetli) | -80 ile -110 |
| Grooming | -30 ile 0 |

---

#### Özellik 3: `nose_y` — Burun Yüksekliği

```
nose_y = nose keypoint'in görüntü y koordinatı
```

**Fiziksel anlamı:** Sıçanın arenada bulunduğu dikey konumu gösterir. Üst duvara yaklaştıkça küçülür, alt duvara yaklaştıkça büyür.

| Konum | nose_y değer aralığı |
|-------|---------------------|
| Üst duvar yakını | 62 – 200 px |
| Arena merkezi | 200 – 400 px |
| Alt duvar yakını | 400 – 570 px |

---

#### Özellik 4: `nose2fp` — Burun–Ön Pati Mesafesi

```python
fp_x = mean(left_forepaw_x, right_forepaw_x)
fp_y = mean(left_forepaw_y, right_forepaw_y)

nose2fp = sqrt( (nose_x - fp_x)² + (nose_y - fp_y)² )
```

**Fiziksel anlamı:** Grooming sırasında sıçan ön patilerini yüzüne götürür ve bu mesafe dramatik biçimde düşer. Normal davranışlarda burun ve ön patiler ayrı bölgelerdedir.

| Durum | nose2fp değer aralığı |
|-------|----------------------|
| Normal yürüyüş | 35 – 65 px |
| Rearing | 20 – 60 px (değişken) |
| Grooming | 4 – 20 px |

---

## 5. Sınıflandırma Algoritması

### 5.1 `classify_frames(feat)`

Her kare tek bir etiket alır. Öncelik sırası: **rearing > grooming > other**

```
if (herhangi bir rearing koşulu)  → "rearing"
elif (grooming koşulu)            → "grooming"
else                              → "other"
```

---

### 5.2 Rearing Tespiti — 4 Kural

#### Kural R1: Kompakt Rearing (Duvar Kompresyonu)

```
htdist < 55 px
```

Sıçan herhangi bir duvara yaslanarak tam anlamıyla dik durduğunda,
kafa ve kuyruk tabanı 2B yansımada birbirine yaklaşır.

**Kalibre edilen ground-truth pencereler:**
- Kareler 685–729 (22.3–24.3 s) — htdist ortalama: 44 px
- Kareler 1233–1293 (41.1–43.1 s) — htdist ortalama: 20–30 px

> Not: 1233–1293 aralığında `tail_base` likelihood < 0.6'ya düşer (kuyruk duvara sıkışır).
> Bu yüzden htdist ham (maskelenmemiş) veriden hesaplanır.

---

#### Kural R2: Üst Duvara Rearing (Genişlemiş)

```
fp_hp_vert > 45  AND  nose_y < 165
```

Sıçan üst duvara yaslanıp ayağa kalktığında vücut 2B'de tam sıkışmaz;
ancak ön patiler arka patilerin belirgin biçimde yukarısına çıkar
ve burun üst duvara (küçük y) yakın olur.

**nose_y < 165 koşulunun gerekçesi:**  
165 px eşiği kullanılmadan (örn. 200 px ile), sıçan üst duvara doğru
*yürürken* de fp_hp_vert > 45 sağlanabiliyordu (57–60. saniye yanlış tespiti).
165 px ile tespit ancak sıçan gerçekten duvara yakın olduğunda (~59.8 s) başlar.

**Kalibre edilen ground-truth pencereler:**
- Kareler 323–371 (10.8–12.4 s) — fp_hp_vert ortalama: 45–63, nose_y: 142–165
- Kareler 1791–1892 (59.7–63.1 s) — fp_hp_vert: 50–92, nose_y: 118–165

---

#### Kural R3: Alt Duvara Rearing (Şiddetli)

```
fp_hp_vert < -80  AND  nose_y > 500
```

Sıçan alt duvara yaslanıp ayağa kalktığında ön patiler zemine (büyük y)
gömülür, arka patiler görece yukarıda (küçük y) kalır. Koordinat
sisteminde bu durum fp_hp_vert'i çok negatif yapar.

**-80 eşiğinin gerekçesi:**  
Alt duvara *yaklaşırken* fp_hp_vert geçici olarak -45 ile -70 arasına
girebilir (örn. kareler 955–984, false positive). Yalnızca gerçek
rearing'de değer -80 altına iner.

**Kalibre edilen ground-truth pencere:**
- Kareler 37–55 (1.2–1.8 s) — fp_hp_vert: -91 ile -109, nose_y: 559–569

---

#### Kural R4: Alt Duvara Rearing (Kompakt)

```
fp_hp_vert < -45  AND  nose_y > 500  AND  htdist < 105
```

Kompakt duruşta fp_hp_vert -45 ile -70 arasında kalır (R3 eşiğinin altına inmez),
ancak vücut sıkışır (htdist < 105). Bu iki koşulun birlikteliği
yürüme sırasındaki geçici alt duvar yakınlığını (htdist > 110)
dışarıda bırakır.

**Kalibre edilen ground-truth pencere:**
- Kareler 1016–1038 (33.9–34.6 s) — fp_hp_vert: -55 ile -67, htdist: 90–99, nose_y: 551–570

---

### 5.3 Grooming Tespiti

```
nose2fp < 22  AND  fp_hp_vert < 10  AND  (rearing == False)
```

| Koşul | Gerekçe |
|-------|---------|
| `nose2fp < 22` | Burun ön patilere çok yakın |
| `fp_hp_vert < 10` | Ön patiler yüksekte **değil** — rearing postürlerini (fp_hp_vert > 45) dışarıda bırakır |
| `~rearing` | Rearing önceliklidir |

**fp_hp_vert < 10 koşulunun gerekçesi:**  
39.7–40.8 s aralığında nose2fp < 22 sağlanıyordu ancak fp_hp_vert = 5–48
(ön patiler havada, rearing benzeri postür). Bu koşul eklenerek false
positive giderildi. Onaylanan grooming'de (kareler 4795–5034)
fp_hp_vert sürekli olarak -20 ile -27 arasındadır.

**Kalibre edilen ground-truth pencere:**
- Kareler 4795–5034 (159.8–167.8 s) — nose2fp: 4–20, fp_hp_vert: -20 ile -27

---

## 6. Bout Gruplandırma

### 6.1 `frames_to_bouts(frames, gap, min_dur)`

Birbirini izleyen kare indekslerini sürelere dönüştürür.

**Algoritma:**
```
1. Tespit edilen kareleri sırayla tara
2. İki ardışık kare arasındaki boşluk ≤ gap ise aynı bout'a ekle
3. Boşluk > gap ise yeni bir bout başlat
4. Tamamlanan bout (end - start) < min_dur ise at
```

**Parametreler:**

| Parametre | Değer | Anlamı |
|-----------|-------|--------|
| `INTER_BOUT_GAP` | 15 kare (0.5 s) | Bu kadar kısa kesintiler bout'u bölmez |
| `MIN_BOUT_FRAMES` | 10 kare (0.33 s) | Bundan kısa bout'lar atılır |

**INTER_BOUT_GAP = 15 gerekçesi:**  
Rearing sırasında bazı kareler likelihood düşüşü veya sınırda kalan
eşik değerleri nedeniyle "other" olarak etiketlenebilir. 15 karelik
köprüleme bu kısa kesintilerin aynı davranış bout'unu böldürmesini engeller.

**MIN_BOUT_FRAMES = 10 gerekçesi:**  
Kompakt alt-duvar rearing sinyali (R3 kuralı) yaklaşık 12 kare sürer
(37–55 arası, ~0.4 s). 20 kare eşiği kullanılsaydı bu onaylanmış
bout tespit edilemezdi. Diğer rearing türleri 20+ kare sürdüğünden
düşürülen eşik yalnızca bu spesifik durumu etkiler.

---

## 7. Eşik Değerleri Özet Tablosu

| Sabit | Değer | Birimi | İlgili Kural | Kalibre Edildiği Aralık |
|-------|-------|--------|-------------|------------------------|
| `LIKELIHOOD_THRESH` | 0.6 | — | Ön işleme | — |
| `REAR_COMPACT_HTDIST` | 55 | px | R1 | 685–729, 1233–1293 |
| `REAR_EXTEND_FPHP` | 45 | px | R2 | 323–371, 1791–1892 |
| `REAR_NOSE_Y_MAX` | 165 | px | R2 | 323–371, 1791–1892 |
| `REAR_BOTTOM_STRONG_FPHP` | -80 | px | R3 | 37–55 |
| `REAR_BOTTOM_COMPACT_FPHP` | -45 | px | R4 | 1016–1038 |
| `REAR_BOTTOM_HTDIST` | 105 | px | R4 | 1016–1038 |
| `REAR_BOTTOM_NOSE_Y` | 500 | px | R3, R4 | 37–55, 1016–1038 |
| `GROOM_NOSE2FP` | 22 | px | Grooming | 4795–5034 |
| `GROOM_MAX_FPHP` | 10 | px | Grooming | 4795–5034 |
| `INTER_BOUT_GAP` | 15 | kare | Bout gruplandırma | — |
| `MIN_BOUT_FRAMES` | 10 | kare | Bout gruplandırma | — |

---

## 8. Fonksiyonlar Referans Tablosu

| Fonksiyon | Girdi | Çıktı | Açıklama |
|-----------|-------|-------|----------|
| `load_dlc_csv(path)` | CSV dosya yolu | DataFrame | DLC CSV'yi düz sütun adlarıyla yükler |
| `mask_low_likelihood(df, thresh)` | Ham DataFrame | Maskelenmiş DataFrame | likelihood < thresh → x/y = NaN |
| `compute_features(raw_df, masked_df)` | İki DataFrame | 4 sütunlu özellik DataFrame | htdist, fp_hp_vert, nose_y, nose2fp |
| `classify_frames(feat)` | Özellik DataFrame | Etiket Series | Her kareye 'rearing'/'grooming'/'other' |
| `frames_to_bouts(frames, gap, min_dur)` | Kare indeks dizisi | [(start, end), ...] | Ardışık kareleri bout'lara birleştirir |
| `bouts_to_dataframe(bouts, label, fps)` | Bout listesi | DataFrame | Bout tablosunu CSV için hazırlar |
| `plot_timeline(...)` | Etiketler + bout'lar | PNG dosyası | 3 satırlı karşılaştırma zaman çizelgesi |

---

## 9. Çıktı Dosyaları

| Dosya | Açıklama |
|-------|----------|
| `*_behavior_bouts.csv` | Her bout için: behaviour, bout no, start_frame, end_frame, start_s, end_s, duration_s |
| `*_behavior_frames.csv` | Her kare için: frame, time_s, behaviour |
| `*_behavior_timeline.png` | Ground truth / tespit / kare bazlı karşılaştırma görseli |

### `*_behavior_bouts.csv` sütunları

| Sütun | Tip | Açıklama |
|-------|-----|----------|
| `behaviour` | str | `rearing` veya `grooming` |
| `bout` | int | Davranış türü içinde sıra numarası |
| `start_frame` | int | Başlangıç kare indeksi (0 tabanlı) |
| `end_frame` | int | Bitiş kare indeksi |
| `start_s` | float | Başlangıç zamanı (saniye) |
| `end_s` | float | Bitiş zamanı (saniye) |
| `duration_s` | float | Süre (saniye) |

---

## 10. Tespit Performansı (OpenFieldMA1_2, 168.9 s)

| Davranış | Ground Truth | Tespit | Örtüşme |
|----------|-------------|--------|---------|
| Rearing ~0:01 | 0.7–1.9 s | 1.2–1.8 s | Kısmi (peak yakalandı) |
| Rearing ~0:11 | 10.7–12.4 s | 10.8–12.4 s | ✅ |
| Rearing ~0:23 | 22.8–24.3 s | 23.0–24.3 s | ✅ |
| Rearing ~0:33 | 33–34 s | 33.9–34.6 s | ✅ |
| Rearing ~0:41 | 41.1–43.1 s | 41.2–43.0 s | ✅ |
| Rearing ~1:00 | 60–63 s | 59.7–63.1 s | ✅ |
| Rearing ~1:08 | 67.5–68.7 s | 67.5–68.7 s | ✅ |
| Grooming ~2:40 | 159.8–167.8 s | 160.2–167.7 s | ✅ |

**Toplam rearing süresi:** 10.6 s (%6.3)  
**Toplam grooming süresi:** 7.5 s (%4.4) — onaylanan bout  

---

## 11. Sınırlılıklar ve Notlar

1. **Koordinat bağımlılığı:** Tüm eşikler piksel cinsinden olup bu videonun çözünürlüğüne (arena x: 397–775, y: 158–532) kalibre edilmiştir. Farklı kamera açısı, zoom veya arena boyutunda yeniden kalibrasyon gerekir.

2. **Tek duvar eksikliği:** Sol ve sağ duvarlar için rearing kuralı tanımlanmamıştır. Bu duvarlarda rearing yaparken sıçanın vücut koordinatları farklı bir desen oluşturur.

3. **Kısa rearing bouts:** Bout 1 (1.2–1.8 s) yalnızca peak karelerini yakalar; duvara yaklaşma ve uzaklaşma fazları eşik altında kalır.

4. **Grooming kısa false positive'ler:** 26 s, 30 s ve 73 s'deki ~0.5 s süreli boutlar onaylanmamıştır. `MIN_BOUT_FRAMES = 15–20` veya `GROOM_NOSE2FP = 18` ile elenebilir.

5. **Likelihood tabanlı eksiklik:** DLC'nin güven skoru düşük olan karelerde özellikler NaN döner ve tespit yapılamaz. `LIKELIHOOD_THRESH` düşürülürse daha fazla kare kapsanır ancak gürültü artar.

---

## 12. Kullanım

```bash
# Varsayılan parametrelerle
python src/behavior_detection.py

# Özel CSV ile
python src/behavior_detection.py \
    --csv data/DLCfiltered/OpenFieldMA1_2.csv \
    --fps 30 \
    --out-dir data/DLCfiltered
```

### Eşik ayarı

`behavior_detection.py` dosyasının üstündeki sabitler doğrudan düzenlenebilir:

```python
# Örnek: grooming false positive'leri azaltmak için
GROOM_NOSE2FP   = 18   # 22 → 18
MIN_BOUT_FRAMES = 15   # 10 → 15
```
