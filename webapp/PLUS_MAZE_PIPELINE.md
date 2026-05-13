# Plus Maze (EPM) Analiz ve Tahmin Pipeline'ı

Bu belge, EPM (Elevated Plus Maze — Yükseltilmiş Artı Labirent) test videosundan
başlayıp **Kontrol / Tedavi** grup tahminine giden tüm süreci sade bir dille
anlatır. Her aşamada hangi dosyanın çalıştığı, ne aldığı, ne ürettiği ve neyin
nasıl hesaplandığı belirtilmiştir. Sadece **Plus Maze** içindir; Open Field veya
T-Maze pipeline'ı kapsam dışıdır.

---

## 0. Genel Akış

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  RAW VIDEO      │ ───► │  DeepLabCut      │ ───► │  FILTERED CSV    │
│  (.avi/.mp4)    │      │  10 keypoint     │      │  pose dosyası    │
└─────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                            │
                                                            ▼
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  TAHMİN         │ ◄─── │  Lojistik Reg.   │ ◄─── │  Metrikler       │
│  Kontrol/Tedavi │      │  (LOOCV, n=37)   │      │  (tmaze_metrics) │
└─────────────────┘      └──────────────────┘      └──────────────────┘
```

Her bir aşama aşağıda ayrıntılı açıklanmıştır.

---

## 1. DeepLabCut ile 10 Noktanın İşaretlenmesi

EPM video kaydından önce farenin vücudu üzerinde **10 anatomik nokta**
(keypoint) işaretlenir. DeepLabCut bu noktaları her karede (frame) izler ve her
biri için `(x, y, likelihood)` döndürür. Likelihood = modelin o karedeki tahmin
güveni (0 – 1).

| # | Keypoint | Türkçe karşılığı | Pipeline'da nerede kullanılır? |
| - | -------- | ---------------- | ------------------------------ |
| 1  | `nose`            | Burun         | Yön / baş hareketi (kullanılmazsa pasif) |
| 2  | `head`            | Baş           | Baş–kuyruk mesafesi (ethological_features) |
| 3  | `left_ear`        | Sol kulak     | Anatomi referansı |
| 4  | `right_ear`       | Sağ kulak     | Anatomi referansı |
| 5  | `body_center`     | Gövde merkezi | **Birincil izleme noktası** — tüm metrikler bu nokta üzerinden hesaplanır |
| 6  | `left_forepaw`    | Sol ön patı   | Etolojik davranış (tımarlanma, vb.) |
| 7  | `right_forepaw`   | Sağ ön patı   | Etolojik davranış |
| 8  | `left_hindpaw`    | Sol arka patı | Ayakta durma (rearing) tespiti |
| 9  | `right_hindpaw`   | Sağ arka patı | Ayakta durma tespiti |
| 10 | `tail_base`       | Kuyruk dibi   | Vücut uzunluğu, anatomi referansı |

DeepLabCut çıktısı, `pandas.read_csv(... header=[1, 2])` ile okunan iki seviyeli
başlığa sahip CSV'dir:

```
                nose                head              body_center        ...
                x      y    lkh     x      y    lkh   x       y     lkh ...
frame_idx=0     412.3  208  0.98   …       …    …    417.0   220   0.99
frame_idx=1     412.7  209  0.97   …       …    …    417.3   220   0.99
```

Sentetik / yapay üretilmiş veriler için `analysis/plus_maze/fix_synthetic_keypoints.py`
keypoint'leri labirent sınırları içine sıkıştırır (anatomik tutarlılığı korur).

**Şu an çalışan modelde sadece `body_center` aktif olarak kullanılıyor.** Diğer
9 nokta etolojik davranış (ayakta durma / tımarlanma) modülünde yer alıyor;
sınıflandırıcıya bu modülün çıktısı feature olarak verilmiyor.

---

## 2. Maze Geometrisi — Kol Koordinatları

EPM artı şeklindedir: **2 açık kol** (sol / sağ) + **2 kapalı kol** (üst / alt)
+ **1 kavşak** (junction). Her kolun video çerçevesindeki piksel sınırları, bir
kez `analysis/plus_maze/show_frame_coords.py` ile elle belirlenir ve şu formatta
tutulur:

```json
{
  "bottom_arm": [547, 603, 402, 713],
  "left_arm":   [260, 552, 347, 403],
  "right_arm":  [602, 894, 345, 402],
  "top_arm":    [546, 604,  33, 347]
}
```

Format: `[xmin, xmax, ymin, ymax]`. Bu dosya `data/arm_coords.json` olarak
saklanır ve webapp otomatik yükler.

Bir kare için `body_center (x, y)` hangi kolun dikdörtgeni içinde kalıyorsa o
zone'a; hiçbirinin içinde değilse `junction`'a; geçersiz kare ise (likelihood
düşük) `unknown`'a etiketlenir → `assign_zones()` fonksiyonu.

---

## 3. Pose Temizleme (Noise Reduction)

`tmaze_metrics.load_body_center()` üç adımda gürültüyü azaltır:

1. **Likelihood eşiği:** `likelihood < 0.6` olan kareler `NaN` (geçersiz).
2. **Sıçrama eşiği:** Ardışık iki kare arasındaki yer değiştirme
   `√(dx² + dy²) > 60 px` ise atlama (DLC hatası) sayılır → `NaN`.
3. **Yuvarlanan medyan:** 5 karelik kayan medyan filtre (`smooth=5`)
   anlık titreşimi yumuşatır; `NaN` kareler korunur.

Bu adımlar olmadan tek bir DLC hatası tüm metrikleri kirletir (örneğin sahte
"toplam giriş" sayısı şişer).

---

## 4. Metrik Çıkarımı — `tmaze_metrics.py`

**Girdi:** Tek bir farenin DLC CSV'si + arm koordinatları
**Çıktı:** `data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>_plus_maze_metrics.csv`
(tek satır)

Hesaplanan metrikler dört grup altında:

### 4.1 Zone Occupancy (Bölge Oranları)

Geçerli karelerin (`valid_labels`) toplamına oran olarak:

| Metric | Hesap |
| ------ | ----- |
| `pct_time_left`     | `(labels == "left_arm").sum()  / n_valid × 100` |
| `pct_time_right`    | `(labels == "right_arm").sum() / n_valid × 100` |
| `pct_time_top`      | `(labels == "top_arm").sum()   / n_valid × 100` |
| `pct_time_bottom`   | `(labels == "bottom_arm").sum()/ n_valid × 100` |
| `pct_time_junction` | `(labels == "junction").sum()  / n_valid × 100` |

Açık kol süresi `pct_open_arm = pct_time_left + pct_time_right`; kapalı kol
süresi benzer şekilde top+bottom toplamıdır. Kapalı kollar, klasik EPM
düzeneğinde duvarlarla çevrili olduğu için fare buralarda daha güvende hisseder
→ yüksek `pct_open_arm` = düşük anksiyete.

### 4.2 Arm Entries (Kol Girişleri)

Bir "giriş" sayılması için fare ilgili kolda **en az 3 ardışık geçerli kare
(≈ 0.1 sn @ 30 fps)** kalmalıdır. Bu eşik, sınır çevresindeki titreşimi giriş
saymaktan korur.

```
total_entries        = tüm kollara yapılan giriş sayısı
left_entries / right_entries / top_entries / bottom_entries
most_visited_arm     = en çok girilen kol
arm_preference_index = (max_giriş − min_giriş) / total   ∈ [0, 1]
```

`arm_preference_index = 0` → kollara eşit dağılım; `1` → tek bir kola yönelim.

### 4.3 Alternasyon ve Perseverasyon

Çalışan bellek / bilişsel esneklik göstergeleri. `sequence` = girilen kolların
sırası.

```
succ_alt   = ardışık iki girişin farklı kola olma sayısı
succ_pers  = ardışık iki girişin aynı kola olma sayısı
successive_alternation_pct = succ_alt  / (total − 1) × 100
perseveration_rate_pct     = succ_pers / (total − 1) × 100
tetrad_alternation_pct     = 4'lü ardışık dizide 4 farklı kol oranı
```

Yüksek alternasyon = esnek keşif. Yüksek perseverasyon = aynı kola takılma /
katı davranış.

### 4.4 Lokomosyon

```
mean_speed_px_s   = ortalama( √(dx² + dy²) × fps ),  geçerli kareler
total_distance_px = toplam yol uzunluğu (piksel)
```

---

## 5. Türetilmiş Özellikler — `batch_metrics.py`

Tüm farelerin tek satırlık metrik dosyalarını birleştirir →
`data/plus_maze_metrics_all.csv`. Aynı anda EPM'ye özel **türetilmiş kolonları**
ekler:

```
pct_open_arm          = pct_time_left + pct_time_right
pct_closed_arm        = pct_time_top  + pct_time_bottom
pct_open_arm_entries  = (left_entries + right_entries) / total × 100
pct_closed_arm_entries= (top_entries + bottom_entries) / total × 100
anxiety_index_epm     = (pct_open_arm + pct_open_arm_entries) / 2
```

`anxiety_index_epm` literatürdeki bütünleşik EPM anksiyete göstergesidir
(yüksek = düşük anksiyete).

---

## 6. Model Eğitimi — `train_epm_classifier.py`

**Girdi:** `data/plus_maze_metrics_all.csv` (37 satır, ~35 kolon)
**Çıktılar:**
- `models/epm_classifier/scaler_epm.pkl` (StandardScaler)
- `models/epm_classifier/lr_epm.pkl`     (LogisticRegression bundle)

### 6.1 Modele Verilen 9 Feature

```python
FEATURE_COLS = [
    "pct_open_arm",
    "anxiety_index_epm",
    "pct_open_arm_entries",
    "total_entries",
    "successive_alternation_pct",
    "perseveration_rate_pct",
    "mean_speed_px_s",
    "arm_preference_index",
    "pct_time_junction",
]
```

### 6.2 Etiket

```python
df["label"] = (df["cohort"] != "Control").astype(int)
# Control → 0,  diğer her kohort (ASP / Greyfurt / ASP+Greyfurt) → 1 (Tedavi)
```

### 6.3 Eğitim Adımları

1. **Imputation:** Eksik feature'lar için sütun ortalaması
   (`SimpleImputer(strategy="mean")`).
2. **Ölçekleme:** `StandardScaler` ile her feature'ı sıfır-ortalama / birim-varyansa.
3. **Sınıflandırıcı:** `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`.
4. **Değerlendirme:** **LOOCV** (Leave-One-Out Cross-Validation) — 37 farenin
   her biri sırayla tek başına teste ayrılır, kalan 36 ile model eğitilir.
   Bu 37 ayrı tahminin birleşimi:
   - AUC = 0.733
   - F1 (Tedavi) = 0.821
   - Duyarlılık = %76.7
5. **Refit:** Tüm 37 fare üzerinde son model yeniden eğitilir ve `.pkl` olarak
   kaydedilir → Webapp bu modeli yükler.

### 6.4 Neden Lojistik Regresyon?

- **Küçük n için uygun:** 37 örneklem üzerinde derin model / random forest
  overfit eğilimi yüksektir.
- **Yorumlanabilir:** Her feature'ın bir lineer katsayısı vardır → "Karar
  Katkıları" panelindeki `katkı = katsayı × normalize_değer` doğrudan modelin
  içinden okunabilir.
- **Olasılık çıktısı:** `P(Tedavi)` ve `P(Kontrol)` sigmoid üzerinden gelir;
  toplam her zaman 1.

---

## 7. Tahmin — `webapp/plus_maze_app.py`

Streamlit dashboard'unun veri akışı:

```
   1. CSV yükle      ───────────────► tmaze_metrics.compute_metrics()
                                                │
                                                ▼
   2. add_derived()   ◄───────── pct_open_arm, anxiety_index_epm, …
                                                │
                                                ▼
   3. Model bundle yükle (scaler_epm.pkl + lr_epm.pkl)
                                                │
                                                ▼
   4. x_imputed → x_scaled → clf.predict_proba()
                                                │
                                                ▼
   5. Karar katkıları:  katsayı × x_scaled[i] sıralanır
                                                │
                                                ▼
   6. UI: 4 sekme
        ├─ Tahmin       → grup kartı + olasılık barı + temel metrikler + Karar Katkıları
        ├─ Metrikler    → kol süreleri (bar), model özellikleri, kol detayları
        ├─ Görselleştirme → Hareket Rotası (orbit_plot) + Isı Haritası (activity_heatmap)
        └─ İndir        → CSV / JSON / PNG çıktıları + önizleme
```

### 7.1 Karar Katkıları (Yorumlama)

Lojistik regresyonda `logit(P(Tedavi)) = Σ wᵢ · zᵢ`, burada `wᵢ` o feature'ın
katsayısı, `zᵢ` ise standardize edilmiş değeridir. Her feature için
`katkı = wᵢ · zᵢ` hesaplanır ve mutlak değere göre sıralanır:

- **Pozitif katkı** → "Tedavi" yönüne iter (kırmızı bar)
- **Negatif katkı** → "Kontrol" yönüne iter (mavi bar)

Tablonun altındaki tıklanabilir kartlarda her feature için Türkçe açıklama ve
formülü görüntülenir.

---

## 8. Görselleştirme Çıktıları

### 8.1 `orbit_plot.py` — Hareket Rotası
`body_center (x, y)` zaman serisini zone renklerine göre boyar
(sol=yeşil, sağ=turuncu, alt=mavi, üst=mor, kavşak=gri). Başlangıç noktası
yeşil daire, bitiş kırmızı baklava ile işaretlenir. `LineCollection` kullanarak
zaman gradyanı oluşturulur.

### 8.2 `activity_heatmap.py` — Isı Haritası
Tüm geçerli `(x, y)` çiftlerini 2B Gaussian KDE ile yumuşatır ve log-ölçeğinde
ısı haritası oluşturur. Hangi bölgelerin daha çok ziyaret edildiğini gösterir.

---

## 9. Performans Özeti

LOOCV ile 37 fare üzerinde elde edilen sonuçlar:

| Metric | Değer | Anlamı |
| ------ | ----- | ------ |
| Algoritma     | Lojistik Regresyon | Lineer, yorumlanabilir |
| Eğitim seti   | 37 sıçan (LOOCV)   | Her sıçan bir kez teste ayrılır |
| AUC           | 0.733              | Rastgele iki örneğin doğru sıralanma olasılığı |
| F1 (Tedavi)   | 0.821              | Tedavi sınıfı için kesinlik–duyarlılık dengesi |
| Duyarlılık    | %76.7              | Tedavi sıçanlarının doğru bulunma oranı |

---

## 10. Sıkça Karıştırılan Noktalar

- **`total_entries` ve `junction` ilişkisi:** Fare bir koldan diğerine geçerken
  zorunlu olarak kavşaktan geçer; ancak kavşak için ayrı bir "giriş" sayısı
  raporlanmaz çünkü bu sayı `total_entries − 1`'in fonksiyonu olur ve yeni bir
  bilgi içermez. Yalnızca `pct_time_junction` raporlanır.
- **Anksiyete yönü:** EPM'de **düşük** `pct_open_arm` ve **düşük**
  `pct_open_arm_entries` **yüksek** anksiyeteye karşılık gelir
  (`anxiety_index_epm` tersine yorumlanır: yüksek değer = düşük anksiyete).
- **Class imbalance:** 37 sıçanın çoğu tedavi kohortlarındandır; bu nedenle
  `class_weight="balanced"` lojistik regresyonun ayarında kritiktir.

---

*Hazırlayan: Plus Maze pipeline'ı `analysis/plus_maze/` altında, webapp
`webapp/plus_maze_app.py` içinde. Bu belge çalıştırılan kodu yansıtır; kod
değişikliklerinde güncellenmelidir.*
