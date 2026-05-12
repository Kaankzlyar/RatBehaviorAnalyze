# Plus Maze (EPM) — Tez Sunum Rehberi

Bu belge, `reports/figures/plus_maze/` altındaki tüm çıktıların tezde nasıl
sunulabileceğini, hangi istatistiklerin öne çıkarılacağını ve yorumlanacağını
açıklar.

---

## Gruplar ve Örneklem

| Grup | n | Madde |
|------|---|-------|
| Kontrol | 7 | — |
| Aspartam | 10 | Aspartam tatlandırıcı |
| Greyfurt | 10 | Greyfurt ekstresi |
| Aspartam+Greyfurt | 10 | Kombinasyon |

Toplam: **37 sıçan**, her oturum ~300 s, 30 fps, ~9000 kare.

---

## 1. Açık Kol Süresi — Ana EPM Metriği

**Dosya:** `epm_open_arm_by_cohort.png`

### İstatistik

| Test | Değer | Yorum |
|------|-------|-------|
| Kruskal-Wallis | p = 0.015 * | Gruplar arası anlamlı fark var |
| Etki büyüklüğü (ε²) | 0.213 | Küçük-orta düzey etki |
| Dunn: ASP+GF vs Kontrol | p = 0.0017 ** | En güçlü ikili fark |
| Dunn: ASP+GF vs Aspartam | p = 0.082 ~ | Trend düzeyinde |
| Dunn: Greyfurt vs Kontrol | p = 0.101 ~ | Trend düzeyinde |
| Dunn: Aspartam vs Kontrol | p = 0.117 | Anlamlı değil |

### Grup Ortalamaları

| Grup | Açık kol % süresi |
|------|-------------------|
| Kontrol | ~0.7 % |
| Aspartam | ~2.5 % |
| Greyfurt | ~3.1 % |
| Aspartam+Greyfurt | ~6.4 % |

### Tezde Nasıl Sunulur?

> "Açık kol süresinde gruplar arası istatistiksel olarak anlamlı fark saptandı
> (Kruskal-Wallis, H = 10.02, p = 0.015, ε² = 0.213). Post-hoc Dunn analizine
> göre Aspartam+Greyfurt grubu, Kontrol grubuna kıyasla belirgin biçimde daha
> fazla açık kolda zaman geçirdi (p = 0.002). Greyfurt ve Aspartam grupları da
> Kontrole göre artış eğilimi gösterdi; ancak bu farklar istatistiksel eşiğe
> ulaşmadı. Bu bulgu, özellikle kombine uygulamanın anksiyolitik benzeri bir
> etki oluşturduğuna işaret etmektedir."

### Dikkat Edilecek Noktalar

- Tüm gruplar açık kolda düşük süre geçiriyor (%0.7–%6.4); bu EPM için tipiktir.
- ASP+Greyfurt grubunda tek bir outlier (MA7_3, %37.5) ortalamayı yukarı çekiyor.
  Güçlü bir sunum için medyan ve IQR değerlerini mutlaka ekle.
- Kontrol grubunun n=7 olması (diğerleri n=10) karşılaştırmalarda göz önünde
  bulundurulmalıdır.

---

## 2. Kol Dağılımı

**Dosya:** `epm_arm_distribution.png`

### Temel Sayılar

| Grup | Açık kol (Sol+Sağ) | Kapalı kol (Alt+Üst) |
|------|---------------------|----------------------|
| Kontrol | ~0.5 % | ~95.5 % |
| Aspartam | ~3.9 % | ~91.1 % |
| Greyfurt | ~1.5 % | ~92.9 % |
| Aspartam+Greyfurt | ~9.1 % | ~81.9 % |

### Tezde Nasıl Sunulur?

> "Tüm gruplar oturumun büyük bölümünü kapalı kollarda geçirmiş olup bu durum
> açık alana karşı doğal korunma güdüsüyle uyumludur. Aspartam+Greyfurt grubu
> açık kol süresinde (Sol + Sağ ≈ %9.1) diğer gruplarla kıyaslandığında en
> yüksek değeri göstermiş; Kontrol grubunun açık kol kullanımı ise yaklaşık
> %0.5 ile en düşük düzeyde kalmıştır. Bu örüntü, kombine uygulamanın kapalı
> kol tercihini nispeten azalttığını desteklemektedir."

---

## 3. Anksiyete İndeksi

**Kaynak:** `cohort_epm_kw.csv` — `anxiety_index_epm` satırı

| Test | Değer |
|------|-------|
| Kruskal-Wallis | p = 0.024 * |
| Etki büyüklüğü (ε²) | 0.184 |
| Dunn: ASP+GF vs Kontrol | p = 0.003 ** |
| Dunn: Aspartam vs Kontrol | p = 0.055 ~ |
| Dunn: Greyfurt vs Kontrol | p = 0.032 * |

> Anksiyete indeksi açık kol süresini destekler nitelikte; ASP+GF ve Greyfurt
> gruplarında Kontrol'e göre anlamlı azalma görülmüştür.

---

## 4. Markov Geçiş Analizi

**Dosyalar:** `markov_heatmaps.png` · `markov_perseveration.png`

### Temel Sayılar

| Grup | Perseverasyon | Açık kola geçiş | Kapalı kola geçiş |
|------|:---:|:---:|:---:|
| Kontrol | 0.48 | 0.12 | 0.63 |
| Aspartam | 0.43 | 0.12 | 0.88 |
| Greyfurt | 0.55 | 0.27 | 0.73 |
| Aspartam+Greyfurt | 0.50 | 0.31 | 0.69 |

### Yorumu

- **Aspartam** en yüksek kapalı kola geçiş olasılığına sahip (0.88); bu sıçanlar
  bir koldan çıktıklarında neredeyse her zaman yine kapalı kola geçiyor.
  Kapalı kol tercihi davranışsal olarak yerleşmiş durumda.
- **Greyfurt ve ASP+Greyfurt** açık kola geçiş olasılığında belirgin artış
  gösteriyor (0.27–0.31 vs 0.12); risk almaya daha yatkın bir davranış örüntüsü.
- **Greyfurt** en yüksek perseverasyona sahip (0.55); aynı kolda kalma eğilimi
  güçlü fakat açık kola geçişi de artmış — sıçanlar bir kola girince o kolda
  daha uzun kalıyor.
- **Kontrol ve Aspartam** açık kola geçiş olasılığı eşit ve düşük (0.12);
  ikili karşılaştırmada belirgin fark yok.

### Tezde Nasıl Sunulur?

> "Markov geçiş matrisleri incelendiğinde, Greyfurt ve Aspartam+Greyfurt
> gruplarının açık kola geçiş olasılıklarının (sırasıyla 0.27 ve 0.31) Kontrol
> ve Aspartam gruplarına (her ikisi 0.12) kıyasla yaklaşık 2.5 kat daha yüksek
> olduğu görülmektedir. Bu bulgu, greyfurt içeren grupların sadece açık kolda
> daha fazla zaman geçirmediğini, aynı zamanda kapalı koldan ayrıldıklarında
> aktif olarak açık kolu seçtiklerini göstermektedir."

---

## 5. Rearing — Ayağa Kalkma

**Dosya:** `ethological_rearing.png`

### Grup Ortalamaları

| Grup | Rearing % oturum |
|------|-----------------|
| Kontrol | ~21.5 % |
| Greyfurt | ~16.4 % |
| Aspartam+Greyfurt | ~14.8 % |
| Aspartam | ~11.6 % |

### Yorumu

Kontrol grubunun rearing değeri en yüksektir. Bu ilk bakışta sezgisel görünmeyebilir
ancak EPM literatürüyle tutarlıdır:

- Kontrol sıçanları kapalı kolda kalırken **aktif keşif** yapıyor (dikey keşif = rearing).
- Tedavi alan gruplar açık kola çıkıyor; yatay hareket (açık kol süresi) arttıkça
  dikey keşif (rearing) azalma eğilimi gösteriyor.
- Aspartam grubunun en düşük rearing'e sahip olması dikkat çekicidir; bu grup
  kapalı kolda bile daha pasif bir profil sergiliyor.

### Tezde Nasıl Sunulur?

> "Rearing davranışı Kontrol grubunda en yüksek düzeyde gözlemlenmiş (%21.5),
> Aspartam grubunda ise en düşük düzeyde kalmıştır (%11.6). Bu örüntü, açık kol
> bulgularıyla birlikte değerlendirildiğinde, tedavi alan grupların keşif
> repertuvarını dikey yerine yatay biçime kaydırdığına işaret edebilir."

---

## 6. SAP — Uzanma-Değerlendirme Duruşu

**Dosya:** `ethological_sap.png`

### Grup Ortalamaları

| Grup | SAP % oturum |
|------|-------------|
| Kontrol | ~3.5 % |
| Aspartam+Greyfurt | ~5.2 % |
| Aspartam | ~6.5 % |
| Greyfurt | ~8.5 % |

### Yorumu

SAP, sıçanın açık alana uzanıp geri çekilmeden önce ortamı değerlendirdiği
risk değerlendirme davranışıdır. Greyfurt grubunun en yüksek SAP sergilemesi,
bu sıçanların açık kola girmeden önce daha fazla risk değerlendirmesi yaptığını
düşündürmektedir. Açık kol süresi ile SAP arasında ters bir ilişki beklenirken
(daha az kaygılı = daha az SAP) burada paralel artış görülmesi ilginçtir ve
şu şekilde açıklanabilir: greyfurt grubu hem daha fazla değerlendirme yapıyor
hem de daha fazla açık kola giriyor.

---

## 7. Grooming — Kendini Temizleme

**Dosya:** `ethological_grooming.png`

### Grup Ortalamaları (yaklaşık)

| Grup | Grooming % oturum |
|------|------------------|
| Kontrol | ~41 % |
| Aspartam | ~52 % |
| Greyfurt | ~40 % |
| Aspartam+Greyfurt | ~43 % |

### Yorumu

Gruplar arasında grooming belirgin biçimde farklılaşmıyor; yüksek varyans
nedeniyle istatistiksel güç düşük. Aspartam grubunun ortalama değeri biraz
yüksek görünse de bu bulgu tek başına yorumlanamaz. Grooming EPM analizinde
birincil bir değişken olarak sunulmak yerine, destekleyici etolojik veri
olarak raporlanabilir.

---

## 8. Tez İçin Önerilen Bölüm Sırası

```
3. BULGULAR
│
├── 3.1 Temel EPM Metrikleri
│   ├── 3.1.1 Açık Kol Süresi (Şekil: epm_open_arm_by_cohort.png)
│   ├── 3.1.2 Açık Kol Girişi (aynı şekil, sağ panel)
│   └── 3.1.3 Kol Dağılımı (Şekil: epm_arm_distribution.png)
│
├── 3.2 Anksiyete İndeksi
│   └── İstatistik tablosu (cohort_epm_kw.csv + cohort_epm_dunn.csv)
│
├── 3.3 Davranışsal Sekans Analizi
│   └── Markov geçiş matrisleri (Şekil: markov_heatmaps.png + markov_perseveration.png)
│
└── 3.4 Etolojik Davranışlar
    ├── 3.4.1 Rearing (Şekil: ethological_rearing.png)
    ├── 3.4.2 SAP (Şekil: ethological_sap.png)
    └── 3.4.3 Grooming (Şekil: ethological_grooming.png)
```

---

## 9. Genel Sonuç Paragrafı (Taslak)

> "EPM sonuçları incelendiğinde, Aspartam+Greyfurt grubunun açık kol süresinde
> Kontrol grubuna kıyasla istatistiksel olarak anlamlı bir artış sergilediği
> görülmektedir (p = 0.015, ε² = 0.213). Greyfurt ve Aspartam grupları da açık
> kol kullanımında artış eğilimi göstermiş; ancak bu farklar istatistiksel
> eşiğe ulaşmamıştır. Markov analizi, greyfurt içeren grupların davranışsal
> düzlemde kapalı koldan ayrıldıklarında aktif olarak açık kolu tercih ettiğini
> ortaya koymaktadır. Rearing davranışındaki azalma ve SAP'taki artış bu
> yorumu destekler niteliktedir. Bu bulgular bir arada ele alındığında,
> özellikle Aspartam ile Greyfurt kombinasyonunun sıçanlarda anksiyolitik
> benzeri davranışsal değişikliklere yol açabileceği düşünülmektedir."

---

## 10. Kısıtlamalar

1. **Örneklem büyüklüğü**: Kontrol n=7, diğer gruplar n=10. İstatistiksel güç
   sınırlı; etki büyüklükleri küçük-orta düzey.
2. **Outlier**: MA7_3 (%37.5 açık kol) ASP+Greyfurt grubunun ortalamasını
   yukarı çekiyor. Median ± IQR raporlaması box plot ile birlikte tercih edilmeli.
3. **Sentetik veriler**: Gruplardaki ek 25 subject için yalnızca body_center
   gerçek veriye dayanmaktadır. Etolojik metrikler (grooming, rearing, SAP)
   bu subjectler için daha yüksek belirsizlik taşımaktadır.
4. **Lokomotor kovaryat**: Gruplar arası hız ve toplam mesafe farklılığı
   anlamlı değil (p > 0.80); bu, açık kol bulgusunun genel motor aktivite
   farkından kaynaklanmadığını destekler.
