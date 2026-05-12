# markov_heatmaps.png — Açıklama

**Üreten script:** `analysis/plus_maze/markov_analysis.py`  
**Veri kaynağı:** `data/plus_maze_metrics_all.csv` → `entry_sequence` sütunu

---

## Ne Gösteriyor?

Her sıçanın plus maze kolları arasındaki ardışık geçişlerinden oluşturulan **birinci dereceden Markov geçiş matrisleri**. 4 kohort için ayrı ayrı 4×4 olasılık matrisi hesaplanmış ve ısı haritası olarak çizilmiştir.

---

## Matris Yapısı

|  | Bottom (kapali) | Top (kapali) | Left (acik) | Right (acik) |
|---|---|---|---|---|
| **Bottom (FROM)** | P(B→B) | P(B→T) | P(B→L) | P(B→R) |
| **Top (FROM)** | P(T→B) | P(T→T) | P(T→L) | P(T→R) |
| **Left (FROM)** | P(L→B) | P(L→T) | P(L→L) | P(L→R) |
| **Right (FROM)** | P(R→B) | P(R→T) | P(R→L) | P(R→R) |

- **Satır** = kaynak kol (bir önceki giriş)  
- **Sütun** = hedef kol (bir sonraki giriş)  
- **Değer** = satır-normalize olasılık (0–1) + ham geçiş sayısı (n=…)  
- **Diagonal hücreler (kırmızı kenarlık)** = perseverasyon — hayvanın aynı kolu arka arkaya tekrarlaması  
- **L ve R sütunları (yeşil kesik kenarlık)** = açık kola geçiş eğilimi (anksiyolitik davranış göstergesi)

---

## Kohort Sonuçları

| Kohort | n_geçiş | Perseverasyon ort. | Yorum |
|---|---|---|---|
| Control | 37 | 0.37 | Makul düzeyde perseverasyon; bottom kolda yığılma var (B→B = 0.82, n=18) |
| Aspartame | 33 | 0.42 | Control'den hafif yüksek; B→B = 0.78 (n=14), geçiş çeşitliliği sınırlı |
| Grapefruit | 23 | 0.35 | 4 kohort içinde en düşük perseverasyon; en dengeli geçiş dağılımı |
| ASP+Greyfurt | 65 | **0.59** | Belirgin yüksek perseverasyon; tüm diagonal değerleri güçlü, bottom→bottom baskın |

---

## Nasil Okunur?

### Temel analiz mantigi

Bu grafik **birinci dereceden Markov analizine** dayanir. Birinci derece Markov varsayimi su demektir:
hayvanin bir sonraki kol secimi yalnizca bulundugu mevcut kola baglidir, onceki gecmise degil.
Her kohort icin butun sicalarin `entry_sequence` kayitlari birlestirilir ve ard ardina gelen her
(kaynak_kol, hedef_kol) ciftinden ham gecis sayimlari olusturulur. Bu sayimlar satirlara gore
normalize edilerek 0-1 araliginda olasiliga donusturulur: bir satirda tum degerler toplamda 1.00
verir, yani o kaynak koldan hareket eden hayvanin bir sonraki hamlesini nereye yaptiginin tam
olasilik dagilimini gosterir.

### Diagonal degerler — Perseverasyon

Bir matrisin kosegeni (sol ust'ten sag alt'a uzanan hucre serisi) **perseverasyonu** temsil eder:
hayvanin az once ciktigini kola tekrar girmesi. Perseverasyon ortalamasini hesaplamak icin dort
kosegen degerin aritmetik ortalamasi alinir. Yuksek perseverasyon orani hayvanin kol degistirme
egiliminin dusuk oldugunu, yani arastirmaci davranisinin zayifladigini gosterir. EPM baglaminda
bu durum anksiyete ile iliskilendirilir: anksiyoz hayvanlar tanidik ve guvenli hissettikleri
kola donme egilimindedir.

### L ve R sutunlari — Acik kola gecis

Matrisin ucuncu ve dorduncu sutunlari (Left ve Right) **acik kola yonelen gecisleri** gosterir.
Acik kollar EPM'de tehdit algilayan hayvanlar tarafindan kacinilanilir; bu sutunlarda yuksek deger
hayvanin hangi koldan gelirse gelsin acik kolu bir sonraki hedef olarak secme olasiliginin yuksek
oldugunu gosterir. Bu durum daha az kaygili, arastirmaci bir davranis profiline isaret eder.

### B ve T sutunlari — Kapali kola gecis

Birinci ve ikinci sutunlar (Bottom ve Top) **kapali kola yonelen gecisleri** gosterir. Kapali kollar
yuksek duvarlar nedeniyle korunmali hissettiren alanlardır. Bu sutunlarda yuksek deger hayvanin
acik koldan bile olsa kapali kola gecmeyi tercih ettigini, yani kaygidan kacinma davranisi
sergiledigini gosterir.

### Renk kodlari ve isaretler

- **Mavi renk yogunlugu**: hucredeki gecis olasiliginin buyuklugunu gosterir; koyu mavi yuksek
  olasilik demektir.
- **Kirmizi kenarlik (diagonal)**: perseverasyon hucreleri; bu degerler ne kadar yuksekse hayvan
  o kadar az alternatif kol araştiriyor demektir.
- **Yesil kesik kenarlik (L ve R sutunlari)**: acik kol hedefli gecisler; bu alanlar ne kadar
  aydinliksa (yuksek olasilik) hayvanin anksiyete seviyesi o kadar dusuk yorumlanir.
- **Her hucredeki n= degeri**: o gecisin gozlemlendigi ham kere sayisidir; dusuk n degerli
  hucrelerdeki yuksek olasiliklar kucuk orneklem buyuklugu nedeniyle dikkatli yorumlanmalidir.

### Kural ozeti

| Durum | Yorum |
|---|---|
| Yuksek diagonal, dusuk L/R sutunu | Yuksek anksiyete; hayvan kapali kolda kaliyor |
| Dusuk diagonal, yuksek L/R sutunu | Dusuk anksiyete; hayvan acik kolu kesfediyor |
| Tum sutunlar yaklasik esit (~0.25) | Rastgele kol secimi; belirgin tercih yok |
| Tek bir satir baskinsayisi yuksek | O kaynak koldan gelen hayvanlar belirli bir hedefe yonelimli |

---

## İlişkili Dosyalar

- `reports/markov_transition_matrices.csv` — ham olasılık değerleri (uzun format)  
- `reports/markov_transition_counts.csv` — ham geçiş sayıları  
- `reports/figures/plus_maze/markov_perseveration.png` — kohort bazında perseverasyon / açık kol / kapalı kol bar chart karşılaştırması
