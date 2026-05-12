# markov_perseveration.png — Aciklama

**Ureten script:** `analysis/plus_maze/markov_analysis.py`  
**Veri kaynagi:** `data/plus_maze_metrics_all.csv` → `entry_sequence` sutunu  
**Iliskili dosya:** `reports/figures/plus_maze/markov_heatmaps.png` — ayni Markov verisinin matris gosterimi

---

## Ne Gosteriyor?

Markov gecis matrislerinden elde edilen uc ozetin **kohort bazinda gruplu bar chart** karsilastirmasi.
Her kohort icin ayni veri; perseverasyon, acik kola gecis ve kapali kola gecis olmak uzere uc farkli
boyuttan gorsellestirilir. Grafigin amaci kohortlar arasindaki davranis farklarini tek bakista
karsilastirmaktir.

---

## Bar Renkleri ve Ne Olcuyorlar?

| Renk | Olcut | Hesaplama |
|---|---|---|
| Kirmizi | **Persiverasyon** | Markov matrisinin diagonal degerlerinin aritmetik ortalamasi (4 kolun ayni kola gecis olasiligi) |
| Yesil | **Acik kola gecis** | Sol Kol ve Sag Kol sutunlarinin satirbazinda ortalama gecis olasiligi |
| Mavi | **Kapali kola gecis** | Alt Kol ve Ust Kol sutunlarinin satirbazinda ortalama gecis olasiligi |

- **Kesik gri cizgi (0.25):** hayvan dort kol arasinda rastgele secim yapsaydi beklenen esit
  olasilik esigi. Bir degerin bu cizginin uzerinde olmasi o yone belirgin bir yonelim oldugunu
  gosterir.
- **Her kohorttaki n= degeri:** o kohortun tum sicalarina ait toplam kol gecisi sayisidir.

---

## Kohort Sonuclari

| Kohort | n_gecis | Persiverasyon | Acik kola gecis | Kapali kola gecis | Yorum |
|---|---|---|---|---|---|
| Kontrol | 37 | 0.37 | 0.02 | 0.73 | Belirgin kapali kol tercihi; acik kola gecis yok denecek kadar az |
| Aspartam | 31 | 0.42 | 0.11 | 0.89 | En yuksek kapali kol yonelimi; acik kola ilgi en az olan ikinci grup |
| Greyfurt | 21 | **0.26** | 0.03 | 0.97 | En dusuk persiverasyon; buna ragmen neredeyse tum gecisler kapali kola |
| ASP+Greyfurt | 65 | **0.59** | 0.39 | 0.61 | En yuksek persiverasyon; tek fark edilir acik kol gecisine sahip grup |

---

## Nasil Okunur?

### Uc cubuklu grubun birlikte yorumu

Persiverasyon + acik kola gecis + kapali kola gecis bir araya getirildiginde, uc degerin toplaminin
1.00'i gecmesi normaldir; bu degerler birbirinden bagimsiz olculer oldugu icin toplamlarinin 1 olmasi
gerekmez. Her olcut o kohortun gecislerini farkli bir mercekten tarif eder.

Bir kohort icin ideal dusuk-anksiyete profili soyle gorunur:
- Kirmizi (persiverasyon) dusuk — hayvan ayni kola takilmiyor
- Yesil (acik kola gecis) yuksek — hayvan acik kolu aktif olarak tercihe aliyor
- Mavi (kapali kola gecis) orta — kapali kol tamamen terk edilmemis ama tek hedef degil

### Esit olasilik esigi (0.25) ne anlama geliyor?

Dort kollu bir labirentte tamamen rastgele hareket eden bir hayvan her kolu esit olasilikla secer;
bu durumda her bar 0.25 civarinda olur. Herhangi bir cubuk 0.25'in belirgin sekilde uzerindeyse
o kohort o yonde istatistiksel anlamda tercih gosteriyor demektir.

### Greyfurt grubunun paradoksal gorunumu

Greyfurt grubu en dusuk persiverasyon degerine (0.26) sahipken acik kola gecis orani yalnizca
0.03'tur. Bu ilk bakista celisik gorunebilir. Aciklama sudur: dusuk persiverasyon ayni kolu
tekrarlamamak anlamina gelir, ancak bu durum acik kolu tercih ettigini garanti etmez. Greyfurt
grubu kol degistiriyor, fakat bu gecislerin buyuk cogunlugu yine kapali kollar arasinda (Alt↔Ust)
gerceklesiyor; acik kollara gecis son derece nadir kalıyor.

### ASP+Greyfurt grubunun ayrisan profili

Bu grup diger uc kohortten tamamen farkli bir profil ortaya koyuyor:
- Persiverasyon 0.59 ile en yuksek — kol degistirme isteksizligi belirgin
- Buna ragmen acik kola gecis 0.39 ile tek anlamli yesil cubuk — bu grubun bir kismi
  acik kolu seciyor, ancak diger kismi ayni kola takilip kaliyor
- Kapali kola gecis 0.61 ile en dusuk — kapali kol hakimiyeti diger gruplara gore zayifladi

Bu profil grup icinde bireysel farklilik oldugunu dusunutuyor: bazi hayvanlar acik kolu aktif
seciyor, bazi hayvanlar ise yuksek persiverasyon sergiliyor.

### Kural ozeti

| Profil | Persiverasyon | Acik gecis | Kapali gecis | Yorum |
|---|---|---|---|---|
| Yuksek anksiyete | Yuksek | Cok dusuk | Cok yuksek | Hayvan kapali kolda kalıyor, hic cikmiyor |
| Dusuk anksiyete | Dusuk | Yuksek | Orta | Hayvan acik kolu aktif olarak kesiyor |
| Rastgele hareket | ~0.25 | ~0.25 | ~0.25 | Belirgin tercih yok |
| Karma profil | Yuksek | Orta | Orta | Grup ici bireysel farklılık var olabilir |

---

## Iliskili Dosyalar

- `reports/markov_transition_matrices.csv` — ham olasilik degerleri (uzun format)
- `reports/markov_transition_counts.csv` — ham gecis sayilari
- `reports/figures/plus_maze/markov_heatmaps.png` — ayni verinin 4x4 matris gosterimi (kol bazinda detay)
- `reports/figures/plus_maze/markov_heatmaps.md` — matris okuma rehberi ve Markov analiz mantigi
