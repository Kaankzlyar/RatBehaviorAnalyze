# Etolojik Özellikler Özet Grafiği

**Grafik dosyası:** `ethological_summary.png`  
**Script:** `analysis/tmaze/ethological_features.py`  
**Veri kaynağı:** 12 sıçanın Plus Maze DLC pose verisi (10 keypoint, 30 fps)

---

## Bu Grafik Ne Gösteriyor?

Grooming ve SAP analizlerini **yan yana** tek bir panel üzerinde sunar. Amaç iki davranışın kohortlar arası örüntüsünü tek bakışta karşılaştırmaktır.

- **Sol panel:** Grooming — Kendini Temizleme (% oturum süresi)
- **Sağ panel:** SAP — Uzanma-Değerlendirme Duruşu (% oturum süresi)
- **Alt başlık:** Her iki davranışın otomatik tespit eşiklerini özetler
- **Alt legend:** Dört kohortun renk kodları

---

## Grafik Nasıl Okunmalı?

### İki paneli birlikte okuma

İki davranış **farklı anksiyete mekanizmalarını** temsil eder:

| Davranış | Arttığında ne anlama gelir | Azaldığında ne anlama gelir |
|----------|---------------------------|----------------------------|
| **Grooming** | Anksiyete yükü artıyor (Cruz 1994) — hayvan enerjiyi temizlenmeye yönlendiriyor | Anksiyete düşük veya hayvan başka davranışla meşgul |
| **SAP** | Hayvan tehlikeli alana girmeden önce uzun süre risk değerlendiriyor | Hayvan daha az tereddütle açık kola giriyor (anksiyolitik etki işareti) |

İdeal anksiyolitik etki için beklenen örüntü: **Grooming azalır + SAP azalır + Açık kol süresi artar.**

### Kohort bazında birlikte yorum

| Kohort | Grooming % | SAP % | Birlikte yorum |
|--------|-----------|-------|---------------|
| **Control** | 10.4% | 9.1% | Her iki davranış en düşük — referans (kapalı kol tercihi ama düşük anksiyete davranışı?) |
| **Aspartame** | 15.6% | 14.5% | İkisi birlikte artmış — stres yükü artmış olabilir |
| **Grapefruit** | **19.3%** | **18.0%** | Her ikisi de en yüksek gruplardandır — **en belirgin anksiyete örüntüsü** |
| **ASP+Greyfurt** | 13.9% | 17.0% | SAP yüksek ama grooming orta — açık kolda aktif keşif ile birlikte yorumlanmalı |

### Kutu grafiğin unsurları (her panel için aynı)

```
        ┌───────┐   ← 3. çeyrek (Q3)
        │       │
   ─────┤   ●   ├─────   ← Medyan (kalın çizgi) + bireysel sıçan (●)
        │       │
        └───────┘   ← 1. çeyrek (Q1)
           │
           │        ← Min (aykırı olmayan)
```

- **Kutunun yüksekliği (IQR):** Bireyler arası varyabilite — geniş kutu = tutarsız grup
- **Medyan çizgisi:** Grup merkezi eğilimi (n=3'te ortanca değer)
- **Noktalar:** Her biri gerçek bir sıçanı gösterir

---

## Bu Davranışlar Nasıl Birbirinden Ayrıldı?

Her iki davranış da "hareketsiz + özel poz" koşulunu paylaşır ama **farklı geometrik sinyaller** kullanır:

### Grooming tespiti

```
Sinyal 1:  min(sol ön pati → burun, sağ ön pati → burun) < 23 px
Sinyal 2:  body_center hızı < 0.50 px/kare
Süre:      ≥ 15 kare (0.5 s)
```

Buradaki temel fikir: grooming sırasında **ön patiler buruna gider**, bu mesafe dramatik biçimde kısalır.

### SAP tespiti

```
Sinyal 1:  burun → kuyruk_tabanı mesafesi > 102 px   (normal ~88 px)
Sinyal 2:  body_center hızı < 0.50 px/kare
Süre:      ≥ 15 kare (0.5 s)
```

Buradaki temel fikir: SAP sırasında **vücut öne doğru gerilir**, burun–kuyruk mesafesi artar.

### İki davranış çakışıyor mu?

Teorik olarak bir sıçan aynı anda hem uzanmış hem ön patisi burnunda olamaz — bu iki sinyal birbirini dışlar. Pratikte küçük bir örtüşme mümkündür (geçiş anları) ama minimum 0.5 s süre filtresi bunu büyük ölçüde engeller.

---

## Tespit Zincirinin Özeti (Her İki Davranış)

```
DLC CSV (30 fps, 10 keypoint)
         │
         ▼
  Keypoint yükle → likelihood < 0.6 olanlar NaN
         │
         ▼
  Her kare için:
    ┌─────────────────────────────────┐
    │ Grooming sinyali (0/1)          │
    │   forepaw-nose < 23px           │
    │   + hız < 0.5px/kare           │
    └─────────────────────────────────┘
    ┌─────────────────────────────────┐
    │ SAP sinyali (0/1)               │
    │   vücut_boyu > 102px            │
    │   + hız < 0.5px/kare           │
    └─────────────────────────────────┘
         │
         ▼
  Bout tespiti:
    - Kısa araliklar birleştir (< 0.33s)
    - Min süre filtrele (≥ 0.5s)
         │
         ▼
  Per-sıçan metrikler:
    bout_count, total_s, pct_time, mean_s, frag_idx
         │
         ▼
  Kohort bazında kutu grafik + CSV
```

---

## Eşiklerin Güvenilirliği

Eşikler 12 sıçanın havuzlanmış dağılımından türetilmiştir:

| Eşik | Değer | Dağılımda karşılığı |
|------|-------|---------------------|
| Forepaw-nose < 23 px | Grooming eşiği | Havuz p25 |
| Vücut boyu > 102 px | SAP eşiği | Havuz p75 |
| Hız < 0.50 px/kare | Hareketsizlik eşiği | Havuz p25–p50 arası |

Ham sinyal yüzdeleri (eşiğin kaç kareyi işaretlediği):

| Kohort | Grooming sinyal % | SAP sinyal % |
|--------|------------------|-------------|
| Control | 7.6% | 5.1% |
| Aspartame | 7.8% | 8.4% |
| Grapefruit | 10.5% | 9.8% |
| ASP+Greyfurt | 7.9% | 9.4% |

Bu değerler makul aralıktadır — çok yüksek olsaydı eşikler çok gevşek, çok düşük olsaydı çok katı demekti. Bout filtrelemesi sonrasındaki `pct_time` değerleri ham sinyal yüzdelerinden yüksek çıkabilir; bu beklenen bir durumdur çünkü birleştirme adımı kısa aralıkları doldurmaktadır.

---

## Tezde Nasıl Kullanılmalı?

Bu özet grafik tek başına Bulgular bölümüne konabilir; grooming ve SAP panelleri ayrı ayrı da sunulabilir.

**Önerilen altyazı:**

> Şekil X. Plus Maze oturumunda kohort bazında grooming (sol) ve SAP (sağ) davranışlarının oturum süresine oranı. Davranışlar DeepLabCut pose verisi üzerinden otomatik olarak tespit edilmiştir: grooming için minimum ön pati–burun mesafesi < 23 px + hız < 0.5 px/kare; SAP için burun–kuyruk tabanı mesafesi > 102 px + hız < 0.5 px/kare koşulu (en az 0.5 s) aranmıştır. Her nokta bir sıçanı temsil etmektedir (n=3/grup). Kutular IQR'yi, yatay çizgiler medyanı göstermektedir.

---

## Sınırlılıklar (Her İki Davranış İçin Geçerli)

1. **Manuel doğrulama yapılmamıştır** — eşikler hesaplamalı, video bazlı onay yoktur.
2. **n=3/grup** — istatistiksel karşılaştırma (Kruskal-Wallis vb.) gücü yetersiz; sonuçlar keşifseldir.
3. **Rearing kapsam dışıdır** — üstten kamerada vücut uzunluğu kısalmasıyla güvenilir biçimde tespit edilemediğinden dahil edilmemiştir.
4. **Grooming türleri ayırt edilmemiştir** — yüz grooming, vücut grooming ve genital grooming tek kategori olarak raporlanmaktadır.
