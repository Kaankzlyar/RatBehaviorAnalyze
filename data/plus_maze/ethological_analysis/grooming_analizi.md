# Grooming (Kendini Temizleme) Analizi

**Grafik dosyası:** `ethological_grooming.png`  
**Script:** `analysis/tmaze/ethological_features.py`  
**Veri kaynağı:** 12 sıçanın Plus Maze DLC pose verisi (10 keypoint, 30 fps)

---

## Bu Grafik Ne Gösteriyor?

Her kohortun (grup) Plus Maze oturumu boyunca **kendini temizleme davranışına ayırdığı sürenin yüzdesi**ni göstermektedir.

- **Y ekseni:** Oturum süresinin yüzdesi olarak grooming süresi (`% Oturum Süresi`)
- **X ekseni:** Dört deney grubu (Control, Aspartame, Grapefruit, ASP+Greyfurt)
- **Kutu (box):** Her gruptaki 3 sıçanın dağılımı — kutunun alt ve üst kenarları 1. ve 3. çeyrek (IQR), ortadaki kalın çizgi medyan
- **Bıyıklar (whisker):** Aykırı olmayan min–maks değerleri
- **Renkli noktalar:** Her nokta = 1 sıçan (n=3/grup)

---

## Grafik Nasıl Okunmalı?

| Kohort | Medyan % Süre | Yorum |
|--------|--------------|-------|
| Control | ~7.0% | En düşük — baseline düzey |
| Aspartame | ~13.5% | Artmış; orta düzey |
| Grapefruit | ~23.5% | **En yüksek** — belirgin artış |
| ASP+Greyfurt | ~12.8% | Orta; Aspartame'e benzer |

**Dikkat edilmesi gereken noktalar:**

- Grapefruit grubunda hem medyan değeri hem de tüm bireyler Control grubunun üzerindedir — bu tutarlı bir grup etkisine işaret eder.
- Aspartame grubunda kutu geniştir (bireyler arası farklılık yüksek): MA3_2 yaklaşık %25 ile aykırı yüksek değer sergilemiş, MA3_1 ise %8.6 ile en düşük kalmıştır.
- ASP+Greyfurt grubunun üç sıçanı birbirine çok yakın değerlerde kümelenmiştir (%12–17 aralığı) — düşük bireysel varyabilite.
- Grapefruit grubundaki bir sıçan (~%6.5) oldukça düşük grooming göstermiş; bu nedenle kutunun alt bıyığı uzundur.

---

## Bu Davranış Nasıl Ayırt Edildi?

### Biyolojik tanım

Grooming (kendini temizleme), sıçanların ön patilerini yüzlerine götürerek strobing hareketleriyle yıkama yapması ve/veya vücutlarını dilleriyle ya da patileriyle taramasıdır. Üstten kamerada karakteristik görünümü: **ön patilerin buruna/başa çok yaklaşması** ve vücudun **hareketsiz kalması**.

### Otomatik tespit kuralı

```
GROOMING koşulu (her kare için):
  min(sol_ön_pati → burun, sağ_ön_pati → burun) < 23 px
  VE
  body_center hızı < 0.50 px/kare
```

**Eşik değerleri nasıl belirlendi?**

- `23 px` — 12 sıçanın tüm karelerinin havuzlandığı forepaw–nose mesafesi dağılımının **25. yüzdeylesi (p25)**. Yani sıçanın patisi burnunun yakınında olduğu anlara karşılık gelir.
- `0.50 px/kare` — hız p25–p50 arasında seçilmiş; hayvan yürürken grooming yapmaz.

**Bout (epizot) tespiti:**

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Minimum bout süresi | 15 kare = 0.5 s | Kısa temaslı geçişleri (pati kaza ile yaklaşması) dışarıda bırakır |
| Birleştirme aralığı | 10 kare = 0.33 s | Aynı grooming seansındaki kısa kesintileri birleştirir |

### Neden forepaw–nose mesafesi?

DLC modeli 10 keypoint takip etmektedir. Grooming sırasında en ayırt edici geometrik değişim **ön patilerin burna olan mesafesinin dramatik biçimde azalmasıdır**. Vücut merkezi (body_center) hareketsiz kalırken bu mesafenin düşmesi, sıçanın yürüyüp geçtiği değil gerçekten temizlendiğini gösterir.

### Sınırlılıklar

- Eşikler veri odaklı (data-driven) olup **manuel video doğrulaması yapılmamıştır**.
- Sıçanın dar bir köşeye sıkışıp ön patisini duvara dayadığı anlarda yanlış pozitif oluşabilir.
- n=3/grup ile istatistiksel karşılaştırma gücü düşüktür; bulgular keşifsel niteliktedir.

---

## Biyolojik Yorum

Artmış grooming, EPM literatüründe **artmış anksiyete yükünün davranışsal göstergesi** olarak yorumlanır (Cruz, 1994; Sturman, 2020). İki yorumlama mekanizması önerilmektedir:

1. **Yyer değiştirme davranışı (displacement):** Hayvan kapalı alanda sıkışıp kalan enerjisini grooming yoluyla boşaltır.
2. **Stereotipi:** Kronik stres/anksiyete durumunda repetitif grooming artabilir.

Grapefruit grubunun yüksek grooming değeri, aynı grubun kapalı kol tercihiyle (Markov analizinde kapalı kola geçiş oranı: 0.972) tutarlıdır. Bu iki bulgu birlikte değerlendirildiğinde Grapefruit grubunun **en yüksek anksiyete örüntüsüne** sahip olduğuna işaret etmektedir.

---

## Tezde Kullanıma Hazır Cümle

> Greyfurt grubunun oturum süresinin %19.3'ünü grooming davranışına ayırdığı saptanmıştır; bu değer Kontrol (%10.4), Aspartam (%15.6) ve Aspartam+Greyfurt (%13.9) gruplarının üzerindedir. Artmış grooming, anksiyete yükünün davranışsal bir göstergesi olarak yorumlanabilir (Cruz, 1994).
