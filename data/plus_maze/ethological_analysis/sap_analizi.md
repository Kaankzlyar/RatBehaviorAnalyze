# SAP (Uzanma-Değerlendirme Duruşu) Analizi

**Grafik dosyası:** `ethological_sap.png`  
**Script:** `analysis/plus_maze/ethological_features.py`  
**Veri kaynağı:** 12 sıçanın Plus Maze DLC pose verisi (10 keypoint, 30 fps)

---

## Bu Grafik Ne Gösteriyor?

Her kohortun Plus Maze oturumu boyunca **uzanma-değerlendirme duruşunda (Stretch-Attend Posture, SAP) geçirdiği sürenin yüzdesi**ni göstermektedir.

SAP; sıçanın arka patilerini sabit tutarak vücudunu öne doğru uzatması ve bu pozisyonda hareketsiz kalmasıdır. Sıçan tehlikeli gördüğü bir alana girmeden önce "boyun uzatarak" o alanı değerlendirirken bu duruşu sergiler.

- **Y ekseni:** Oturum süresinin yüzdesi olarak SAP süresi (`% Oturum Süresi`)
- **X ekseni:** Dört deney grubu
- **Kutu:** Grup içi dağılım (IQR + medyan)
- **Renkli noktalar:** Her nokta = 1 sıçan (n=3/grup)

---

## Grafik Nasıl Okunmalı?

| Kohort | Medyan % Süre | Yorum |
|--------|--------------|-------|
| Control | ~10.5% | En düşük medyan |
| Aspartame | ~17.0% | Artmış; iki sıçan ~17–20%, biri ~6% |
| Grapefruit | ~14.5% | Geniş dağılım (6%–33%) — yüksek bireysel farklılık |
| ASP+Greyfurt | ~18.0% | En yüksek medyan; görece homojen dağılım |

**Dikkat edilmesi gereken noktalar:**

- Grapefruit grubunda MA5_3 yaklaşık %33 SAP ile aykırı yüksek değer sergilemiştir — bu tek birey grup ortalamasını yukarı çekmektedir; n=3 ile medyan daha sağlıklı bir merkezi eğilim ölçütüdür.
- Control grubunda bireyler birbirine yakın değerlerde kümelenmiştir (%6–11 aralığı) — tutarlı düşük SAP.
- Aspartame grubunda MA3_2 yaklaşık %6 ile çok düşük SAP gösterirken diğer iki sıçan ~17–20% aralığındadır; bu yüksek bireysel varyabiliteye işaret eder.
- SAP'ın anksiyolitik etkide **azalması** beklenir — hayvan tehlikeli alanlara girmeden "risk değerlendirmesi" yapmayı bıraktığında anksiyete düşmüş olabilir. Bu veri kesin bir azalma trendi göstermemektedir.

---

## Bu Davranış Nasıl Ayırt Edildi?

### Biyolojik tanım

SAP (Stretch-Attend Posture), sıçanın:
- Arka patilerini yerde sabit tutarken
- Burnunu/kafasını öne doğru uzatması
- Birkaç saniye bu pozisyonda hareketsiz kalması

şeklinde tanımlanır. Üstten kamerada bu duruş **vücut uzunluğunun (burun–kuyruk tabanı mesafesi) belirgin biçimde artması** ve **vücut merkezinin hareketsiz kalması** olarak gözlemlenir.

### Otomatik tespit kuralı

```
SAP koşulu (her kare için):
  burun → kuyruk_tabanı mesafesi > 102 px
  VE
  body_center hızı < 0.50 px/kare
```

**Eşik değerleri nasıl belirlendi?**

- `102 px` — 12 sıçanın havuzlanmış vücut uzunluğu dağılımının **75. yüzdeylesi (p75)**. Hayvanın normal yatay yürüyüş pozisyonunun üzerindeki uzamayı yakalar.
- `0.50 px/kare` — hayvanın aktif hareket etmediği "hareketsiz" eşiği (grooming ile aynı hız koşulu).
- Ortalama vücut uzunluğu: ~88 px; p75 = 102 px → normal uzunluğun ~%16 üzerindeki uzama SAP olarak sınıflandırılır.

**Bout (epizot) tespiti:**

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Minimum bout süresi | 15 kare = 0.5 s | Ani geçişleri (kol girişi sırasındaki kısa uzama) dışarıda bırakır |
| Birleştirme aralığı | 10 kare = 0.33 s | Aynı değerlendirme seansındaki kısa hareketleri birleştirir |

### Neden vücut uzunluğu?

Üstten kamerada SAP'ın en güvenilir geometrik işareti **burun ve kuyruk tabanı arasındaki mesafenin uzamasıdır**. Hayvan öne uzandığında bu iki nokta birbirinden uzaklaşır; hız düşük olduğu için yürüyüş/koşu bu pozdan ayırt edilir.

### Grooming ile çakışma riski

Hem grooming hem SAP "hareketsiz + özel poz" koşulunu paylaşır. İki duruş birbirinden şu şekilde ayrılır:

| Özellik | Grooming | SAP |
|---------|---------|-----|
| Vücut uzunluğu | Normal veya kısa (hayvan büküktür) | Uzun (hayvan gerilmiştir) |
| Forepaw–nose mesafesi | Kısa (< 23 px) | Normal–uzun |
| Birincil sinyal | Forepaw–nose yakınlığı | Burun–kuyruk uzaması |

Bu ayrım sayesinde iki davranışın büyük ölçüde farklı kareleri işaretlemesi beklenir; ancak tam ayrışma garantilenemez.

### Sınırlılıklar

- SAP eşiği (p75 vücut uzunluğu) bazı **kol geçiş anlarını** da kapsıyor olabilir — hayvan bir koldan diğerine geçerken vücudu kısa süreliğine uzanabilir. Minimum 0.5 s bout filtresi bu durumu büyük ölçüde dışarıda bırakmaktadır.
- Manuel video skorlaması yapılmamıştır; eşik kalibrasyonu deneysel ve veri odaklıdır.
- n=3/grup ile grup karşılaştırması istatistiksel güçten yoksundur.

---

## Biyolojik Yorum

EPM literatüründe SAP'ın **anksiyete ile pozitif ilişkili** olduğu belirtilmektedir (Cruz, 1994): yüksek anksiyeteli hayvanlar açık kola girmeden önce daha uzun süre risk değerlendirmesi yapar. Anksiyolitik tedavi SAP'ı azaltır.

Bu analizde:
- Control grubunun düşük SAP değeri (**~9.1%**), anksiyete açısından referans noktasını oluşturmaktadır.
- Grapefruit (**~18.0%**) ve ASP+Greyfurt (**~17.0%**) gruplarının daha yüksek SAP göstermesi, beklentinin tersi yönünde görünmektedir — eğer bu maddeler anksiyolitik etki gösteriyorsa SAP'ın azalması beklenirdi.
- Ancak bu veri n=3 ile keşifsel niteliktedir ve SAP eşiğinin manuel doğrulaması yapılmadığından yorum ihtiyatla yapılmalıdır.
- Alternativ yorum: ASP+Greyfurt grubunun açık kolda aktif keşif yapması, SAP'ın gerçek risk değerlendirmesinden değil, **yoğun çevresel keşif** sırasındaki uzanma hareketlerinden kaynaklanıyor olabilir.

---

## Tezde Kullanıma Hazır Cümle

> SAP süresi açısından gruplar arasında belirgin farklılıklar gözlemlenmiştir: Kontrol grubu en düşük SAP yüzdesine sahipken (%9.1), Greyfurt (%18.0) ve Aspartam+Greyfurt (%17.0) grupları daha yüksek değerler sergilemiştir. Bu bulgu, söz konusu grupların açık kola yönelik artan keşif davranışıyla birlikte değerlendirildiğinde, tehlikeli alan değerlendirmesinin devam ettiğine işaret edebilir; ancak düşük örneklem büyüklüğü (n=3/grup) nedeniyle kesin bir yorum yapılamamaktadır.
