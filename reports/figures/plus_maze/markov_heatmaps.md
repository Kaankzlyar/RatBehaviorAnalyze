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

## Nasıl Okunur?

- **Yüksek diagonal değer** → hayvan aynı kolu tekrar seçiyor (kaşifçi değil, rutine bağlı)  
- **Yüksek L/R sütun değerleri** → hayvan açık kola geçiş yapıyor (daha az anksiyöz davranış)  
- **Düşük perseverasyon + yüksek L/R geçişi** → en iyi alternasyon profili (Grapefruit grubu bu yönde öne çıkıyor)  
- ASP+Greyfurt grubunun 0.59 perseverasyon ortalaması, diğer grupların belirgin üzerinde; bu grup daha az spontan alternasyon yapıyor

---

## İlişkili Dosyalar

- `reports/markov_transition_matrices.csv` — ham olasılık değerleri (uzun format)  
- `reports/markov_transition_counts.csv` — ham geçiş sayıları  
- `reports/figures/plus_maze/markov_perseveration.png` — kohort bazında perseverasyon / açık kol / kapalı kol bar chart karşılaştırması
