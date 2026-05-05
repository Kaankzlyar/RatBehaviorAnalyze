# Synthetic DLC dataset — internal notes

Bu klasördeki CSV'ler `src/synthesize_keypoints.py` ile üretilmiştir. Demo
amaçlı. Asıl tezdeki bulgularda kullanılmamalıdır.

## Üretim yöntemi

Her MAT subject için:

1. `data/DLCfiltered/Kare/<group>/MA{k}-{n}_res.mat` dosyasından centroid
   trajectory (xc, yc) okunur.
2. Aynı kohorttan rastgele bir DLC subject "donor" olarak seçilir.
3. Donor'ın frame-frame keypoint offsetleri (her keypoint − body_center)
   hesaplanır.
4. MAT uzunluğuna time-warp yapılır.
5. `synth_keypoint[t] = mat_centroid[t] + warped_donor_offset[t]`

## Bilinmesi gerekenler

- Subject id'ler `OpenFieldMA{k}_{n}` formatında, gerçek DLC dosyalarıyla
  aynı isimlendirme — ama farklı dizinde (`synthetic_DLCfiltered/`).
- Bu dosyalar gerçek pose ölçümleri **değil** — DLC tarafından çekilmiş
  görünüyorlar ama keypoint pozisyonları başka farelerin kayıtlarından
  türetilmiş.
- Üretim parametreleri `synthesis_log.csv` dosyasında izlenir (her synth
  subject için: hangi MAT'tan, hangi donor DLC'den, kaç frame interpolate
  edildi).

## Branch ayrımı

Bu klasör sadece `experimental/synthetic-keypoints` branch'inde tutulur,
`develop` veya `main`'e merge edilmez. Demo bittikten sonra branch'ı
silebilir veya saklayabilirsin.
