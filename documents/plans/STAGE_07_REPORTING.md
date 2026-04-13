# Stage 07 — Psikolojik Analiz & Raporlama
## Plan & Roadmap

---

## Objective

Tüm pipeline çıktılarını yapılandırılmış bir analiz raporu olarak sentezle: hayvan başına davranışsal profiller, grup karşılaştırmaları (Kontrol / ASP / Greyfurt / ASP & Greyfurt), davranış sınıflandırma (rearing / grooming) bulguları, model performansı ve tez yazımı için hazır psikolojik durum yorumları.

---

## Status

- [ ] Başlanmadı — Stage 02B, 03A, 03B, 04, 05, 06 çıktıları gerektirir

---

## Görevler

### 7.1 — Hayvan Başına Davranışsal Profiller

Her hayvan (MA1, MA3, MA5, MA7) için tek sayfalık özet:

- [ ] OFT occupancy heatmap (3 seans yan yana)
- [ ] OFT rearing + grooming heatmap (3 seans yan yana)
- [ ] OFT metrik eğilimleri seans boyunca (çizgi grafikleri)
- [ ] T-maze dönüş yanlılığı ve rota verimliliği seans boyunca
- [ ] T-maze seçim zonu rearing % seans boyunca
- [ ] Stage 06 tahminlenen psikolojik durum etiketleri

```
reports/profiles/MA1_profile.pdf   (Kontrol)
reports/profiles/MA3_profile.pdf   (ASP)
reports/profiles/MA5_profile.pdf   (Greyfurt)
reports/profiles/MA7_profile.pdf   (ASP & Greyfurt)
```

### 7.2 — Grup Karşılaştırma İstatistikleri

- [ ] Her özellik için Kruskal-Wallis testi (4 grup, n=3/grup)
- [ ] Bonferroni düzeltmeli Dunn post-hoc testi
- [ ] Etki büyüklüklerini raporla (eta-kare veya rank-biserial r)
- [ ] Tablo üret: özellik × p-değeri × etki büyüklüğü × yön

```python
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

groups_map = {
    'Kontrol': 'MA1', 'ASP': 'MA3',
    'Greyfurt': 'MA5', 'ASP_Greyfurt': 'MA7'
}

results = []
for col in feature_cols:
    vals = [features[features['group'] == g][col].values for g in groups_map]
    stat, p = kruskal(*vals)
    dunn = posthoc_dunn(vals, p_adjust='bonferroni')
    results.append({'feature': col, 'kruskal_p': p, 'dunn': dunn.values.tolist()})
```

**Öncelikli test edilecek hipotezler:**

| Hipotez | İlgili Özellikler |
|---------|-------------------|
| ASP grubu daha yüksek anksiyete | `peripheral_time_ratio`, `decision_latency_mean`, `davranis_kaygisi` |
| Greyfurt grubu farklı rearing profili | `rearing_pct`, `rearing_in_secim_pct`, `rearing_context_ratio` |
| ASP & Greyfurt additif etki | Tüm özellikler: ASP + Greyfurt arasında mı, dışında mı? |
| Grooming anksiyetenin alternatif göstergesi | `grooming_pct`, `grooming_in_peripheral_pct` |

### 7.3 — Çapraz Arena Korelasyon Matrisi

- [ ] Tüm OFT ve T-maze özellikleri arasında Spearman korelasyonları hesapla
- [ ] Açıklamalı seaborn clustermap üret
- [ ] Önemli korelasyonları vurgula (düzeltme sonrası p < 0.05)
- [ ] Temel hipotez kontrolü: `oft_peripheral_time ↔ tmaze_decision_latency`
- [ ] Davranış köprüsü: `oft_rearing_pct ↔ tmaze_rearing_in_secim_pct`

```python
import seaborn as sns
corr = features[feature_cols].corr(method="spearman")
g = sns.clustermap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                   center=0, figsize=(16, 16))
g.savefig("reports/figures/statistics/correlation_matrix.png", dpi=300)
```

### 7.4 — Heatmap Galerisi

**OFT heatmap'leri:**
- [ ] 4 grup × 3 seans için occupancy heatmap'leri (12 görsel)
- [ ] 4 grup × rearing heatmap'leri (grup ortalamaları)
- [ ] 4 grup × grooming heatmap'leri (grup ortalamaları)

**T-maze heatmap'leri:**
- [ ] 4 grup × 3 seans için occupancy heatmap'leri
- [ ] 4 grup × rearing heatmap'leri — seçim zonunda yoğunlaşma var mı?
- [ ] 4 grup × grooming heatmap'leri

**Karşılaştırma panelleri:**
- [ ] Seans 1 vs Seans 3 yan yana (öğrenme etkisi)
- [ ] Kontrol vs ASP vs Greyfurt vs ASP&Greyfurt yan yana

```
reports/figures/heatmaps/
├── open_field/
│   ├── group_all_occupancy.png        # 4-panel
│   ├── group_rearing.png              # 4-panel
│   └── group_grooming.png             # 4-panel
└── tmaze/
    ├── group_all_occupancy.png        # 4-panel
    ├── group_rearing.png              # 4-panel → seçim zonu yoğunluğu
    └── group_grooming.png             # 4-panel
```

### 7.5 — Öğrenme Eğrisi Analizi

- [ ] Her davranışsal metriği seans 1→2→3 boyunca çiz (hayvan başına)
- [ ] Bireysel hayvanların + grup ortalaması ± SD'nin üst üste bindirilmesi
- [ ] Seans-içi önemli değişiklikleri işaretle (Wilcoxon işaret-rank testi)
- [ ] **Yeni:** rearing ve grooming eğilimleri seans boyunca (olağandışı değişim var mı?)

### 7.6 — Karar Yanlılığı Haritaları

- [ ] Her hayvan için seans başına Sol/Sağ seçim çubuğu grafiği
- [ ] Grup düzeyi: koşul başına sol/sağ seçeneklerin oranı
- [ ] Lateralizasyon yanlılığı testi (iki terimli test: p(sol) ≠ 0.5)

### 7.7 — Davranış Zaman Çizelgesi (Ethogram Görselleştirmesi)

- [ ] Seans başına örnek ethogram zaman çizelgesi çiz (rearing/grooming/locomotion renk kodlu)
- [ ] Grup başına ortalama bout sıklığı çubuğu grafiği
- [ ] Bout süresi dağılımı (violin/box grafikleri, 4 grup)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_ethogram(ethogram: pd.DataFrame, fps: int = 25,
                  title: str = "", save_path: str = None):
    """
    Tam seans ethogramı: x=zaman(s), y=davranış kanalları
    """
    time_axis = np.arange(len(ethogram)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(14, 4), sharex=True)
    behavior_colors = {
        'rearing':    '#F44336',
        'grooming':   '#4CAF50',
        'locomotion': '#2196F3',
    }

    for ax, (behavior, color) in zip(axes, behavior_colors.items()):
        mask = ethogram[behavior].values if behavior in ethogram.columns else \
               (~ethogram['rearing'] & ~ethogram['grooming']).values
        ax.fill_between(time_axis, 0, mask.astype(int), color=color, alpha=0.7)
        ax.set_ylabel(behavior, fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])

    axes[-1].set_xlabel("Zaman (s)")
    fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
```

### 7.8 — Model Performans Özeti

- [ ] Tablo: model × etiket × doğruluk × F1 × AUC
- [ ] En iyi model için SHAP özet grafiği
- [ ] Etiket başına karışıklık matrisleri (hayvan ID'leriyle TP/FP/FN/TN)
- [ ] SHAP'ta davranış özelliklerinin katkısını vurgula:
  - `rearing_in_secim_pct`, `davranis_kaygisi`, `rearing_context_ratio`

### 7.9 — Tez Rapor Yapısı

```
reports/
├── figures/
│   ├── heatmaps/
│   │   ├── open_field/         # Genel + rearing + grooming heatmap'leri
│   │   └── tmaze/              # Genel + rearing + grooming heatmap'leri
│   ├── trajectories/           # Davranış renkli rota grafikleri
│   ├── ethograms/              # Seans zaman çizelgeleri
│   ├── statistics/             # Grup karşılaştırma grafikleri
│   ├── learning_curves/        # Seans-seans eğilimler
│   └── model/                  # SHAP, karışıklık matrisleri, ROC eğrileri
├── tables/
│   ├── oft_metrics_summary.csv
│   ├── tmaze_metrics_summary.csv
│   ├── behavior_bout_stats.csv    # Rearing/grooming bout istatistikleri
│   ├── group_comparisons.csv
│   └── model_comparison.csv
└── profiles/
    ├── MA1_profile.pdf    (Kontrol)
    ├── MA3_profile.pdf    (ASP)
    ├── MA5_profile.pdf    (Greyfurt)
    └── MA7_profile.pdf    (ASP & Greyfurt)
```

**Rapor bölümleri:**

1. **Deney Özeti** — veri seti, koşullar, arena türleri, gruplar
2. **Grup Demografisi** — hayvan ID'leri, gruplar, seans sayıları
3. **Davranış Tespiti** — rearing/grooming metodolojisi (02B), kalibrasyon sonuçları
4. **OFT Heatmap Galerisi** — genel + rearing + grooming, grup karşılaştırması
5. **T-Maze Heatmap & Rota Analizi** — occupancy, rota çizgileri, zon occupancy
6. **Davranış Zaman Çizelgeleri** — ethogram örnekleri, bout istatistikleri
7. **Çapraz Arena Korelasyonlar** — OFT ↔ T-maze ↔ Rearing/Grooming ilişkileri
8. **Özellik Dağılımları & İstatistikler** — kutu grafikleri, grup karşılaştırmaları
9. **Model Performans Raporu** — doğruluk, F1, SHAP
10. **Psikolojik Durum Sınıflandırmaları** — hayvan başına tahminlenen etiketler
11. **Sonuçlar** — davranışsal yorum, sınırlamalar, gelecek çalışma

### 7.10 — Notebook Sonlandırma

- [ ] `01_dlc_analysis.ipynb` — DLC QC ve tracking görselleştirme
- [ ] `02_behavior_classification.ipynb` — Rearing/grooming tespiti, kalibrasyon (02B)
- [ ] `03_heatmap_generation.ipynb` — Tüm heatmap çıktıları, davranış katmanlı
- [ ] `04_path_analysis.ipynb` — T-maze metrik kılavuzu, davranış-deneme entegrasyonu
- [ ] `05_feature_engineering.ipynb` — Özellik türetimi ve korelasyon analizi
- [ ] `06_model_training.ipynb` — Model eğitimi, değerlendirme, SHAP yorumu

---

## Kabul Kriterleri

- 4 hayvan için hayvan başına davranışsal profiller üretilmiş
- Rearing ve grooming bout istatistiklerini içeren grup karşılaştırma tablosu (p-değerleri ile)
- En az 3 anlamlı çapraz arena özellik korelasyonu olan korelasyon matrisi
- En az 2 model × 3 etiket için model performans tablosu
- Tüm görseller ≥ 300 DPI baskı kalitesinde dışa aktarılmış

---

## Çıktı Dosyaları

| Yol | Açıklama |
|-----|----------|
| `reports/figures/` | Tüm yayın kalitesinde görseller |
| `reports/tables/` | Özet istatistik tabloları |
| `reports/profiles/` | Hayvan başına davranışsal profil PDF'leri |
| `notebooks/*.ipynb` | Son analiz notebook'ları |

---

## Notlar

- Tüm analizler özellik çıkarımı sırasında koşul etiketlerine kör yapılmalı
- Model psikolojik durum etiketleri davranışsal proxy'lerdir — klinik yorumlamadan önce uzman doğrulaması gerekir
- n=12 ile korelasyonların düşük istatistiksel gücü var — hipotez oluşturucu olarak ele al, doğrulayıcı değil
- Rearing/grooming tespiti için kalibrasyon eşik değerlerini ve doğrulama metriklerini (F1, precision, recall) raporda belgele
