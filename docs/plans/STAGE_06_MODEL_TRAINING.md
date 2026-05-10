# Stage 06 — Model Eğitimi
## Plan & Roadmap

---

## Objective

Stage 05'te oluşturulan davranışsal özellik veri setinden denetimli ML modelleri eğit ve değerlendir. Birincil hedef: **4 grup sınıflandırması** (Kontrol / ASP / Greyfurt / ASP & Greyfurt). İkincil hedef: anksiyete seviyesi ve rearing/grooming profil etiketleri.

---

## Status

- [ ] Başlanmadı — Stage 05 özellik veri seti gerektirir

---

## Veri Seti Kısıtları

| Özellik | Değer |
|---------|-------|
| Toplam örnek | 12 (4 hayvan × 3 seans) |
| Özellikler | ~35–45 davranışsal metrik |
| Birincil hedef | 4 sınıf (grup etiketi) |
| İkincil hedefler | ikili: anksiyete_seviyesi, rearing_profili, grooming_profili |
| Kritik bölünme riski | Aynı hayvanın farklı seansları arasında veri sızıntısı |

> **Önemli:** Bölünme hayvan bazında yapılmalı, seans bazında değil. Aynı hayvanın farklı seansları hem eğitim hem test kümesinde bulunursa veri sızıntısı oluşur.

---

## Görevler

### 6.1 — Veri Hazırlama

- [ ] `data/features/features_normalized.csv` ve `data/features/labels.csv` yükle
- [ ] **Hayvan bazlı çapraz doğrulama** uygula (Leave-One-Rat-Out = LORO-CV)
  - 4 fold: her fold bir hayvanın 3 seansını test için ayırır
- [ ] Her etiket için sınıf dengesini kontrol et — %25'in altındaki sınıfı bayrakla

```python
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd

features = pd.read_csv("data/features/features_normalized.csv")
labels   = pd.read_csv("data/features/labels.csv")

X = features.drop(columns=["rat_id", "session", "group"]).values
y_group = labels["group"].values
groups  = features["rat_id"].values  # hayvan bazlı bölünme

logo = LeaveOneGroupOut()

for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y_group, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_group[train_idx], y_group[test_idx]
    # model eğit + değerlendir
```

### 6.2 — Temel Modeller (Birincil: Grup Sınıflandırması)

`src/model.py` içinde uygula. LORO-CV ile değerlendir.

| Model | Kütüphane | Notlar |
|-------|-----------|--------|
| Random Forest | `sklearn` | Temel model + özellik önemi |
| Gradient Boosting (XGBoost) | `xgboost` | Küçük tablo verisi için en iyi |
| SVM (RBF kernel) | `sklearn` | Küçük veri setleri için güçlü |
| Lojistik Regresyon (çok sınıflı) | `sklearn` | Yorumlanabilir temel |

- [ ] Her modeli 4 LORO fold üzerinde eğit
- [ ] 4 sınıflı görev için: ortalama doğruluk, F1 (makro + ağırlıklı), AUC-OvR raporu
- [ ] En iyi modeli her etiket için `models/classifier/`'a kaydet

### 6.3 — İkincil Etiket Modelleri

`anksiyete_seviyesi`, `rearing_profili`, `grooming_profili` için ikili sınıflandırıcılar:

```python
secondary_labels = ['anksiyete_seviyesi', 'rearing_profili', 'grooming_profili']

for label in secondary_labels:
    y = labels[label].values
    # Yukarıdaki aynı LORO-CV döngüsünü uygula
```

### 6.4 — Derin Öğrenme Modelleri (Veri Seti Artırılırsa)

n=12 ile derin öğrenme ancak artırma veya deneme düzeyinde veri (n ≈ 120) ile mümkün.

| Model | Girdi | Kullanım Durumu |
|-------|-------|-----------------|
| LSTM / GRU | Seans başına deneme dizisi | Deneme ilerleyişindeki zamansal örüntü |
| CNN | Occupancy heatmap numpy dizisi | Görüntüden uzaysal sınıflandırma |

- [ ] LSTM: her seans için deneme bazında T-maze metrik dizisini girdi olarak kullan
- [ ] CNN: `data/figures/heatmaps/` dizinlerinden üst üste bindirilmiş heatmap dizilerini kullan

### 6.5 — Hiperparametre Ayarlama

- [ ] En yüksek performanslı modeller için Optuna veya `GridSearchCV` kullan
- [ ] Hiperparametre ayarlamasını CV döngüsü içinde yap (test seti kirlenmesi olmasın)
- [ ] Tüm denemeleri `reports/hyperparam_search.csv`'e kaydet

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth    = trial.suggest_int("max_depth", 2, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    scores = cross_val_score(model, X_train, y_train, cv=logo, groups=groups_train,
                              scoring="f1_macro")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 6.6 — Değerlendirme

Her model × etiket kombinasyonu için:

- [ ] Doğruluk (Accuracy)
- [ ] F1-score (makro, ağırlıklı)
- [ ] AUC-ROC (OvR çok sınıflı)
- [ ] Karışıklık matrisi (hayvan ID'leriyle)
- [ ] SHAP özellik önemi (model başına top-10 özellik)

```python
import shap

explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

**Davranış özelliklerinin katkısı:** SHAP çıktısında `rearing_in_secim_pct`, `grooming_pct`, `davranis_kaygisi` gibi Stage 02B kaynaklı özelliklerin önemini izle.

### 6.7 — Model Karşılaştırma Raporu

- [ ] Karşılaştırma tablosu: tüm modeller × tüm etiketler × tüm metrikler
- [ ] Etiket başına en iyi modeli belirle
- [ ] En prediktif özellikleri raporla (SHAP top-5)
- [ ] Öğrenme eğrilerini görselleştir (LORO fold'lar üzerinde eğitim vs doğrulama skoru)

### 6.8 — Son Modelleri Kaydet

- [ ] `joblib.dump()` ile etiket başına en iyi modeli kaydet
- [ ] Özellik adları listesini kaydet (çıkarım girdi sırası eşleşmeli)
- [ ] Model versiyon ve eğitim tarihini belgele: `models/classifier/model_card.md`

---

## Değerlendirme Hedefleri

| Metrik | Minimum | İyi |
|--------|---------|-----|
| Doğruluk | > 0.65 | > 0.80 |
| F1-score (makro) | > 0.60 | > 0.75 |
| AUC-ROC (OvR) | > 0.70 | > 0.85 |

> Not: n=12 ile fold'lar arası yüksek varyans bekleniyor. Nokta tahminleri değil, eğilimleri yorumla.

---

## Çıktı Dosyaları

| Yol | Açıklama |
|-----|----------|
| `models/classifier/rf_group.pkl` | 4 grup RF modeli |
| `models/classifier/xgb_anksiyete.pkl` | Anksiyete etiketi XGBoost modeli |
| `models/classifier/scaler.pkl` | Özellik ölçekleyicisi |
| `models/classifier/feature_names.json` | Çıkarım için özellik sırası |
| `reports/model_comparison.csv` | Tüm model × metrik sonuçları |
| `reports/figures/shap_*.png` | SHAP önem grafikleri |
| `src/model.py` | Eğitim ve değerlendirme betiği |

---

## Sonraki Adım

→ **Stage 07:** `STAGE_07_REPORTING.md`
