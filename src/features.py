"""
Aşama 05 — Özellik Mühendisliği
Çıktılar:
  data/features/features_raw.csv
  data/features/features_normalized.csv
  data/features/labels.csv
  models/classifier/scaler.pkl
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
OUT  = DATA / "features"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "models" / "classifier").mkdir(parents=True, exist_ok=True)

# ── 1. Kaynak dosyaları yükle ────────────────────────────────────────────────

oft = pd.read_csv(DATA / "oft_metrics_all.csv")
beh = pd.read_csv(DATA / "behavior_summary.csv")

# beh'ten yalnızca oft'ta olmayan benzersiz sütunlar alınır
BEH_EXTRA = ["rear_mean_s", "rear_max_s", "rear_frag_idx",
             "groom_mean_s", "groom_max_s", "groom_frag_idx"]

df = oft.merge(
    beh[["subject_id"] + BEH_EXTRA],
    on="subject_id",
    how="left",
)

# artık gerekmeyen meta sütunları düşür
df = df.drop(columns=["n_frames", "bouts_csv_found"], errors="ignore")

# ── 2. Türetilmiş özellikler ─────────────────────────────────────────────────

df["locomotion_pct"]   = (100
                          - df["rear_pct_time"]
                          - df["groom_pct_time"]
                          - df["pct_time_freeze"]).clip(lower=0)

df["rear_per_min"]     = df["rear_bout_count"]  / (df["session_duration_s"] / 60)
df["groom_per_min"]    = df["groom_bout_count"] / (df["session_duration_s"] / 60)
df["exploration_ratio"] = df["center_zone_entries"] / (df["session_duration_s"] / 60)

# ── 3. Sütun sıralaması ──────────────────────────────────────────────────────

ID_COLS = ["subject_id", "cohort", "group", "session_duration_s"]

FEATURE_COLS = [
    # lokomotor
    "total_distance_px", "mean_speed_px_s", "max_speed_px_s",
    # uzamsal
    "pct_time_center", "pct_time_periphery", "center_zone_entries",
    "exploration_ratio", "spatial_entropy_norm",
    # donma
    "freeze_bout_count", "total_freeze_s", "pct_time_freeze",
    # rearing
    "rear_bout_count", "rear_per_min", "rear_total_s", "rear_pct_time",
    "rear_mean_s", "rear_max_s", "rear_frag_idx",
    # grooming
    "groom_bout_count", "groom_per_min", "groom_total_s", "groom_pct_time",
    "groom_mean_s", "groom_max_s", "groom_frag_idx",
    # genel aktivite
    "locomotion_pct",
]

features_raw = df[ID_COLS + FEATURE_COLS].copy()
features_raw.to_csv(OUT / "features_raw.csv", index=False)
print(f"[OK] features_raw.csv  — {features_raw.shape[0]} satir, {len(FEATURE_COLS)} ozellik")

# ── 4. Normalizasyon ─────────────────────────────────────────────────────────

scaler = StandardScaler()
scaled = scaler.fit_transform(features_raw[FEATURE_COLS])

features_norm = features_raw[ID_COLS].copy()
features_norm[FEATURE_COLS] = scaled
features_norm.to_csv(OUT / "features_normalized.csv", index=False)

with open(ROOT / "models" / "classifier" / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("[OK] features_normalized.csv + scaler.pkl")

# ── 5. Etiketler ─────────────────────────────────────────────────────────────

labels = features_raw[["subject_id", "cohort", "group"]].copy()

# 5a. anxiety_score — bileşik z-skor
def z(col):
    return stats.zscore(features_raw[col], ddof=1)

labels["anxiety_score"] = (
    z("pct_time_periphery")
    + z("pct_time_freeze")
    - z("center_zone_entries")
    - z("pct_time_center")
)

# 5b. anxiety_level — 3 sınıf
def score_to_level(s):
    if s < -0.5:
        return "low"
    elif s > 0.5:
        return "high"
    else:
        return "moderate"

labels["anxiety_level"] = labels["anxiety_score"].apply(score_to_level)

# 5c. rearing_profile — rear_pct_time eşiğine göre
def rear_profile(v):
    if v < 7:
        return "low"
    elif v <= 15:
        return "moderate"
    else:
        return "high"

labels["rearing_profile"] = features_raw["rear_pct_time"].apply(rear_profile)

# 5d. grooming_profile — groom_pct_time eşiğine göre
def groom_profile(v):
    if v < 4:
        return "low"
    elif v <= 10:
        return "moderate"
    else:
        return "high"

labels["grooming_profile"] = features_raw["groom_pct_time"].apply(groom_profile)

labels.to_csv(OUT / "labels.csv", index=False)
print("[OK] labels.csv")

# ── 6. Özet yazdır ───────────────────────────────────────────────────────────

print("\n=== OZELLIK TABLOSU ===")
print(features_raw[["subject_id", "group"] + FEATURE_COLS[:6]].to_string(index=False))

print("\n=== ETİKETLER ===")
print(labels.to_string(index=False))

print("\n=== SINIF DAGILIMI ===")
for col in ["anxiety_level", "rearing_profile", "grooming_profile"]:
    print(f"\n{col}:")
    print(labels.groupby("group")[col].value_counts().to_string())

# ── 7. OFT + Plus Maze birleşik feature tablosu ──────────────────────────────

PM_CSV = DATA / "plus_maze_metrics_all.csv"

PM_FEATURE_COLS = [
    "pct_open_arm", "pct_closed_arm",
    "pct_open_arm_entries", "pct_closed_arm_entries",
    "anxiety_index_epm",
    "total_entries", "mean_speed_px_s", "total_distance_px",
    "arm_preference_index",
    "successive_alternation_pct", "perseveration_rate_pct",
    "pct_time_left", "pct_time_right",
    "pct_time_top", "pct_time_bottom", "pct_time_junction",
]

if PM_CSV.exists():
    pm = pd.read_csv(PM_CSV)
    # subject_id eslestirmesi: "PlusMazeMA1_1" -> "MA1_1"
    pm["subject_id"] = pm["subject_id"].str.replace("PlusMaze", "", regex=False)
    pm_feats = pm[["subject_id"] + PM_FEATURE_COLS].copy()
    pm_feats = pm_feats.rename(
        columns={c: f"pm_{c}" for c in PM_FEATURE_COLS}
    )

    combined = features_raw.merge(pm_feats, on="subject_id", how="inner")
    all_feat_cols = FEATURE_COLS + [f"pm_{c}" for c in PM_FEATURE_COLS]

    combined[ID_COLS + all_feat_cols].to_csv(OUT / "features_combined.csv", index=False)
    print(f"\n[OK] features_combined.csv  — {len(combined)} satir, {len(all_feat_cols)} ozellik")

    scaler_comb = StandardScaler()
    scaled_comb = scaler_comb.fit_transform(combined[all_feat_cols])
    features_comb_norm = combined[ID_COLS].copy()
    features_comb_norm[all_feat_cols] = scaled_comb
    features_comb_norm.to_csv(OUT / "features_combined_normalized.csv", index=False)

    with open(ROOT / "models" / "classifier" / "scaler_combined.pkl", "wb") as f:
        pickle.dump(scaler_comb, f)

    print(f"[OK] features_combined_normalized.csv + scaler_combined.pkl")
    print(f"\nKohort dagilimi (combined):")
    print(combined.groupby("cohort")[["pct_time_center", "pct_time_freeze",
                                       "pm_pct_open_arm",
                                       "pm_anxiety_index_epm"]].mean().round(2).to_string())
else:
    print(f"\n[skip] {PM_CSV} bulunamadi — combined tablo atlanıyor")
