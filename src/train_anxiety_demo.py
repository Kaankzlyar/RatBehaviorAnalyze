"""
Train a binary anxiety classifier for the educational demo.

Combines:
  - Real DLC subjects:  data/DLCfiltered/<group>/OpenField*/<subject>.csv
  - Synthetic subjects: data/synthetic_DLCfiltered/<group>/<subject>/<subject>.csv

For each subject, computes the 4 anxiety-axis features
(pct_time_center, pct_time_periphery, center_zone_entries, pct_time_freeze),
derives a binary "anxious vs calm" label from the sign of the standard
anxiety_score composite, and trains a RandomForest classifier with LOOCV
accuracy reported.

Usage
-----
    python -m src.train_anxiety_demo

Output
------
    models/anxiety_demo/anxiety_classifier.pkl
    models/anxiety_demo/scaler.pkl
    models/anxiety_demo/feature_columns.json
    models/anxiety_demo/training_log.csv
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src._metrics_minimal import compute_anxiety_features, load_dlc, ANXIETY_FEATURES

REAL_DIR = ROOT / "data" / "DLCfiltered"
SYNTH_DIR = ROOT / "data" / "synthetic_DLCfiltered"
MODEL_DIR = ROOT / "models" / "anxiety_demo"


def collect_subjects() -> pd.DataFrame:
    rows: list[dict] = []

    # real DLC subjects (skip Kare, only canonical pose CSVs)
    for csv in REAL_DIR.glob("*/*/*.csv"):
        if "Kare" in csv.parts:
            continue
        if csv.stem != csv.parent.name:
            continue
        if not csv.parent.name.startswith("OpenField"):
            continue
        try:
            dlc = load_dlc(csv)
            feats = compute_anxiety_features(dlc)
        except Exception as exc:
            print(f"  err real {csv.name}: {exc}")
            continue
        feats.update({"subject_id": csv.parent.name, "n_frames": len(dlc)})
        rows.append(feats)

    # synthetic subjects (if available)
    if SYNTH_DIR.exists():
        for csv in SYNTH_DIR.glob("*/*/*.csv"):
            if csv.stem != csv.parent.name:
                continue
            try:
                dlc = load_dlc(csv)
                feats = compute_anxiety_features(dlc)
            except Exception as exc:
                print(f"  err synth {csv.name}: {exc}")
                continue
            feats.update({"subject_id": csv.parent.name, "n_frames": len(dlc)})
            rows.append(feats)

    return pd.DataFrame(rows)


def main() -> None:
    df = collect_subjects()
    print(f"[load] {len(df)} subjects total")
    print(df[["subject_id", "n_frames"] + ANXIETY_FEATURES].to_string(index=False))

    df = df.dropna(subset=ANXIETY_FEATURES).reset_index(drop=True)
    print(f"[load] {len(df)} after dropping NaN-feature subjects")

    X = df[ANXIETY_FEATURES].to_numpy(dtype=float)

    # binary anxiety label from anxiety_score sign
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    z = (X - means) / np.where(stds > 0, stds, 1.0)

    idx = {f: ANXIETY_FEATURES.index(f) for f in ANXIETY_FEATURES}
    score = (
        z[:, idx["pct_time_periphery"]]
        + z[:, idx["pct_time_freeze"]]
        - z[:, idx["center_zone_entries"]]
        - z[:, idx["pct_time_center"]]
    )
    y = (score > 0).astype(int)
    print(f"[label] anxious={y.sum()} / calm={(y == 0).sum()}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                    class_weight="balanced", random_state=42)

    if 2 <= len(np.unique(y)) and len(df) > 2:
        scores = cross_val_score(model, Xs, y, cv=LeaveOneOut(), scoring="accuracy")
        print(f"[loocv] accuracy = {scores.mean():.3f} ± {scores.std():.3f}  (n={len(df)})")
    else:
        print("[loocv] skipped (not enough class diversity)")

    model.fit(Xs, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "anxiety_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(ANXIETY_FEATURES, f)

    out_log = df[["subject_id", "n_frames"] + ANXIETY_FEATURES].copy()
    out_log["anxiety_score"] = score
    out_log["label"] = ["anxious" if v else "calm" for v in y]
    out_log.to_csv(MODEL_DIR / "training_log.csv", index=False)

    print(f"[save] {MODEL_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
