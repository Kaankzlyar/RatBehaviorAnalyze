"""
Window-classifier inference — bir DLC pose CSV'sinden frame-bazlı
davranış tahminleri ve bout listesi üretir.

Usage
-----
    python -m src.window_classifier.infer --csv data/DLCfiltered/.../<subject>.csv
    python -m src.window_classifier.infer --csv <path> --model lightgbm \\
                                          --out-dir results/predictions/

Output
------
    <out-dir>/<subject>_predicted_frames.csv
        frame, time_s, predicted_label, prob_<class>...
    <out-dir>/<subject>_predicted_bouts.csv
        start_frame, end_frame, start_s, end_s, behaviour,
        duration_s, n_frames

Detaylar
--------
Pencere tahminleri `predict_proba` ile alınıp her frame için, o frame'i
içeren tüm pencerelerin olasılıklarının ortalaması alınır → argmax
frame-bazlı etiketi verir. Bu yaklaşım LightGBM/XGBoost/RF için aynı
şekilde çalışır.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.window_classifier.features import (
    DEFAULT_WINDOW, DEFAULT_STRIDE, DEFAULT_LONG_WINDOW, LIKELIHOOD_THRESH,
    apply_likelihood_mask, compute_rule_signals, extract_windows,
    load_dlc_flat_raw,
)

ROOT     = Path(__file__).resolve().parent.parent.parent
MODELS   = ROOT / "models" / "window_classifier"
DEFAULT_FPS = 30.0


def load_model_bundle(model_name: str) -> dict:
    pkl = MODELS / f"{model_name.lower()}.pkl"
    if not pkl.exists():
        avail = sorted(p.stem for p in MODELS.glob("*.pkl"))
        raise FileNotFoundError(
            f"{pkl} bulunamadı. Mevcut modeller: {avail or 'yok — '}"
            "önce src/window_classifier/train.py'yi çalıştır."
        )
    with open(pkl, "rb") as f:
        return pickle.load(f)


def windows_to_frame_probs(window_probs: np.ndarray,
                           window_starts: np.ndarray,
                           window_ends: np.ndarray,
                           n_frames: int) -> np.ndarray:
    """Pencere olasılıklarını frame-bazlı olasılığa indirger.

    Pencereler örtüşür; her frame için kendisini içeren tüm pencerelerin
    olasılıklarının ortalaması alınır. Hiçbir pencerede olmayan frame'ler
    son geçerli olasılığı taşır (çoğunlukla session başı/sonu kenarları).
    """
    n_classes = window_probs.shape[1]
    accum = np.zeros((n_frames, n_classes), dtype=np.float64)
    counts = np.zeros(n_frames, dtype=np.int32)
    for prob, s, e in zip(window_probs, window_starts, window_ends):
        accum[s:e + 1] += prob
        counts[s:e + 1] += 1
    valid = counts > 0
    accum[valid] /= counts[valid, None]

    # kenarlar: counts==0 olan frame'leri en yakın geçerliye genişlet
    if (~valid).any():
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            # hiç pencere yok? olmaz ama yine de boşa varsayım
            accum[:, 0] = 1.0
        else:
            for i in np.flatnonzero(~valid):
                nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
                accum[i] = accum[nearest]
    return accum


def runs_to_bouts(frame_labels: np.ndarray, fps: float) -> pd.DataFrame:
    """Aynı etiketi taşıyan ardışık frame koşusunu bout'a dönüştürür."""
    if len(frame_labels) == 0:
        return pd.DataFrame(columns=[
            "start_frame", "end_frame", "start_s", "end_s",
            "behaviour", "duration_s", "n_frames",
        ])
    change = np.flatnonzero(frame_labels[1:] != frame_labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends   = np.concatenate((change - 1, [len(frame_labels) - 1]))
    rows = []
    for s, e in zip(starts, ends):
        rows.append({
            "start_frame": int(s),
            "end_frame":   int(e),
            "start_s":     round(s / fps, 3),
            "end_s":       round(e / fps, 3),
            "behaviour":   str(frame_labels[s]),
            "duration_s":  round((e - s + 1) / fps, 3),
            "n_frames":    int(e - s + 1),
        })
    return pd.DataFrame(rows)


def predict_one(csv: Path, bundle: dict, window: int, stride: int,
                lik_thresh: float, fps: float,
                long_window: int = DEFAULT_LONG_WINDOW
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = csv.parent.name
    print(f"[load] {csv.relative_to(ROOT) if csv.is_relative_to(ROOT) else csv}")
    raw = load_dlc_flat_raw(csv)
    dlc = apply_likelihood_mask(raw, lik_thresh)
    rule_sig = compute_rule_signals(raw, dlc, fps=fps)
    n_frames = len(dlc)
    print(f"  frames = {n_frames}  (~{n_frames/fps:.1f} s @ {fps:.0f} fps)")

    win_df = extract_windows(dlc, subject, window, stride,
                             long_window_size=long_window, fps=fps,
                             rule_signals=rule_sig)
    print(f"  windows = {len(win_df)}  (window={window}, stride={stride}, "
          f"long_window={long_window})")

    # feature matrix — eğitim sırasında kullanılan kolon sırasına göre hizala
    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in win_df.columns]
    if missing:
        raise ValueError(
            f"window_features çıktısında eksik kolonlar var: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} — "
            "extractor sürümü modelinkinden farklı olabilir."
        )
    X = win_df[feature_cols].values.astype(np.float32)
    X = bundle["imputer"].transform(X)

    proba = bundle["model"].predict_proba(X)
    le = bundle["label_encoder"]
    class_names = list(le.classes_)

    starts = win_df["window_start"].to_numpy(int)
    ends   = win_df["window_end"].to_numpy(int)
    frame_probs = windows_to_frame_probs(proba, starts, ends, n_frames)
    frame_pred_idx = frame_probs.argmax(axis=1)
    frame_pred = le.inverse_transform(frame_pred_idx)

    frames_df = pd.DataFrame({
        "frame":           np.arange(n_frames),
        "time_s":          np.round(np.arange(n_frames) / fps, 3),
        "predicted_label": frame_pred,
    })
    for ci, cn in enumerate(class_names):
        frames_df[f"prob_{cn}"] = np.round(frame_probs[:, ci], 4)

    bouts_df = runs_to_bouts(frame_pred, fps)
    return frames_df, bouts_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True,
                   help="DLC pose CSV (filtered)")
    p.add_argument("--model", default="lightgbm",
                   help="Model adı: randomforest | xgboost | lightgbm")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--long-window", type=int, default=DEFAULT_LONG_WINDOW,
                   dest="long_window",
                   help="Co-centred slow-context window size in frames; "
                        "must match training (default keeps in sync)")
    p.add_argument("--likelihood-thresh", type=float,
                   default=LIKELIHOOD_THRESH, dest="lik_thresh")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Tahmin dosyalarının yazılacağı klasör "
                        "(varsayılan: input CSV'nin yanı)")
    args = p.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    bundle = load_model_bundle(args.model)
    frames_df, bouts_df = predict_one(
        args.csv, bundle,
        window=args.window, stride=args.stride,
        lik_thresh=args.lik_thresh, fps=args.fps,
        long_window=args.long_window,
    )

    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    subject = args.csv.parent.name
    frames_path = out_dir / f"{subject}_predicted_frames.csv"
    bouts_path  = out_dir / f"{subject}_predicted_bouts.csv"
    frames_df.to_csv(frames_path, index=False)
    bouts_df.to_csv(bouts_path, index=False)

    rare = bouts_df[bouts_df["behaviour"].isin(("rearing", "grooming"))]
    summary = rare.groupby("behaviour").agg(
        n_bouts=("behaviour", "size"),
        total_s=("duration_s", "sum"),
        mean_s=("duration_s", "mean"),
    ).round(2)

    print(f"\n[write] {frames_path.name}  ({len(frames_df)} frame)")
    print(f"[write] {bouts_path.name}  ({len(bouts_df)} bout)")
    if len(summary):
        print("\n[özet] davranış-bazlı:")
        print(summary.to_string())
    else:
        print("\n[özet] modelin rearing/grooming bout'u tespit etmediği bir oturum.")


if __name__ == "__main__":
    main()
