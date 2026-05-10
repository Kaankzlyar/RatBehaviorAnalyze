"""
Window-classifier prediction vs rule-based ground-truth karşılaştırması.

For one or more session(s) bu modül:
  1. (opsiyonel) modeli çağırıp predicted_frames.csv üretir
  2. <subject>_behavior_frames.csv (rule-based) ile aynı uzunluğa hizalar
  3. frame-bazlı confusion matrix + classification report basar
  4. <out-dir>/<subject>_compare_frames.csv (frame, true, predicted, prob_*)
     ve <out-dir>/<subject>_compare_cm.png yazar

Usage
-----
    # tek session, mevcut prediction CSV'si üzerinden
    python -m src.window_classifier.compare \\
        --pred results/test_inference/OpenFieldMA1_1_predicted_frames.csv

    # birden çok subject, eksikse otomatik infer
    python -m src.window_classifier.compare \\
        --subjects OpenFieldMA1_1 OpenFieldMA1_2 OpenFieldMA1_3 \\
        --model lightgbm --out-dir results/test_inference --auto-infer

Not
---
"Ground truth" olarak rule-based detector çıktısı kullanılır
(<subject>_behavior_frames.csv). Insan etiketi değildir; bu yüzden
karşılaştırma "model vs heuristic" olarak okunmalıdır.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

ROOT    = Path(__file__).resolve().parent.parent.parent
DLC_DIR = ROOT / "data" / "DLCfiltered"


def find_subject_paths(subject: str) -> tuple[Path, Path]:
    """<subject>.csv (DLC) ve <subject>_behavior_frames.csv (GT) konumlarını bulur."""
    dlc_matches = list(DLC_DIR.glob(f"*/{subject}/{subject}.csv"))
    gt_matches  = list(DLC_DIR.glob(f"*/{subject}/{subject}_behavior_frames.csv"))
    if not dlc_matches:
        raise FileNotFoundError(f"DLC CSV bulunamadı: {subject}.csv")
    if not gt_matches:
        raise FileNotFoundError(
            f"Ground-truth bulunamadı: {subject}_behavior_frames.csv"
        )
    return dlc_matches[0], gt_matches[0]


def maybe_run_inference(subject: str, model: str, out_dir: Path,
                        force: bool = False) -> Path:
    """predicted_frames.csv yoksa ya da --force ise infer.predict_one'ı çağırır."""
    pred_path = out_dir / f"{subject}_predicted_frames.csv"
    if pred_path.exists() and not force:
        return pred_path

    # lazy import — infer modülü ağır bağımlılıklar getiriyor
    from src.window_classifier.infer import (
        DEFAULT_FPS, load_model_bundle, predict_one,
    )
    from src.window_classifier.features import (
        DEFAULT_WINDOW, DEFAULT_STRIDE, DEFAULT_LONG_WINDOW, LIKELIHOOD_THRESH,
    )

    dlc_csv, _ = find_subject_paths(subject)
    bundle = load_model_bundle(model)
    frames_df, bouts_df = predict_one(
        dlc_csv, bundle,
        window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE,
        lik_thresh=LIKELIHOOD_THRESH, fps=DEFAULT_FPS,
        long_window=DEFAULT_LONG_WINDOW,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_df.to_csv(pred_path, index=False)
    bouts_df.to_csv(out_dir / f"{subject}_predicted_bouts.csv", index=False)
    print(f"[infer] {pred_path.name}  ({len(frames_df)} frame)")
    return pred_path


def load_pred_and_gt(pred_csv: Path, gt_csv: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    gt   = pd.read_csv(gt_csv)
    n = min(len(pred), len(gt))
    if len(pred) != len(gt):
        print(f"  [warn] uzunluk uyumsuz: pred={len(pred)} gt={len(gt)} → {n}'e kırpılıyor")
    df = pd.DataFrame({
        "frame":     pred["frame"].values[:n],
        "time_s":    pred["time_s"].values[:n],
        "true":      gt["behaviour"].astype(str).values[:n],
        "predicted": pred["predicted_label"].astype(str).values[:n],
    })
    for col in pred.columns:
        if col.startswith("prob_"):
            df[col] = pred[col].values[:n]
    return df


def plot_confusion_matrix(cm: np.ndarray, labels: list[str],
                          title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # raw + normalize edilmiş yüzde annotation
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums > 0)
    threshold = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]*100:.1f}%)",
                    ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate(subject: str, df: pd.DataFrame, out_dir: Path) -> dict:
    classes = sorted(set(df["true"]) | set(df["predicted"]))
    cm = confusion_matrix(df["true"], df["predicted"], labels=classes)
    acc = accuracy_score(df["true"], df["predicted"])
    macro_f1    = f1_score(df["true"], df["predicted"], labels=classes,
                           average="macro", zero_division=0)
    weighted_f1 = f1_score(df["true"], df["predicted"], labels=classes,
                           average="weighted", zero_division=0)

    report = classification_report(
        df["true"], df["predicted"], labels=classes,
        digits=4, zero_division=0,
    )

    print(f"\n=== {subject} ===")
    print(f"  n_frames={len(df)}  acc={acc:.4f}  "
          f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")
    print(report)

    # bout sayısı (basit run-length): true vs predicted
    def bout_counts(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {}
        change = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends   = np.concatenate((change, [len(arr)]))
        out: dict[str, int] = {}
        for s, e in zip(starts, ends):
            out[arr[s]] = out.get(arr[s], 0) + 1
        return out

    true_bouts = bout_counts(df["true"].to_numpy())
    pred_bouts = bout_counts(df["predicted"].to_numpy())
    print("  bouts (true → pred):")
    for c in classes:
        print(f"    {c:9s}  {true_bouts.get(c, 0):3d}  →  "
              f"{pred_bouts.get(c, 0):3d}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{subject}_compare_frames.csv", index=False)
    plot_confusion_matrix(
        cm, classes,
        title=f"{subject} — frame-level (acc={acc:.3f}, macroF1={macro_f1:.3f})",
        out_path=out_dir / f"{subject}_compare_cm.png",
    )

    row = {"subject": subject, "n_frames": len(df),
           "accuracy": round(acc, 4),
           "macro_f1": round(macro_f1, 4),
           "weighted_f1": round(weighted_f1, 4)}
    for c in classes:
        row[f"true_bouts_{c}"] = true_bouts.get(c, 0)
        row[f"pred_bouts_{c}"] = pred_bouts.get(c, 0)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, default=None,
                   help="Tek bir prediction CSV (öncelik: bu varsa --subjects yok sayılır)")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject adları (örn. OpenFieldMA1_1 OpenFieldMA1_2)")
    p.add_argument("--model", default="lightgbm",
                   help="--auto-infer ile kullanılacak model adı")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results" / "test_inference",
                   help="Hem prediction hem karşılaştırma çıktılarının yeri")
    p.add_argument("--auto-infer", action="store_true",
                   help="prediction CSV yoksa infer.predict_one ile üret")
    p.add_argument("--force-infer", action="store_true",
                   help="prediction CSV mevcut olsa bile yeniden üret")
    args = p.parse_args()

    rows: list[dict] = []
    if args.pred is not None:
        subject = args.pred.name.replace("_predicted_frames.csv", "")
        _, gt_csv = find_subject_paths(subject)
        df = load_pred_and_gt(args.pred, gt_csv)
        rows.append(evaluate(subject, df, args.out_dir))
    elif args.subjects:
        for subject in args.subjects:
            try:
                if args.auto_infer or args.force_infer:
                    pred_csv = maybe_run_inference(
                        subject, args.model, args.out_dir,
                        force=args.force_infer,
                    )
                else:
                    pred_csv = args.out_dir / f"{subject}_predicted_frames.csv"
                    if not pred_csv.exists():
                        print(f"[skip] {subject}: {pred_csv} yok "
                              "(--auto-infer eklemeyi düşün)")
                        continue
                _, gt_csv = find_subject_paths(subject)
                df = load_pred_and_gt(pred_csv, gt_csv)
                rows.append(evaluate(subject, df, args.out_dir))
            except FileNotFoundError as exc:
                print(f"[skip] {subject}: {exc}")
    else:
        p.error("--pred ya da --subjects ver")

    if len(rows) > 1:
        summary = pd.DataFrame(rows)
        summary_path = args.out_dir / "compare_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n[write] {summary_path.relative_to(ROOT)}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
