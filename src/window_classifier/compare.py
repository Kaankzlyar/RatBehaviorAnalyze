"""
Window-classifier prediction vs rule-based ground-truth karşılaştırması.

İki seviyede değerlendirme:

  frame-level   modelin frame-bazlı tahmini (window prob ortalaması →
                argmax) vs rule-based detector frame label'ları
                — "praktik tahmin kalitesi"

  window-level  modelin window-bazlı argmax tahmini vs aynı
                pencerenin majority frame label'ı (label_join.py'nin
                training için kullandığı şema)
                — "modelin training hedefini ne kadar öğrendiği";
                  CV macro-F1 ile apples-to-apples karşılaştırılabilir

Usage
-----
    # tek session, mevcut prediction CSV'si üzerinden (frame-level)
    python -m src.window_classifier.compare \\
        --pred results/test_inference/OpenFieldMA1_1_predicted_frames.csv

    # window-level + frame-level birlikte (subject + model gerekir)
    python -m src.window_classifier.compare --subjects OpenFieldMA1_1 --model lightgbm --level both --auto-infer

    # birden çok subject
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


def run_full_pipeline(subject: str, model: str, out_dir: Path,
                      write_frame_csv: bool = True) -> dict:
    """Pipeline'ı tek seferde çalıştırıp window + frame seviyesinde tahmin verir.

    Returns
    -------
    dict with keys:
        win_df       : pencere feature DataFrame (window_start, window_end, ...)
        win_pred     : np.ndarray[str], window-level argmax
        frame_pred   : np.ndarray[str], frame-level argmax (window prob ort.)
        frame_probs  : np.ndarray[(n_frames, n_classes)]
        class_names  : list[str]
        n_frames     : int
    """
    from src.window_classifier.infer import (
        DEFAULT_FPS, load_model_bundle, windows_to_frame_probs,
    )
    from src.window_classifier.features import (
        DEFAULT_WINDOW, DEFAULT_STRIDE, DEFAULT_LONG_WINDOW, LIKELIHOOD_THRESH,
        apply_likelihood_mask, compute_rule_signals, extract_windows,
        load_dlc_flat_raw,
    )

    dlc_csv, _ = find_subject_paths(subject)
    bundle = load_model_bundle(model)
    raw = load_dlc_flat_raw(dlc_csv)
    dlc = apply_likelihood_mask(raw, LIKELIHOOD_THRESH)
    rule_sig = compute_rule_signals(raw, dlc, fps=DEFAULT_FPS)
    win_df = extract_windows(
        dlc, subject, DEFAULT_WINDOW, DEFAULT_STRIDE,
        long_window_size=DEFAULT_LONG_WINDOW, fps=DEFAULT_FPS,
        rule_signals=rule_sig,
    )

    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in win_df.columns]
    if missing:
        raise ValueError(
            f"window_features çıktısında eksik kolonlar var: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    X = win_df[feature_cols].values.astype(np.float32)
    X = bundle["imputer"].transform(X)
    proba = bundle["model"].predict_proba(X)
    le = bundle["label_encoder"]
    win_pred = le.inverse_transform(proba.argmax(axis=1))

    n_frames = len(dlc)
    starts = win_df["window_start"].to_numpy(int)
    ends   = win_df["window_end"].to_numpy(int)
    frame_probs = windows_to_frame_probs(proba, starts, ends, n_frames)
    frame_pred = le.inverse_transform(frame_probs.argmax(axis=1))

    if write_frame_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_df = pd.DataFrame({
            "frame": np.arange(n_frames),
            "time_s": np.round(np.arange(n_frames) / DEFAULT_FPS, 3),
            "predicted_label": frame_pred,
        })
        for ci, cn in enumerate(list(le.classes_)):
            frames_df[f"prob_{cn}"] = np.round(frame_probs[:, ci], 4)
        frames_df.to_csv(out_dir / f"{subject}_predicted_frames.csv",
                         index=False)

    return {
        "win_df": win_df,
        "win_pred": np.asarray(win_pred),
        "frame_pred": np.asarray(frame_pred),
        "frame_probs": frame_probs,
        "class_names": list(le.classes_),
        "n_frames": n_frames,
    }


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


def _bout_counts(arr: np.ndarray) -> dict:
    """Run-length sayımı: ardışık aynı etiket koşularını sayar."""
    if len(arr) == 0:
        return {}
    arr = np.asarray(arr)
    change = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    starts = np.concatenate(([0], change))
    out: dict[str, int] = {}
    for s in starts:
        out[arr[s]] = out.get(arr[s], 0) + 1
    return out


def _print_metrics(subject: str, level: str, n_unit_label: str,
                   y_true: np.ndarray, y_pred: np.ndarray,
                   classes: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=classes,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=classes,
                           average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=classes, digits=4, zero_division=0,
    )
    print(f"\n=== {subject}  [{level.upper()}-LEVEL] ===")
    print(f"  {n_unit_label}={len(y_true)}  acc={acc:.4f}  "
          f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")
    print(report)
    return {
        "cm": cm, "accuracy": acc,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
    }


def evaluate(subject: str, df: pd.DataFrame, out_dir: Path) -> dict:
    """Frame-level eval + bout count tablosu."""
    classes = sorted(set(df["true"]) | set(df["predicted"]))
    y_true = df["true"].to_numpy()
    y_pred = df["predicted"].to_numpy()
    m = _print_metrics(subject, "frame", "n_frames", y_true, y_pred, classes)

    true_bouts = _bout_counts(y_true)
    pred_bouts = _bout_counts(y_pred)
    print("  bouts (true → pred):")
    for c in classes:
        print(f"    {c:9s}  {true_bouts.get(c, 0):3d}  →  "
              f"{pred_bouts.get(c, 0):3d}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{subject}_compare_frames.csv", index=False)
    plot_confusion_matrix(
        m["cm"], classes,
        title=f"{subject} — frame-level "
              f"(acc={m['accuracy']:.3f}, macroF1={m['macro_f1']:.3f})",
        out_path=out_dir / f"{subject}_compare_cm.png",
    )

    row = {"subject": subject, "level": "frame", "n": len(df),
           "accuracy": round(m["accuracy"], 4),
           "macro_f1": round(m["macro_f1"], 4),
           "weighted_f1": round(m["weighted_f1"], 4)}
    for c in classes:
        row[f"true_bouts_{c}"] = true_bouts.get(c, 0)
        row[f"pred_bouts_{c}"] = pred_bouts.get(c, 0)
    return row


def evaluate_window_level(subject: str, win_df: pd.DataFrame,
                          win_pred: np.ndarray, gt_csv: Path,
                          out_dir: Path) -> dict:
    """Window-level eval: model pencere argmax vs majority frame label.

    Aynı pencereleme şemasını (label_join.majority_label) kullandığımız için
    bu rakamlar training CV macro-F1 ile apples-to-apples okunabilir.
    """
    from src.window_classifier.label_join import majority_label

    gt = pd.read_csv(gt_csv)
    frame_labels = gt["behaviour"].astype(str).to_numpy()

    starts = win_df["window_start"].to_numpy(int)
    ends   = win_df["window_end"].to_numpy(int)

    win_true: list[str] = []
    keep: list[int] = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        lbl = majority_label(frame_labels, int(s), int(e))
        if lbl is not None:
            win_true.append(str(lbl))
            keep.append(i)

    if not keep:
        print(f"[skip] {subject}: window-level GT eşleşmesi yok")
        return {}
    y_true = np.array(win_true)
    y_pred = np.asarray(win_pred)[keep]
    classes = sorted(set(y_true) | set(y_pred))

    m = _print_metrics(subject, "window", "n_windows", y_true, y_pred, classes)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "window_start": starts[keep],
        "window_end":   ends[keep],
        "true":         y_true,
        "predicted":    y_pred,
    }).to_csv(out_dir / f"{subject}_compare_windows.csv", index=False)
    plot_confusion_matrix(
        m["cm"], classes,
        title=f"{subject} — window-level "
              f"(acc={m['accuracy']:.3f}, macroF1={m['macro_f1']:.3f})",
        out_path=out_dir / f"{subject}_compare_window_cm.png",
    )

    return {"subject": subject, "level": "window", "n": len(y_true),
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "weighted_f1": round(m["weighted_f1"], 4)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, default=None,
                   help="Tek bir prediction CSV (sadece --level=frame ile)")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject adları (örn. OpenFieldMA1_1 OpenFieldMA1_2)")
    p.add_argument("--model", default="lightgbm",
                   help="Model adı: randomforest | xgboost | lightgbm")
    p.add_argument("--level", choices=("frame", "window", "both"),
                   default="frame",
                   help="Karşılaştırma seviyesi (default: frame). "
                        "'window' / 'both' --subjects + --model ister.")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results" / "test_inference",
                   help="Hem prediction hem karşılaştırma çıktılarının yeri")
    p.add_argument("--auto-infer", action="store_true",
                   help="prediction CSV yoksa infer.predict_one ile üret "
                        "(yalnızca --level=frame için anlamlı)")
    p.add_argument("--force-infer", action="store_true",
                   help="prediction CSV mevcut olsa bile yeniden üret")
    args = p.parse_args()

    if args.level in ("window", "both") and args.pred is not None:
        p.error("--pred sadece --level=frame ile kullanılabilir; "
                "window-level eval için --subjects + --model ver")

    rows: list[dict] = []

    # --- (a) tek dosya, frame-level only ---
    if args.pred is not None:
        subject = args.pred.name.replace("_predicted_frames.csv", "")
        _, gt_csv = find_subject_paths(subject)
        df = load_pred_and_gt(args.pred, gt_csv)
        rows.append(evaluate(subject, df, args.out_dir))

    # --- (b) subject listesi ---
    elif args.subjects:
        for subject in args.subjects:
            try:
                _, gt_csv = find_subject_paths(subject)

                if args.level == "frame":
                    # eski yol: predicted_frames.csv'yi oku
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
                    df = load_pred_and_gt(pred_csv, gt_csv)
                    rows.append(evaluate(subject, df, args.out_dir))

                else:
                    # window veya both → pipeline'ı tek seferde sür
                    out = run_full_pipeline(
                        subject, args.model, args.out_dir,
                        write_frame_csv=True,
                    )
                    if args.level in ("frame", "both"):
                        # frame-level — taze tahminleri kullan
                        gt = pd.read_csv(gt_csv)
                        n = min(out["n_frames"], len(gt))
                        df = pd.DataFrame({
                            "frame": np.arange(n),
                            "time_s": np.round(np.arange(n) / 30.0, 3),
                            "true": gt["behaviour"].astype(str).values[:n],
                            "predicted": out["frame_pred"][:n],
                        })
                        for ci, cn in enumerate(out["class_names"]):
                            df[f"prob_{cn}"] = np.round(
                                out["frame_probs"][:n, ci], 4)
                        rows.append(evaluate(subject, df, args.out_dir))
                    if args.level in ("window", "both"):
                        rows.append(evaluate_window_level(
                            subject, out["win_df"], out["win_pred"],
                            gt_csv, args.out_dir,
                        ))
            except FileNotFoundError as exc:
                print(f"[skip] {subject}: {exc}")

    else:
        p.error("--pred ya da --subjects ver")

    rows = [r for r in rows if r]
    if len(rows) > 1:
        summary = pd.DataFrame(rows)
        summary_path = args.out_dir / "compare_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n[write] {summary_path.relative_to(ROOT)}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
