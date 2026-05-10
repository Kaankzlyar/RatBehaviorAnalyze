"""
Window label join — birleştir window features + rule-based frame labels.

Inputs
------
    data/windows_all.parquet                              (src/window_classifier/features.py)
    data/DLCfiltered/.../<subject>_behavior_frames.csv    (rule-based detector)

Output
------
    data/windows_labeled.parquet
        cohort, subject_id (MA<n>_<m>), window_start, window_end, label,
        <feature_1..N>

Labeling şemaları
-----------------
    --scheme majority   (default) pencerenin frame'lerinin çoğunluğu
                        — kısa rare-class bout'ları kaybeder (≥16/30 frame
                        olmazsa "other" olarak etiketlenirler)

    --scheme any        pencerede priority_classes'tan biri en az
                        --any-thresh frame ise o sınıfla etiketle (eğer
                        birden fazlası eşiği aşıyorsa daha çok frame
                        içereni seç). Aksi halde majority'ye düşer.
                        Kısa bout recall'unu artırır, "other" sınıfını
                        küçültür ⇒ class imbalance azalır ama
                        rare-class label noise biraz artar.

Usage
-----
    python -m src.window_classifier.label_join
    python -m src.window_classifier.label_join \\
        --scheme any --any-thresh 5 --priority-classes grooming,rearing
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent.parent
DLC_DIR  = ROOT / "data" / "DLCfiltered"
WIN_PATH = ROOT / "data" / "windows_all.parquet"
OUT_PATH = ROOT / "data" / "windows_labeled.parquet"

SUBJECT_RE = re.compile(r"MA(\d+)_(\d+)")


def load_windows(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"{path} (or {csv_path}) bulunamadı — önce "
        "src/window_classifier/features.py'yi çalıştır."
    )


def find_frames_csv(subject_dir_name: str) -> Path | None:
    matches = list(DLC_DIR.glob(
        f"*/{subject_dir_name}/{subject_dir_name}_behavior_frames.csv"
    ))
    if not matches:
        matches = list(DLC_DIR.glob(
            f"*/*/{subject_dir_name}_behavior_frames.csv"
        ))
    return matches[0] if matches else None


def majority_label(frame_labels: np.ndarray, start: int, end_inclusive: int):
    if start < 0 or end_inclusive >= len(frame_labels):
        return None
    window = frame_labels[start:end_inclusive + 1]
    if len(window) == 0:
        return None
    vals, counts = np.unique(window, return_counts=True)
    return vals[counts.argmax()]


def any_label(frame_labels: np.ndarray, start: int, end_inclusive: int,
              priority_classes: tuple[str, ...] = ("grooming", "rearing"),
              min_frames: int = 5):
    """Pencerede priority_classes'tan biri >= min_frames frame içeriyorsa
    o sınıfı döndür (eşiği aşan birden fazlası varsa en çok frame içereni —
    eşitlikte priority_classes sırasına göre).

    Aksi halde majority_label'a düşer.
    """
    if start < 0 or end_inclusive >= len(frame_labels):
        return None
    window = frame_labels[start:end_inclusive + 1]
    if len(window) == 0:
        return None

    best_class = None
    best_count = 0
    for c in priority_classes:
        cnt = int(np.sum(window == c))
        if cnt >= min_frames and cnt > best_count:
            best_class = c
            best_count = cnt
    if best_class is not None:
        return best_class

    vals, counts = np.unique(window, return_counts=True)
    return vals[counts.argmax()]


def normalize_subject(subject_id: str) -> tuple[str, str]:
    """'OpenFieldMA5_2' -> ('MA5_2', 'MA5')."""
    m = SUBJECT_RE.search(subject_id)
    if not m:
        return subject_id, "unknown"
    cohort    = f"MA{m.group(1)}"
    canonical = f"MA{m.group(1)}_{m.group(2)}"
    return canonical, cohort


def join_labels(windows: pd.DataFrame, label_fn=majority_label) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for raw_subject, group in windows.groupby("subject_id", sort=False):
        canonical, cohort = normalize_subject(raw_subject)
        frames_path = find_frames_csv(raw_subject)
        if frames_path is None:
            print(f"  [skip] {raw_subject}: behavior_frames.csv yok")
            continue
        frames = pd.read_csv(frames_path)
        labels = frames["behaviour"].to_numpy()

        starts = group["window_start"].to_numpy()
        ends   = group["window_end"].to_numpy()
        win_labels = np.empty(len(group), dtype=object)
        for i, (s, e) in enumerate(zip(starts, ends)):
            win_labels[i] = label_fn(labels, int(s), int(e))

        annotated = group.copy()
        annotated.insert(0, "cohort", cohort)
        annotated["subject_id"] = canonical
        annotated["label"]      = win_labels
        out_parts.append(annotated)

        rear   = int((win_labels == "rearing").sum())
        groom  = int((win_labels == "grooming").sum())
        other  = int((win_labels == "other").sum())
        unm    = int(sum(1 for x in win_labels if x is None))
        print(f"  {canonical:8s} ({cohort})  n={len(group):4d}  "
              f"rear={rear:3d}  groom={groom:3d}  other={other:4d}  "
              f"unmatched={unm}")
    if not out_parts:
        raise RuntimeError("eşleştirilebilen denek yok")
    return pd.concat(out_parts, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="inp", type=Path, default=WIN_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--scheme", choices=("majority", "any"),
                   default="majority",
                   help="Pencere etiketleme şeması (default: majority)")
    p.add_argument("--any-thresh", type=int, default=5, dest="any_thresh",
                   help="--scheme=any için pencerede priority sınıfının "
                        "asgari frame sayısı (default: 5)")
    p.add_argument("--priority-classes", default="grooming,rearing",
                   dest="priority_classes",
                   help="--scheme=any için öncelik sınıfları, virgülle "
                        "ayrılmış (default: grooming,rearing)")
    args = p.parse_args()

    windows = load_windows(args.inp)
    print(f"[load] {len(windows)} pencere × {windows.shape[1]} sütun")

    if args.scheme == "any":
        priority = tuple(c.strip() for c in args.priority_classes.split(",")
                         if c.strip())
        if not priority:
            p.error("--priority-classes boş olamaz")
        print(f"[scheme] any  thresh={args.any_thresh}  priority={priority}")

        def label_fn(labels, s, e):
            return any_label(labels, s, e,
                             priority_classes=priority,
                             min_frames=args.any_thresh)
    else:
        print("[scheme] majority")
        label_fn = majority_label

    labeled = join_labels(windows, label_fn=label_fn)

    n_drop = labeled["label"].isna().sum()
    if n_drop:
        print(f"[warn] etiketlenemeyen {n_drop} pencere atılıyor")
        labeled = labeled.dropna(subset=["label"])

    # cohort ve subject sütunları öne alınmış halde feature kolonlarıyla birlikte:
    cols = ["cohort", "subject_id", "window_start", "window_end", "label"]
    feat_cols = [c for c in labeled.columns if c not in cols]
    labeled = labeled[cols + feat_cols]

    print(f"\n[summary] {len(labeled)} etiketli pencere")
    print(labeled["label"].value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        labeled.to_parquet(args.out, index=False)
        target = args.out
    except (ImportError, ValueError) as exc:
        target = args.out.with_suffix(".csv")
        labeled.to_csv(target, index=False)
        print(f"[warn] parquet kullanılamıyor ({exc.__class__.__name__}); CSV yazıldı")
    try:
        display = target.resolve().relative_to(ROOT)
    except ValueError:
        display = target
    print(f"[write] {display}: "
          f"{len(labeled)} satır × {labeled.shape[1]} sütun")


if __name__ == "__main__":
    main()
