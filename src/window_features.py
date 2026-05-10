"""
Window-level feature extractor for DLC pose CSVs.

For each subject, slides a window of ``--window`` frames across the session
with stride ``--stride`` and computes:

  Per-keypoint × {x, y}: mean, std, min, max, range, |vel| mean/max/std,
                         |acc| mean/max
  Per-keypoint:          mean likelihood (tracking quality signal)
  Cross-keypoint:        htdist (nose↔tail_base), nose→forepaw,
                         ear-axis angular variability, bbox area, body
                         centroid speed

Coordinates with likelihood < ``--likelihood-thresh`` are masked to NaN
before computing stats. NaN-aware stats (np.nanmean etc.) tolerate
isolated drop-outs within a window.

Auto-discovers keypoints from the CSV header so the same script works for
both the OFT 9-keypoint set and the planned T-maze 5-keypoint set.

Usage
-----
    python -m src.window_features                    # all subjects
    python -m src.window_features --subject OpenFieldMA1_1
    python -m src.window_features --window 60 --stride 30

Output
------
    data/windows_all.parquet  (or .csv if pyarrow missing)

Schema:
    subject_id, window_start, window_end, <feature_1..N>
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DLC_DIR = ROOT / "data" / "DLCfiltered"
OUT_DEFAULT = ROOT / "data" / "windows_all.parquet"

LIKELIHOOD_THRESH = 0.6
DEFAULT_WINDOW = 30   # 1 s @ 30 fps
DEFAULT_STRIDE = 15   # 50 % overlap


# ── load + mask ──────────────────────────────────────────────────────────────

def load_dlc_flat(csv_path: Path, lik_thresh: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = [f"{bp}_{coord}" for _, bp, coord in df.columns]

    # mask (x, y) where likelihood < threshold
    bps = discover_keypoints(df)
    for bp in bps:
        lik_col = f"{bp}_likelihood"
        if lik_col not in df.columns:
            continue
        bad = df[lik_col] < lik_thresh
        if f"{bp}_x" in df.columns:
            df.loc[bad, f"{bp}_x"] = np.nan
        if f"{bp}_y" in df.columns:
            df.loc[bad, f"{bp}_y"] = np.nan
    return df


def discover_keypoints(df: pd.DataFrame) -> list[str]:
    bps = set()
    for c in df.columns:
        for suf in ("_x", "_y", "_likelihood"):
            if c.endswith(suf):
                bps.add(c[: -len(suf)])
                break
    return sorted(bps)


# ── feature blocks ───────────────────────────────────────────────────────────

def _safe(fn, *args, **kw):
    """np.nan*** with all-NaN warning silenced; returns NaN on empty/all-NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            v = fn(*args, **kw)
        except (ValueError, TypeError):
            return np.nan
    return v


def per_keypoint_features(win: pd.DataFrame, keypoints: list[str]) -> dict:
    feats: dict = {}
    for bp in keypoints:
        for coord in ("x", "y"):
            col = f"{bp}_{coord}"
            if col not in win.columns:
                continue
            v = win[col].to_numpy(dtype=float)
            feats[f"{bp}_{coord}_mean"] = _safe(np.nanmean, v)
            feats[f"{bp}_{coord}_std"] = _safe(np.nanstd, v)
            feats[f"{bp}_{coord}_min"] = _safe(np.nanmin, v)
            feats[f"{bp}_{coord}_max"] = _safe(np.nanmax, v)
            feats[f"{bp}_{coord}_range"] = (
                feats[f"{bp}_{coord}_max"] - feats[f"{bp}_{coord}_min"]
                if not (np.isnan(feats[f"{bp}_{coord}_max"])
                        or np.isnan(feats[f"{bp}_{coord}_min"]))
                else np.nan
            )
            # |velocity|
            vel = np.diff(v)
            feats[f"{bp}_{coord}_vel_mean"] = _safe(np.nanmean, np.abs(vel))
            feats[f"{bp}_{coord}_vel_max"] = _safe(np.nanmax, np.abs(vel)) if len(vel) else np.nan
            feats[f"{bp}_{coord}_vel_std"] = _safe(np.nanstd, vel)
            # |acceleration|
            acc = np.diff(vel)
            feats[f"{bp}_{coord}_acc_mean"] = _safe(np.nanmean, np.abs(acc)) if len(acc) else np.nan
            feats[f"{bp}_{coord}_acc_max"] = _safe(np.nanmax, np.abs(acc)) if len(acc) else np.nan
        lik_col = f"{bp}_likelihood"
        if lik_col in win.columns:
            feats[f"{bp}_lik_mean"] = float(win[lik_col].mean())
    return feats


def _pair_distance(win: pd.DataFrame, a: str, b: str) -> np.ndarray | None:
    if any(f"{x}_{c}" not in win.columns for x in (a, b) for c in ("x", "y")):
        return None
    ax = win[f"{a}_x"].to_numpy(dtype=float)
    ay = win[f"{a}_y"].to_numpy(dtype=float)
    bx = win[f"{b}_x"].to_numpy(dtype=float)
    by = win[f"{b}_y"].to_numpy(dtype=float)
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def cross_keypoint_features(win: pd.DataFrame, keypoints: list[str]) -> dict:
    feats: dict = {}

    # head-tail distance
    htd = _pair_distance(win, "nose", "tail_base")
    if htd is not None:
        feats["htdist_mean"] = _safe(np.nanmean, htd)
        feats["htdist_std"] = _safe(np.nanstd, htd)
        feats["htdist_min"] = _safe(np.nanmin, htd)
        feats["htdist_max"] = _safe(np.nanmax, htd)
        d = np.diff(htd)
        feats["htdist_vel_mean"] = _safe(np.nanmean, np.abs(d)) if len(d) else np.nan

    # nose ↔ forepaw (min of L/R)
    nfp_l = _pair_distance(win, "nose", "left_forepaw")
    nfp_r = _pair_distance(win, "nose", "right_forepaw")
    if nfp_l is not None or nfp_r is not None:
        if nfp_l is not None and nfp_r is not None:
            nfp = np.fmin(nfp_l, nfp_r)
        else:
            nfp = nfp_l if nfp_l is not None else nfp_r
        feats["nose_fp_mean"] = _safe(np.nanmean, nfp)
        feats["nose_fp_min"] = _safe(np.nanmin, nfp)
        feats["nose_fp_std"] = _safe(np.nanstd, nfp)

    # ear-axis angular variability (kafa rotation — useful for grooming)
    if all(f"{bp}_{c}" in win.columns for bp in ("left_ear", "right_ear") for c in ("x", "y")):
        ex = win["left_ear_x"].to_numpy() - win["right_ear_x"].to_numpy()
        ey = win["left_ear_y"].to_numpy() - win["right_ear_y"].to_numpy()
        ang = np.arctan2(ey, ex)
        ang = ang[~np.isnan(ang)]
        if len(ang) > 1:
            ang = np.unwrap(ang)
            feats["ear_angle_std"] = float(np.std(ang))
            feats["ear_angle_range"] = float(np.max(ang) - np.min(ang))
            feats["ear_angle_vel_mean"] = float(np.mean(np.abs(np.diff(ang))))

    # bbox area (all-keypoint convex extent)
    x_cols = [f"{bp}_x" for bp in keypoints if f"{bp}_x" in win.columns]
    y_cols = [f"{bp}_y" for bp in keypoints if f"{bp}_y" in win.columns]
    if x_cols and y_cols:
        xa = win[x_cols].to_numpy(dtype=float)
        ya = win[y_cols].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_min = np.nanmin(xa, axis=1)
            x_max = np.nanmax(xa, axis=1)
            y_min = np.nanmin(ya, axis=1)
            y_max = np.nanmax(ya, axis=1)
        bbox = (x_max - x_min) * (y_max - y_min)
        feats["bbox_area_mean"] = _safe(np.nanmean, bbox)
        feats["bbox_area_std"] = _safe(np.nanstd, bbox)
        feats["bbox_area_min"] = _safe(np.nanmin, bbox)
        feats["bbox_area_max"] = _safe(np.nanmax, bbox)

        # centroid speed
        cx = _safe(np.nanmean, xa, axis=1)
        cy = _safe(np.nanmean, ya, axis=1)
        if isinstance(cx, np.ndarray) and isinstance(cy, np.ndarray):
            spd = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
            feats["centroid_speed_mean"] = _safe(np.nanmean, spd)
            feats["centroid_speed_max"] = _safe(np.nanmax, spd) if len(spd) else np.nan
            feats["centroid_speed_std"] = _safe(np.nanstd, spd)

    return feats


# ── window driver ────────────────────────────────────────────────────────────

def extract_windows(dlc: pd.DataFrame, subject_id: str,
                    window_size: int, stride: int) -> pd.DataFrame:
    keypoints = discover_keypoints(dlc)
    n_frames = len(dlc)
    rows: list[dict] = []
    for start in range(0, n_frames - window_size + 1, stride):
        end = start + window_size
        win = dlc.iloc[start:end]
        feats = {
            "subject_id": subject_id,
            "window_start": start,
            "window_end": end - 1,
        }
        feats.update(per_keypoint_features(win, keypoints))
        feats.update(cross_keypoint_features(win, keypoints))
        rows.append(feats)
    return pd.DataFrame(rows)


def discover_subject_csvs() -> list[Path]:
    """Return canonical pose CSVs for OpenField only: data/DLCfiltered/<group>/<subject>/<subject>.csv,
    where subject folder starts with 'OpenField', excluding the Kare reference dir."""
    out: list[Path] = []
    for csv in DLC_DIR.glob("*/*/*.csv"):
        if "Kare" in csv.parts:
            continue
        subject = csv.parent.name
        if not subject.startswith("OpenField"):
            continue
        if csv.stem != subject:
            continue
        out.append(csv)
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"Window size in frames (default {DEFAULT_WINDOW})")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help=f"Stride in frames (default {DEFAULT_STRIDE})")
    p.add_argument("--likelihood-thresh", type=float, default=LIKELIHOOD_THRESH)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--subject", help="Process only this subject (else all)")
    args = p.parse_args()

    csvs = discover_subject_csvs()
    if args.subject:
        csvs = [c for c in csvs if c.parent.name == args.subject]
    if not csvs:
        print("[err] no subject CSVs found")
        return

    print(f"[load] {len(csvs)} subject(s); window={args.window} stride={args.stride}")

    parts: list[pd.DataFrame] = []
    for csv in csvs:
        subject = csv.parent.name
        dlc = load_dlc_flat(csv, args.likelihood_thresh)
        df = extract_windows(dlc, subject, args.window, args.stride)
        print(f"  {subject:24s} -> {len(df):5d} windows × {df.shape[1]:3d} cols  (n_frames={len(dlc)})")
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # prefer parquet, fall back to csv if pyarrow missing
    try:
        out.to_parquet(args.out, index=False)
        target = args.out
    except (ImportError, ValueError) as exc:
        target = args.out.with_suffix(".csv")
        out.to_csv(target, index=False)
        print(f"  parquet unavailable ({exc.__class__.__name__}), wrote CSV instead")

    print(f"[write] {target.relative_to(ROOT)}: {len(out)} windows × {out.shape[1]} cols")
    feat_cols = [c for c in out.columns if c not in ("subject_id", "window_start", "window_end")]
    nan_pct = out[feat_cols].isna().mean().mean() * 100
    print(f"[stats] {len(feat_cols)} feature columns; mean NaN rate {nan_pct:.2f}%")


if __name__ == "__main__":
    main()
