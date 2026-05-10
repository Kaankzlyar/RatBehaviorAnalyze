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
    python -m src.window_classifier.features                    # all subjects
    python -m src.window_classifier.features --subject OpenFieldMA1_1
    python -m src.window_classifier.features --window 60 --stride 30

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

ROOT = Path(__file__).resolve().parent.parent.parent
DLC_DIR = ROOT / "data" / "DLCfiltered"
OUT_DEFAULT = ROOT / "data" / "windows_all.parquet"

LIKELIHOOD_THRESH = 0.6
DEFAULT_WINDOW = 30        # 1 s @ 30 fps
DEFAULT_STRIDE = 15        # 50 % overlap
DEFAULT_LONG_WINDOW = 90   # 3 s @ 30 fps — co-centered slow-context window
DEFAULT_FPS = 30.0
GROOMING_BAND_HZ = (4.0, 8.0)  # rat forepaw oscillation during grooming


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


# ── long (slow-context) window features ─────────────────────────────────────
# Grooming bouts last several seconds and have a rhythmic forepaw oscillation
# in the 4–8 Hz band. The 1 s window often catches only a fragment, so for
# every short window we also compute a co-centred 3 s window with:
#   - slow-context time-domain stats (suffix _w3s)
#   - frequency-domain band power on forepaw_y and nose↔forepaw distance
#   - bilateral forepaw correlation + a stationarity ratio (paw motion / body
#     motion) — high during grooming, low during locomotion.

def _interp_nans(y: np.ndarray) -> np.ndarray | None:
    """Linear-interpolate NaNs; return None if all-NaN or too short."""
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return None
    nans = np.isnan(y)
    if nans.all():
        return None
    if nans.any():
        idx = np.arange(len(y))
        y = y.copy()
        y[nans] = np.interp(idx[nans], idx[~nans], y[~nans])
    return y


def fft_band_features(y: np.ndarray, prefix: str, fps: float = DEFAULT_FPS) -> dict:
    """Spectral features for a 1-D signal.

    Returns peak frequency in 1–12 Hz, total power in the grooming band, and
    band-power ratio (grooming / 1–12 Hz). Empty dict if signal too short or
    fully NaN.
    """
    feats: dict = {}
    yi = _interp_nans(y)
    if yi is None or len(yi) < 16:
        return feats
    yi = yi - np.mean(yi)
    n = len(yi)
    spectrum = np.abs(np.fft.rfft(yi)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= 1.0) & (freqs <= 12.0)
    if not band.any():
        return feats
    g_band = (freqs >= GROOMING_BAND_HZ[0]) & (freqs <= GROOMING_BAND_HZ[1])
    total = float(spectrum[band].sum())
    grooming = float(spectrum[g_band].sum()) if g_band.any() else 0.0
    feats[f"{prefix}_fft_peak_hz"] = float(freqs[band][np.argmax(spectrum[band])])
    feats[f"{prefix}_fft_power_4_8"] = grooming
    feats[f"{prefix}_fft_band_ratio"] = grooming / total if total > 0 else 0.0
    return feats


def long_window_features(dlc: pd.DataFrame, start: int, end_inclusive: int,
                         keypoints: list[str], long_window_size: int,
                         fps: float = DEFAULT_FPS) -> dict:
    """Slow-context features on a window of size ``long_window_size`` co-centred
    on [start, end_inclusive]. Suffix _w3s; clipped at session edges."""
    feats: dict = {}
    if long_window_size <= 0:
        return feats
    n_frames = len(dlc)
    centre = (start + end_inclusive) // 2
    half = long_window_size // 2
    long_start = max(0, centre - half)
    long_end = min(n_frames, centre + half)
    if long_end - long_start < 16:
        return feats
    win = dlc.iloc[long_start:long_end]

    # nose ↔ forepaw distance — slow stats + spectrum
    nfp_l = _pair_distance(win, "nose", "left_forepaw")
    nfp_r = _pair_distance(win, "nose", "right_forepaw")
    nfp = None
    if nfp_l is not None and nfp_r is not None:
        nfp = np.fmin(nfp_l, nfp_r)
    elif nfp_l is not None or nfp_r is not None:
        nfp = nfp_l if nfp_l is not None else nfp_r
    if nfp is not None:
        feats["nose_fp_w3s_mean"] = _safe(np.nanmean, nfp)
        feats["nose_fp_w3s_min"]  = _safe(np.nanmin, nfp)
        feats["nose_fp_w3s_p25"]  = _safe(np.nanpercentile, nfp, 25)
        feats["nose_fp_w3s_std"]  = _safe(np.nanstd, nfp)
        feats.update(fft_band_features(nfp, "nose_fp_w3s", fps))

    # head-tail distance slow context
    htd = _pair_distance(win, "nose", "tail_base")
    if htd is not None:
        ht_max = _safe(np.nanmax, htd)
        ht_min = _safe(np.nanmin, htd)
        feats["htdist_w3s_std"] = _safe(np.nanstd, htd)
        feats["htdist_w3s_range"] = (
            ht_max - ht_min
            if not (np.isnan(ht_max) or np.isnan(ht_min)) else np.nan
        )

    # forepaw vertical motion FFTs (rhythmic grooming signature)
    if "left_forepaw_y" in win.columns:
        feats.update(fft_band_features(
            win["left_forepaw_y"].to_numpy(dtype=float), "lfp_y_w3s", fps))
    if "right_forepaw_y" in win.columns:
        feats.update(fft_band_features(
            win["right_forepaw_y"].to_numpy(dtype=float), "rfp_y_w3s", fps))

    # bilateral forepaw correlation — high during symmetric grooming
    if "left_forepaw_y" in win.columns and "right_forepaw_y" in win.columns:
        lfy = win["left_forepaw_y"].to_numpy(dtype=float)
        rfy = win["right_forepaw_y"].to_numpy(dtype=float)
        mask = ~np.isnan(lfy) & ~np.isnan(rfy)
        if mask.sum() >= 8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats["forepaw_y_corr_w3s"] = float(
                    np.corrcoef(lfy[mask], rfy[mask])[0, 1])

    # body centroid speed slow context + stationarity ratio
    x_cols = [f"{bp}_x" for bp in keypoints if f"{bp}_x" in win.columns]
    y_cols = [f"{bp}_y" for bp in keypoints if f"{bp}_y" in win.columns]
    if x_cols and y_cols:
        xa = win[x_cols].to_numpy(dtype=float)
        ya = win[y_cols].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cx = np.nanmean(xa, axis=1)
            cy = np.nanmean(ya, axis=1)
        spd = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        cs_mean = _safe(np.nanmean, spd)
        feats["centroid_speed_w3s_mean"] = cs_mean
        feats["centroid_speed_w3s_max"] = _safe(np.nanmax, spd) if len(spd) else np.nan
        nfp_std = feats.get("nose_fp_w3s_std")
        if (cs_mean is not None and nfp_std is not None
                and not (np.isnan(cs_mean) or np.isnan(nfp_std))
                and cs_mean > 1e-6):
            feats["stationary_paw_motion_w3s"] = float(nfp_std / (cs_mean + 1e-6))

    return feats


# ── window driver ────────────────────────────────────────────────────────────

def extract_windows(dlc: pd.DataFrame, subject_id: str,
                    window_size: int, stride: int,
                    long_window_size: int = DEFAULT_LONG_WINDOW,
                    fps: float = DEFAULT_FPS) -> pd.DataFrame:
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
        feats.update(long_window_features(
            dlc, start, end - 1, keypoints, long_window_size, fps))
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
    p.add_argument("--long-window", type=int, default=DEFAULT_LONG_WINDOW,
                   dest="long_window",
                   help=f"Co-centred slow-context window in frames "
                        f"(default {DEFAULT_LONG_WINDOW}; 0 disables)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help="Recording FPS; only used for FFT freq bins")
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

    print(f"[load] {len(csvs)} subject(s); window={args.window} stride={args.stride}"
          f" long_window={args.long_window} fps={args.fps:g}")

    parts: list[pd.DataFrame] = []
    for csv in csvs:
        subject = csv.parent.name
        dlc = load_dlc_flat(csv, args.likelihood_thresh)
        df = extract_windows(dlc, subject, args.window, args.stride,
                             long_window_size=args.long_window, fps=args.fps)
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
