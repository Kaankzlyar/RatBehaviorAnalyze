"""
Minimal subset of OFT metrics needed for the anxiety binary classifier demo.

Reuses conventions from src/behavior_detection.py and analysis/oft_metrics.py
but standalone (no arena interactive picker etc.) so that train + predict
can run from any DLC CSV without per-subject configuration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Default arena rectangle (px) — derived from MA1_2 and consistent across
# OFT recordings in this dataset; see docs/behavior_detection_documentation.md
ARENA = {"x_left": 397.0, "x_right": 775.0, "y_top": 158.0, "y_bottom": 532.0}
LIKELIHOOD_THRESH = 0.6
FPS = 30.0
PERIPHERY_MARGIN = 0.20    # inner zone defined by 20% inset
FREEZE_SPEED_THRESH = 5.0  # px/s
FREEZE_MIN_FRAMES = 30     # 1 s @ 30 fps

ANXIETY_FEATURES = [
    "pct_time_center", "pct_time_periphery",
    "center_zone_entries", "pct_time_freeze",
]


def load_dlc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = [f"{bp}_{coord}" for _, bp, coord in df.columns]
    return df


def _mask_low_likelihood(x: pd.Series, lik: pd.Series) -> pd.Series:
    out = x.copy().astype(float)
    out[lik < LIKELIHOOD_THRESH] = np.nan
    return out


def compute_anxiety_features(dlc: pd.DataFrame) -> dict:
    """Return the 4 anxiety-axis features from a DLC pose DataFrame."""
    bc_x = _mask_low_likelihood(dlc["body_center_x"], dlc["body_center_likelihood"])
    bc_y = _mask_low_likelihood(dlc["body_center_y"], dlc["body_center_likelihood"])

    valid = bc_x.notna() & bc_y.notna()
    n_valid = int(valid.sum())
    if n_valid < 10:
        return {f: np.nan for f in ANXIETY_FEATURES}

    # speed (px/s) frame-to-frame
    vx = bc_x.diff() * FPS
    vy = bc_y.diff() * FPS
    speed = np.sqrt(vx ** 2 + vy ** 2)

    # arena → inner (center) zone
    A = ARENA
    cx = (A["x_left"] + A["x_right"]) / 2
    cy = (A["y_top"] + A["y_bottom"]) / 2
    half_w = (A["x_right"] - A["x_left"]) / 2
    half_h = (A["y_bottom"] - A["y_top"]) / 2
    inner_half_w = half_w * (1 - PERIPHERY_MARGIN)
    inner_half_h = half_h * (1 - PERIPHERY_MARGIN)

    in_center_raw = ((bc_x - cx).abs() < inner_half_w) & ((bc_y - cy).abs() < inner_half_h)
    # restrict to valid frames
    pct_time_center = float(in_center_raw[valid].mean() * 100) if n_valid else np.nan
    pct_time_periphery = 100.0 - pct_time_center

    # center zone entries: count False→True transitions in valid frames only
    in_center = in_center_raw.where(valid, other=False).astype(int)
    center_zone_entries = int(((in_center.diff() == 1) & valid).sum())

    # freezing: speed below thresh for ≥ 1 s
    is_slow = (speed < FREEZE_SPEED_THRESH).fillna(False)
    runs = (is_slow != is_slow.shift()).cumsum()
    run_lengths = is_slow.groupby(runs).transform("size")
    is_freezing = is_slow & (run_lengths >= FREEZE_MIN_FRAMES)
    pct_time_freeze = float(is_freezing[valid].mean() * 100) if n_valid else np.nan

    return {
        "pct_time_center": pct_time_center,
        "pct_time_periphery": pct_time_periphery,
        "center_zone_entries": center_zone_entries,
        "pct_time_freeze": pct_time_freeze,
    }
