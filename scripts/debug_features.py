import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from behavior_detection import *

df = load_dlc_csv(str(ROOT / "data" / "DLCfiltered" / "control" / "OpenFieldMA1_1" / "OpenFieldMA1_1.csv"))
masked = mask_low_likelihood(df, LIKELIHOOD_THRESH)
feat = compute_features(df, masked)

print('FEATURES (min/max/mean):')
for col in feat.columns:
    print(f'{col:15s}: {feat[col].min():.1f} / {feat[col].max():.1f} / {feat[col].mean():.1f}')

print('\nTHRESHOLDS (rearing):')
print(f'  htdist < {REAR_COMPACT_HTDIST}')
print(f'  fp_hp_vert > {REAR_EXTEND_FPHP} & nose_y < {REAR_NOSE_Y_MAX}')
print(f'  fp_hp_vert < {REAR_BOTTOM_STRONG_FPHP} & nose_y > {REAR_BOTTOM_NOSE_Y}')
print(f'  fp_hp_vert < {REAR_BOTTOM_COMPACT_FPHP} & htdist < {REAR_BOTTOM_HTDIST} & nose_y > {REAR_BOTTOM_NOSE_Y}')
print(f'THRESHOLD (grooming):')
print(f'  nose2fp < {GROOM_NOSE2FP} & fp_hp_vert < {GROOM_MAX_FPHP}')

print('\n\nDEBUG: How many frames match each rearing condition:')
rear_compact  = feat["htdist"] < REAR_COMPACT_HTDIST
rear_top_wall = (feat["fp_hp_vert"] > REAR_EXTEND_FPHP) & (feat["nose_y"] < REAR_NOSE_Y_MAX)
rear_bot_strong  = (feat["fp_hp_vert"] < REAR_BOTTOM_STRONG_FPHP) & (feat["nose_y"] > REAR_BOTTOM_NOSE_Y)
rear_bot_compact = (feat["fp_hp_vert"] < REAR_BOTTOM_COMPACT_FPHP) & (feat["nose_y"] > REAR_BOTTOM_NOSE_Y) & (feat["htdist"] < REAR_BOTTOM_HTDIST)
rear_side_wall = (
    (feat["htd_y"] > REAR_SIDE_HTD_Y_MIN) & (feat["htd_y"] < REAR_SIDE_HTD_Y_MAX)
    & ((feat["nose_x"] > REAR_SIDE_NOSE_X_RIGHT) | (feat["nose_x"] < REAR_SIDE_NOSE_X_LEFT))
    & (feat["fp_hp_vert"] < 25)
)

print(f'  Compact rearing (htdist < {REAR_COMPACT_HTDIST}): {rear_compact.sum()} frames')
print(f'  Top-wall rearing: {rear_top_wall.sum()} frames')
print(f'  Bottom-strong rearing: {rear_bot_strong.sum()} frames')
print(f'  Bottom-compact rearing: {rear_bot_compact.sum()} frames')
print(f'  Side-wall rearing: {rear_side_wall.sum()} frames')

print('\n\nDEBUG: Grooming frames:')
grooming = (feat["nose2fp"] < GROOM_NOSE2FP) & (feat["fp_hp_vert"] < GROOM_MAX_FPHP)
print(f'  Grooming (nose2fp < {GROOM_NOSE2FP} & fp_hp_vert < {GROOM_MAX_FPHP}): {grooming.sum()} frames')
