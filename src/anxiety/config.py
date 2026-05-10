"""
Anxiety Pipeline — Arena & Inner-Zone Configuration
---------------------------------------------------
Single source of truth for arena bounding box and inner (center) zone used
across spatial-rearing analysis, predict_anxiety_v2, and Kare batch runs.

These pixel coordinates are *manually defined* from video annotation; they
are NOT derived from a generic margin (e.g. oft_metrics.auto_inner_zone with
20%). Any script that needs to classify body_center positions as
center-vs-wall must import from here so the definition stays consistent.

Arena
-----
    x ∈ [397, 777]   (width 380 px)
    y ∈ [156, 535]   (height 379 px)

Inner zone (center)
-------------------
    x ∈ [422, 748]   (width 326 px → ~13.7% margin per side)
    y ∈ [182, 506]   (height 324 px)

History
-------
Initially the spatial_rearing analysis used `auto_inner_zone(arena, 0.20)`
which yielded a noticeably smaller center (~472–700, 228–455). That
classified some genuinely centered rear bouts as wall. Authoritative zone
restored from analysis/open_field/run_kare_batch.py (lines 29–30, "Kare
arena coordinates confirmed from video annotation").
"""
from __future__ import annotations

# Arena bounding box: (x_min, x_max, y_min, y_max)
ARENA: tuple[float, float, float, float] = (397.0, 777.0, 156.0, 535.0)

# Inner / center zone: (x_min, x_max, y_min, y_max) — body_center in this box
# during a rearing bout is classified as 'center'; outside is 'wall'.
INNER_ZONE: tuple[float, float, float, float] = (422.0, 748.0, 182.0, 506.0)

# Frame rate for all sessions in this dataset
FPS: float = 30.0
