"""
Anxiety Pipeline — Arena & Inner-Zone Configuration
---------------------------------------------------
Single source of truth for arena bounding box and inner (center) zone used
across spatial-rearing analysis, predict_anxiety_v2, and Kare batch runs.

Inner zone definition
---------------------
Bu pipeline'da inner zone arena'nın her kenarında %20 margin alınarak
hesaplanır — bu, klasik OFT (Open Field Test) literatürünün kullandığı
"center" tanımına yakın ve oft_metrics.py default'u ile aynı margin oranıdır.

  margin = 0.20
  inner_x ∈ [arena_x_min + 0.20·width, arena_x_max - 0.20·width]
  inner_y ∈ [arena_y_min + 0.20·height, arena_y_max - 0.20·height]

  Arena (397, 777, 156, 535) için:
    inner_x ∈ [473, 701]   (width 228 px)
    inner_y ∈ [232, 459]   (height 227 px)

History
-------
- İlk versiyonda spatial_rearing auto_inner_zone(arena, 0.20) kullanıyordu.
- Sonra run_kare_batch.py'deki manuel zone (422, 748, 182, 506) — %7
  margin'a denk geliyor — single source olarak alındı.
- Manuel zone arenanın ~%85'ini kapladığı için neredeyse tüm rear bout'ları
  "center" sayıyordu; gruplar arası fark kayboldu.
- %20 margin'a geri döndü: literatür standardına yakın, biyolojik olarak
  thigmotaxis halkası ile center'ı net ayırıyor, gruplar arası kalitatif
  fark yakalıyor.
"""
from __future__ import annotations

# Arena bounding box: (x_min, x_max, y_min, y_max) — manually annotated
ARENA: tuple[float, float, float, float] = (397.0, 777.0, 156.0, 535.0)

# Inner-zone margin (fraction of arena dimension on each side)
INNER_MARGIN: float = 0.20


def _derive_inner(arena: tuple, margin: float) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = arena
    w = x1 - x0
    h = y1 - y0
    return (x0 + margin * w, x1 - margin * w,
            y0 + margin * h, y1 - margin * h)


# Derived center / inner zone — keep as a tuple at import time so callers
# can use it as a constant. If you want a different margin for a one-off
# analysis, call _derive_inner(ARENA, your_margin) directly.
INNER_ZONE: tuple[float, float, float, float] = _derive_inner(ARENA, INNER_MARGIN)

# Frame rate for all sessions in this dataset
FPS: float = 30.0
