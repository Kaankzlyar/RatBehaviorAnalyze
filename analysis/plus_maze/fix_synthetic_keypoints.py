"""
fix_synthetic_keypoints.py
--------------------------
Sadece body_center gerçek veriye sahip sentetik PlusMaze CSV'lerindeki
sınır dışı keypoint'leri düzeltir.

Yöntem:
  Her keypoint için body_center → keypoint vektörü ölçeklenerek
  maksimum geçerli mesafe binary search ile bulunur.
  body_center dokunulmaz; yön korunur; anatomi bozulmaz.

Sentetik subjectler:
  control        : MA1_4  MA1_5  MA2_1  MA2_2
  ASP            : MA3_4  MA3_5  MA4_1–MA4_5
  Greyfurt       : MA5_4  MA5_5  MA6_1–MA6_5
  ASP ve Greyfurt: MA7_4  MA7_5  MA8_1–MA8_5

Gereksinim:
  data/arm_coords.json  (show_frame_coords.py ile üretilmiş olmalı)
"""

import json
import pathlib

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DLCDIR  = ROOT / "data" / "DLCfiltered"
ARMFILE = ROOT / "data" / "arm_coords.json"

SYNTHETIC = {
    "control"        : ["MA1_4", "MA1_5", "MA2_1", "MA2_2"],
    "ASP"            : ["MA3_4", "MA3_5", "MA4_1", "MA4_2", "MA4_3", "MA4_4", "MA4_5"],
    "Greyfurt"       : ["MA5_4", "MA5_5", "MA6_1", "MA6_2", "MA6_3", "MA6_4", "MA6_5"],
    "ASP ve Greyfurt": ["MA7_4", "MA7_5", "MA8_1", "MA8_2", "MA8_3", "MA8_4", "MA8_5"],
}

BODY_PARTS = [
    "nose", "head", "left_ear", "right_ear", "body_center",
    "left_forepaw", "right_forepaw", "left_hindpaw", "right_hindpaw", "tail_base",
]

# Sütun indeksleri (0: frame_idx, sonra her bodypart için x,y,likelihood)
BP_COL = {bp: (1 + i * 3, 2 + i * 3) for i, bp in enumerate(BODY_PARTS)}
# nose:(1,2)  head:(4,5)  ...  body_center:(13,14)  ...  tail_base:(28,29)

MARGIN = 8  # maze sınırlarına px cinsinden tolerans


def build_regions(arm_coords):
    """Maze'in geçerli bölgelerini döndürür (4 kol + merkez kavşak)."""
    regions = []
    for v in arm_coords.values():
        regions.append((v[0] - MARGIN, v[1] + MARGIN,
                        v[2] - MARGIN, v[3] + MARGIN))

    # Dikey kol x aralığı × yatay kol y aralığı = merkez kavşak
    v_xn = min(arm_coords["top_arm"][0], arm_coords["bottom_arm"][0]) - MARGIN
    v_xx = max(arm_coords["top_arm"][1], arm_coords["bottom_arm"][1]) + MARGIN
    h_yn = min(arm_coords["left_arm"][2], arm_coords["right_arm"][2]) - MARGIN
    h_yx = max(arm_coords["left_arm"][3], arm_coords["right_arm"][3]) + MARGIN
    regions.append((v_xn, v_xx, h_yn, h_yx))
    return regions


def in_maze(px, py, regions):
    return any(xn <= px <= xx and yn <= py <= yx
               for (xn, xx, yn, yx) in regions)


def clamp_to_nearest(px, py, regions):
    """En yakın geçerli noktaya sıkıştır (fallback)."""
    best_d, best = float("inf"), (px, py)
    for (xn, xx, yn, yx) in regions:
        cx = max(xn, min(xx, px))
        cy = max(yn, min(yx, py))
        d = (cx - px) ** 2 + (cy - py) ** 2
        if d < best_d:
            best_d, best = d, (cx, cy)
    return best


def scale_to_maze(anchor_x, anchor_y, pt_x, pt_y, regions):
    """
    body_center (anchor) → keypoint (pt) vektörünü binary search ile
    ölçekle; keypoint maze içinde kalacak maksimum konumu bul.
    """
    if in_maze(pt_x, pt_y, regions):
        return pt_x, pt_y

    # Binary search: s=1 → orijinal; s=0 → anchor
    lo, hi = 0.0, 1.0
    for _ in range(26):          # ~1/67M hassasiyet
        mid = (lo + hi) * 0.5
        mx = anchor_x + mid * (pt_x - anchor_x)
        my = anchor_y + mid * (pt_y - anchor_y)
        if in_maze(mx, my, regions):
            lo = mid
        else:
            hi = mid

    nx = anchor_x + lo * (pt_x - anchor_x)
    ny = anchor_y + lo * (pt_y - anchor_y)

    # anchor da dışarıdaysa (olmamalı ama güvenli taraf)
    if not in_maze(nx, ny, regions):
        return clamp_to_nearest(pt_x, pt_y, regions)
    return nx, ny


def fix_csv(csv_path, regions):
    """Tek CSV'yi yerinde düzeltir; düzeltilen keypoint sayısını döndürür."""
    with open(csv_path, "r", newline="") as f:
        raw = f.read()

    lines     = raw.splitlines(keepends=True)
    headers   = lines[:3]
    data_lines = lines[3:]

    bx, by = BP_COL["body_center"]   # sütun 13, 14
    modified = 0
    out_lines = []

    for line in data_lines:
        stripped = line.rstrip("\r\n")
        if not stripped:
            out_lines.append(line)
            continue

        parts = stripped.split(",")

        try:
            bc_x = float(parts[bx])
            bc_y = float(parts[by])
        except (ValueError, IndexError):
            out_lines.append(line)
            continue

        for bp in BODY_PARTS:
            if bp == "body_center":
                continue
            xi, yi = BP_COL[bp]
            try:
                px = float(parts[xi])
                py = float(parts[yi])
            except (ValueError, IndexError):
                continue
            if px != px or py != py:   # NaN kontrolü
                continue

            nx, ny = scale_to_maze(bc_x, bc_y, px, py, regions)
            if abs(nx - px) > 0.001 or abs(ny - py) > 0.001:
                parts[xi] = f"{nx:.7f}"
                parts[yi] = f"{ny:.7f}"
                modified += 1

        ending = "\r\n" if line.endswith("\r\n") else "\n"
        out_lines.append(",".join(parts) + ending)

    with open(csv_path, "w", newline="") as f:
        f.writelines(headers)
        f.writelines(out_lines)

    return modified


def main():
    if not ARMFILE.exists():
        print(f"[HATA] arm_coords.json bulunamadi: {ARMFILE}")
        print("       Once show_frame_coords.py --video <video> calistirin.")
        return

    with open(ARMFILE) as f:
        arm_coords = json.load(f)

    regions = build_regions(arm_coords)
    print(f"Maze bolgeleri: {len(regions)} dikdortgen ({MARGIN}px margin)")
    for name, vals in arm_coords.items():
        print(f"  {name:>16}: x[{vals[0]}-{vals[1]}]  y[{vals[2]}-{vals[3]}]")
    print()

    total_files = total_kp = 0

    for group, subjects in SYNTHETIC.items():
        group_dir = DLCDIR / group
        for subj in subjects:
            csv_path = group_dir / f"PlusMaze{subj}" / f"PlusMaze{subj}.csv"
            if not csv_path.exists():
                print(f"  [ATLA] {group:>16} / PlusMaze{subj}: CSV yok")
                continue
            n = fix_csv(csv_path, regions)
            flag = "  [OK]  " if n == 0 else f"  [FIX] "
            print(f"{flag}{group:>16} / PlusMaze{subj}: {n} keypoint duzeltildi")
            total_files += 1
            total_kp += n

    print(f"\nToplam: {total_files} dosya islendi, {total_kp} keypoint duzeltildi.")
    if total_kp > 0:
        print("\nSimdiki adim: orbit_plot.py ve ethological_features.py yeniden calistir.")


if __name__ == "__main__":
    main()
