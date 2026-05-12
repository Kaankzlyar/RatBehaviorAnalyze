# -*- coding: utf-8 -*-
"""
ethological_features.py
-----------------------
Plus Maze DLC verisinden pose-bazli etolojik davranis tespiti.

Tespit edilen davranislar:
  Grooming  — Kendini temizleme (OFT ile esit hassasiyet: 4-dal mantigi)
  Rearing   — Ayaga kalkma (EPM kollarına adapte 6-kural mantigi)
  SAP       — Stretch-Attend Posture: vucut uzuyor + hareketsiz

----------------------------------------------------------------------
GROOMING — 4 dal, herhangi biri yeterliyse tetiklenir:
  1. Tight    — burun patiye cok yakin (fpn < 22px) + fp alcakta + hareketsiz
  2. Loose    — burun biraz uzak (fpn < 35px) + fp alcakta + hareketsiz
  3. Upright  — fp yuksekte (oturarak temizlik) + burun kol ucundan uzakta
  4. Occluded — burun kapali (pati altinda) + hareketsiz + fp yuksekte
  Son adim: Grooming = posture AND NOT rearing

REARING — 6 kural, herhangi biri yeterliyse tetiklenir
  (arm_coords.json dosyasi gerektirir; dosya yoksa rearing hesaplanmaz):
  R1 Compact    — govde kisaliyor (htdist < 75px), grooming degil
  R2 Top arm    — fp yukari kalkmis + burun top kol ucuna yakin
  R3/R4 Bot arm — fp asagi sarkmis + burun bottom kol ucuna yakin
  R5 Side arm   — govde kompakt + burun left/right kol ucuna yakin
  R6 Wall-press — burun kol siniri disinda + govde kisalmis (kapali kollar)

SAP:
  vucut boyu > 102px AND hiz < 15 px/s

----------------------------------------------------------------------
Arm koordinat dosyasi:
  tmaze_metrics.py calistirildiginda her subject klasorune kaydedilir:
    data/DLCfiltered/<kohort>/PlusMaze<ID>/PlusMaze<ID>_arm_coords.json
  Bu dosya yoksa rearing atlanir; grooming + SAP yine hesaplanir.

Ciktilar:
  reports/ethological_metrics_epm.csv
  reports/figures/plus_maze/ethological_grooming.png
  reports/figures/plus_maze/ethological_rearing.png  (arm_coords mevcutsa)
  reports/figures/plus_maze/ethological_sap.png
  reports/figures/plus_maze/ethological_summary.png

Kullanim:
  python analysis/plus_maze/ethological_features.py
"""

import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"]        = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"]         = 150
matplotlib.rcParams["pdf.fonttype"]       = 42
matplotlib.rcParams["ps.fonttype"]        = 42

# ── Proje yolu ─────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DLC_DIR = ROOT / "data" / "DLCfiltered"
REPORTS = ROOT / "reports"
FIGS    = ROOT / "reports" / "figures" / "plus_maze"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Sabitler ──────────────────────────────────────────────────────────────────
FPS               = 30
LIKELIHOOD_THRESH = 0.6

# Grooming esikleri (OFT behavior_detection.py ile hizalanmis)
GROOMING_FPN_TIGHT   = 22.0   # px — tight: burun patiye cok yakin
GROOMING_FPN_LOOSE   = 35.0   # px — loose: genis pencere (sirtust/gevrek temizlik)
GROOM_MAX_FPHP       = 10.0   # px — fp yukseltme siniri (upright/alcak ayrimi)
GROOM_MAX_VEL        = 25.0   # px/s — OFT ile ayni hiz siniri
GROOM_VEL_WINDOW     = 5      # kare — hiz kayan pencere (OFT: 5)

# SAP esikleri
SAP_BL_THRESH  = 102.0  # px — vucut boyu p75 (uzama = SAP)
SAP_MAX_VEL    = 15.0   # px/s — SAP icin katı durma siniri (0.50 px/kare @ 30fps)

# Rearing esikleri (OFT behavior_detection.py'den EPM'e uyarlandi)
REAR_COMPACT_HTDIST    = 75    # px — head-to-tail kisa: govde sikismis
REAR_EXTEND_FPHP       = 40    # px — fp yukari kalkmis (top kol rearing)
REAR_BOTTOM_FPHP       = -40   # px — fp asagi sarkmis (bottom kol rearing)
REAR_BOTTOM_HTDIST     = 105   # px — bottom kol compact rearing icin govde siniri
REAR_SIDE_HTDIST       = 95    # px — yan kol (left/right) rearing icin govde siniri
NEAR_ARM_END_PX        = 40    # px — kol ucuna "yakin" sayilan mesafe
REAR_WALL_PRESS_HTDIST = 115   # px — wall-press rearing (kapali kollar)

# Body-still parametreleri (OFT'tan alinmistir)
BODY_STILL_WINDOW  = 30   # kare (~1s)
BODY_STILL_MIN     = 10   # min gecerli kare
BODY_STILL_RANGE   = 15   # px — pozisyon aralik siniri
BODY_INSIDE_MARGIN = 20   # px — kol ucu marji (body_inside tanimı)

# Bout parametreleri
MIN_BOUT_FRAMES  = 15    # kare (0.5s)
MERGE_GAP_FRAMES = 10    # kare (0.33s)

COHORT_ORDER  = ["Control", "Aspartame", "Grapefruit", "ASP+Greyfurt"]
COHORT_COLORS = {
    "Control":      "#4CAF50",
    "Aspartame":    "#2196F3",
    "Grapefruit":   "#FF9800",
    "ASP+Greyfurt": "#9C27B0",
}
COHORT_DISPLAY = {
    "Control":      "Kontrol",
    "Aspartame":    "Aspartam",
    "Grapefruit":   "Greyfurt",
    "ASP+Greyfurt": "Aspartam+Greyfurt",
}


def load_cohort_map() -> dict:
    pm = pd.read_csv(ROOT / "data" / "plus_maze_metrics_all.csv")
    return dict(zip(pm["subject_id"].str.replace("PlusMaze", ""), pm["cohort"]))


# ── DLC CSV okuyucu ───────────────────────────────────────────────────────────

def load_dlc(csv_path: pathlib.Path) -> pd.DataFrame:
    """3 satirlik baslik (scorer/bodypart/coord) ile yukle."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples([(b, c) for _, b, c in df.columns])
    return df


def get_keypoint(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """x, y dizisi; likelihood < esik olan kareler NaN."""
    x  = df[name]["x"].astype(float).values.copy()
    y  = df[name]["y"].astype(float).values.copy()
    lk = df[name]["likelihood"].astype(float).values
    bad = lk < LIKELIHOOD_THRESH
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y


def dist(x1, y1, x2, y2) -> np.ndarray:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ── Arm koordinat yukleyici ───────────────────────────────────────────────────

def load_arm_coords(csv_path: pathlib.Path) -> dict | None:
    """
    PlusMaze<ID>_arm_coords.json yukler.
    tmaze_metrics.py tarafindan kaydedilir; yoksa None dondurur.
    """
    json_path = csv_path.parent / f"{csv_path.stem}_arm_coords.json"
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: tuple(v) for k, v in data.items()}


# ── Bout tespit motoru ────────────────────────────────────────────────────────

def detect_bouts(signal: np.ndarray,
                 min_frames: int = MIN_BOUT_FRAMES,
                 merge_gap: int  = MERGE_GAP_FRAMES) -> list[tuple[int, int]]:
    """1D binary sinyalden bout baslangic-bitis listesi; kisa araliklar birlestirilir."""
    sig     = np.nan_to_num(signal.astype(float)).astype(bool)
    padded  = np.concatenate([[0], sig, [0]])
    changes = np.diff(padded.astype(int))
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    bouts   = list(zip(starts, ends))

    merged = []
    for s, e in bouts:
        if merged and (s - merged[-1][1]) <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged if (e - s) >= min_frames]


def summarize_bouts(bouts: list[tuple[int, int]],
                    total_frames: int,
                    fps: float = FPS) -> dict:
    n        = len(bouts)
    total_s  = sum(e - s for s, e in bouts) / fps
    mean_s   = (total_s / n) if n > 0 else 0.0
    max_s    = (max(e - s for s, e in bouts) / fps) if n > 0 else 0.0
    pct_time = total_s / (total_frames / fps) * 100 if total_frames > 0 else 0.0
    frag_idx = n / total_s if total_s > 0 else 0.0   # bout/s
    return {
        "bout_count": n,
        "total_s":    round(total_s, 2),
        "pct_time":   round(pct_time, 2),
        "mean_s":     round(mean_s, 2),
        "max_s":      round(max_s, 2),
        "frag_idx":   round(frag_idx, 3),
    }


# ── Ana islem: tek sican ──────────────────────────────────────────────────────

def process_subject(csv_path: pathlib.Path) -> dict:
    df = load_dlc(csv_path)
    n  = len(df)

    # ── Tum keypoint'ler ──
    nx,  ny  = get_keypoint(df, "nose")
    hx,  hy  = get_keypoint(df, "head")
    bx,  by  = get_keypoint(df, "body_center")
    tx,  ty  = get_keypoint(df, "tail_base")
    lfx, lfy = get_keypoint(df, "left_forepaw")
    rfx, rfy = get_keypoint(df, "right_forepaw")
    lhx, lhy = get_keypoint(df, "left_hindpaw")
    rhx, rhy = get_keypoint(df, "right_hindpaw")

    # Forepaw / hindpaw ortalama y (NaN-safe)
    with np.errstate(all="ignore"):
        fp_y = np.nanmean(np.stack([lfy, rfy]), axis=0)
        hp_y = np.nanmean(np.stack([lhy, rhy]), axis=0)

    # ── Postural ozellikler ──
    fpn        = np.fmin(dist(lfx, lfy, nx, ny),
                         dist(rfx, rfy, nx, ny))   # forepaw-nose (min)
    fp_hp_vert = hp_y - fp_y                        # + = fp yukarda (imaj koordinati)
    htdist     = dist(hx, hy, tx, ty)               # head-to-tail mesafesi
    body_len   = dist(nx, ny, tx, ty)               # vucut boyu (SAP icin)

    # Hiz: kayan ortalama px/s — OFT ile ayni (5 kare pencere)
    spd_raw  = np.concatenate([[np.nan],
                                np.sqrt(np.diff(bx) ** 2 + np.diff(by) ** 2)])
    body_vel = (pd.Series(spd_raw)
                .rolling(GROOM_VEL_WINDOW, center=True, min_periods=1)
                .mean()
                .values * FPS)

    # Body-still: 1s pencerede pozisyon araligi — OFT'tan alinmistir
    bx_win = pd.Series(bx).rolling(BODY_STILL_WINDOW, min_periods=BODY_STILL_MIN, center=True)
    by_win = pd.Series(by).rolling(BODY_STILL_WINDOW, min_periods=BODY_STILL_MIN, center=True)
    body_still = (
        ((bx_win.max() - bx_win.min()) < BODY_STILL_RANGE) &
        ((by_win.max() - by_win.min()) < BODY_STILL_RANGE)
    ).values

    # ── Arm koordinatlarindan turetilen maskeler ──
    arm_coords = load_arm_coords(csv_path)

    if arm_coords is not None:
        arm_b = arm_coords["bottom_arm"]   # (xmin, xmax, ymin, ymax)
        arm_t = arm_coords["top_arm"]
        arm_l = arm_coords["left_arm"]
        arm_r = arm_coords["right_arm"]

        # Kol ucuna yakinlik (burun koordinatina gore)
        near_top    = ny < (arm_t[2] + NEAR_ARM_END_PX)    # top arm: kucuk y = yukari
        near_bottom = ny > (arm_b[3] - NEAR_ARM_END_PX)    # bottom arm: buyuk y = asagi
        near_left   = nx < (arm_l[0] + NEAR_ARM_END_PX)    # left arm: kucuk x = sol
        near_right  = nx > (arm_r[1] - NEAR_ARM_END_PX)    # right arm: buyuk x = sag

        # Tum labirent siniri (nose_inside kontrolu)
        maze_xmin = min(arm_b[0], arm_t[0], arm_l[0], arm_r[0])
        maze_xmax = max(arm_b[1], arm_t[1], arm_l[1], arm_r[1])
        maze_ymin = min(arm_b[2], arm_t[2], arm_l[2], arm_r[2])
        maze_ymax = max(arm_b[3], arm_t[3], arm_l[3], arm_r[3])

        nose_inside  = ((nx > maze_xmin) & (nx < maze_xmax) &
                        (ny > maze_ymin) & (ny < maze_ymax))
        nose_outside = ~nose_inside & ~np.isnan(nx)

        # body_inside: kol uclarindan uzakta (grooming occluded dali icin)
        body_inside = ~(
            (bx < arm_l[0] + BODY_INSIDE_MARGIN) |
            (bx > arm_r[1] - BODY_INSIDE_MARGIN) |
            (by < arm_t[2] + BODY_INSIDE_MARGIN) |
            (by > arm_b[3] - BODY_INSIDE_MARGIN)
        )
    else:
        near_top = near_bottom = near_left = near_right = np.zeros(n, dtype=bool)
        nose_inside  = ~np.isnan(nx)
        nose_outside = np.zeros(n, dtype=bool)
        body_inside  = ~np.isnan(bx)

    # ── GROOMING: 4 dal (OFT ile esit hassasiyet) ──────────────────────────────
    # Hiz kapisi: tum dallar icin ortak
    still = (body_vel < GROOM_MAX_VEL) & ~np.isnan(body_vel)

    # Dal 1: Tight — burun patiye cok yakin, fp alcakta
    groom_tight = (still
                   & (fpn < GROOMING_FPN_TIGHT)
                   & (fp_hp_vert < GROOM_MAX_FPHP)
                   & nose_inside)

    # Dal 2: Loose — daha genis pencere, fp alcakta (sirtust/yan yatis temizlik)
    groom_loose = (still
                   & (fpn < GROOMING_FPN_LOOSE)
                   & (fp_hp_vert < GROOM_MAX_FPHP)
                   & nose_inside)

    # Dal 3: Upright — fp yuksekte, oturarak temizlik; kol ucundan uzakta
    # ~near_top: top arm ucunda fp yukari kalkmasi rearing olabilir
    groom_upright = (still
                     & (fpn < GROOMING_FPN_LOOSE)
                     & (fp_hp_vert > GROOM_MAX_FPHP)
                     & nose_inside
                     & ~near_top)

    # Dal 4: Occluded — burun kapali (pati altinda), body_still + fp yuksekte
    groom_occluded = (np.isnan(ny)
                      & body_still
                      & body_inside
                      & (fp_hp_vert > GROOM_MAX_FPHP)
                      & ~np.isnan(fp_hp_vert))

    grooming_posture = groom_tight | groom_loose | groom_upright | groom_occluded

    # ── REARING: EPM'e adapte 6 kural ─────────────────────────────────────────
    if arm_coords is not None:
        feat_ok = ~np.isnan(htdist) & ~np.isnan(fp_hp_vert)

        # R1: Compact — govde sikisiyor, grooming degil, rahat oturma degil
        sitting_upright = body_still & body_inside
        rear_compact    = ((htdist < REAR_COMPACT_HTDIST)
                           & ~grooming_posture
                           & ~sitting_upright
                           & feat_ok)

        # R2: Top arm extended — fp yukari + burun top kolun ucuna yakin
        rear_top        = ((fp_hp_vert > REAR_EXTEND_FPHP)
                           & near_top
                           & feat_ok)

        # R3: Bottom arm strong — fp belirgin asagi + bottom ucuna yakin
        rear_bot_strong = ((fp_hp_vert < REAR_BOTTOM_FPHP)
                           & near_bottom
                           & feat_ok)

        # R4: Bottom arm compact — orta sinyal + govde sikismis
        rear_bot_cmpct  = ((fp_hp_vert < -30)
                           & near_bottom
                           & (htdist < REAR_BOTTOM_HTDIST)
                           & feat_ok)

        # R5: Side arm — govde kompakt + left/right kol ucuna yakin
        rear_side       = ((htdist < REAR_SIDE_HTDIST)
                           & (near_left | near_right)
                           & ~grooming_posture
                           & feat_ok)

        # R6: Wall-press — burun labirent disinda + govde sikismis
        #     Sadece kapali kollar (top/bottom); acik kollarda duvar yok
        rear_wall_press = (nose_outside
                           & (htdist < REAR_WALL_PRESS_HTDIST)
                           & (near_bottom | near_top)
                           & feat_ok)

        rearing = (rear_compact | rear_top
                   | rear_bot_strong | rear_bot_cmpct
                   | rear_side | rear_wall_press)
    else:
        rearing = np.zeros(n, dtype=bool)

    # Son adim: grooming, rearing ile cakisan kareleri dislar
    groom_sig = grooming_posture & ~rearing

    # ── SAP ────────────────────────────────────────────────────────────────────
    sap_ok  = ~np.isnan(body_len) & ~np.isnan(body_vel)
    sap_sig = sap_ok & (body_len > SAP_BL_THRESH) & (body_vel < SAP_MAX_VEL)

    # ── Bout tespiti ──
    groom_bouts = detect_bouts(groom_sig.astype(float))
    rear_bouts  = detect_bouts(rearing.astype(float))
    sap_bouts   = detect_bouts(sap_sig.astype(float))

    result = {
        "n_frames":    n,
        "session_s":   round(n / FPS, 1),
        "has_rearing": arm_coords is not None,
    }
    for prefix, bouts in [("groom", groom_bouts),
                           ("rear",  rear_bouts),
                           ("sap",   sap_bouts)]:
        for k, v in summarize_bouts(bouts, n).items():
            result[f"{prefix}_{k}"] = v

    result["groom_signal_pct"] = round(float(np.nanmean(groom_sig)) * 100, 2)
    result["sap_signal_pct"]   = round(float(np.nanmean(sap_sig))   * 100, 2)

    return result


# ── Tum sicanlari isle ────────────────────────────────────────────────────────

def run_all() -> pd.DataFrame:
    cohort_map = load_cohort_map()
    pm_csvs    = sorted([
        p for p in DLC_DIR.rglob("*.csv")
        if "PlusMaze" in p.name and "metrics" not in p.name
    ])

    rows = []
    for csv_path in pm_csvs:
        m = re.search(r"PlusMaze(MA\d+_\d+)", csv_path.name, re.IGNORECASE)
        if not m:
            continue
        subj_id = m.group(1)
        cohort  = cohort_map.get(subj_id, "Unknown")

        has_coords = (csv_path.parent / f"{csv_path.stem}_arm_coords.json").exists()
        tag = "arm_coords" if has_coords else "no coords => rearing skip"
        print(f"  {subj_id:10s} ({cohort}) [{tag}]...", end=" ")

        try:
            metrics = process_subject(csv_path)
            metrics["subject_id"] = subj_id
            metrics["cohort"]     = cohort
            rows.append(metrics)
            print(
                f"groom={metrics['groom_bout_count']}bouts/{metrics['groom_total_s']:.1f}s  "
                f"rear={metrics['rear_bout_count']}bouts/{metrics['rear_total_s']:.1f}s  "
                f"SAP={metrics['sap_bout_count']}bouts/{metrics['sap_total_s']:.1f}s"
            )
        except Exception as e:
            print(f"HATA: {e}")

    df = pd.DataFrame(rows)
    first_cols = ["subject_id", "cohort", "n_frames", "session_s", "has_rearing"]
    rest = [c for c in df.columns if c not in set(first_cols)]
    return df[first_cols + rest].sort_values(["cohort", "subject_id"])


# ── Gorsellestime: nokta + ortalama cizgisi + SD golgesi ─────────────────────

def draw_dot_mean_sd(ax, xi: int, vals: np.ndarray, color: str,
                     rng: np.random.Generator,
                     w: float = 0.28, jitter: float = 0.12) -> None:
    mean_v = float(np.mean(vals))
    sd_v   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    ax.add_patch(plt.Rectangle(
        (xi - w, mean_v - sd_v), 2 * w, 2 * sd_v,
        color=color, alpha=0.13, zorder=1, linewidth=0,
    ))
    ax.hlines(mean_v, xi - w, xi + w, colors=color, linewidth=3.0, zorder=3)

    jit_x = rng.uniform(-jitter, jitter, size=len(vals))
    for v, jx in zip(vals, jit_x):
        ax.plot([xi + jx, xi], [v, mean_v],
                color=color, alpha=0.28, linewidth=1.2, zorder=2)
    ax.scatter(xi + jit_x, vals,
               color=color, s=115, edgecolors="white",
               linewidths=1.3, zorder=5, alpha=0.95)
    ax.scatter(xi, mean_v, marker="D", color="white", s=90,
               edgecolors=color, linewidths=2.2, zorder=6)


def plot_behavior_panel(df: pd.DataFrame, metric_col: str,
                        ylabel: str, title: str,
                        out_path: pathlib.Path) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.12, right=0.97)

    groups = [g for g in COHORT_ORDER if g in df["cohort"].values]
    x_pos  = list(range(len(groups)))
    rng    = np.random.default_rng(42)

    for xi, grp in zip(x_pos, groups):
        vals = df.loc[df["cohort"] == grp, metric_col].dropna().values
        if len(vals) == 0:
            continue
        draw_dot_mean_sd(ax, xi, vals, COHORT_COLORS[grp], rng)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([COHORT_DISPLAY[g] for g in groups], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    cohort_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.8, label=COHORT_DISPLAY[g])
        for g in COHORT_ORDER
    ]
    ax.legend(handles=cohort_patches, loc="upper right", fontsize=10, frameon=False)

    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
               markersize=11, label="= Fare"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor="#555555", markeredgewidth=2.2,
               markersize=10, label="= Grup Ortalaması"),
        Line2D([0], [0], color="#555555", linewidth=2.5, label="= Ortalama"),
    ]
    fig.legend(
        handles=marker_handles, loc="lower center", ncol=3,
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01),
        handlelength=1.5, handletextpad=0.4, columnspacing=1.2,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_summary(df: pd.DataFrame) -> None:
    has_rearing = bool(df.get("has_rearing", pd.Series(False)).any())

    metrics = [
        ("groom_pct_time", "% Oturum Suresi", "Grooming — Kendini Temizleme"),
        ("sap_pct_time",   "% Oturum Suresi", "SAP — Uzanma-Degerlendirme Durusu"),
    ]
    if has_rearing:
        metrics.append(
            ("rear_pct_time", "% Oturum Suresi", "Rearing — Ayaga Kalkma")
        )

    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.06, right=0.98, wspace=0.35)

    groups = [g for g in COHORT_ORDER if g in df["cohort"].values]
    x_pos  = list(range(len(groups)))
    rng    = np.random.default_rng(42)

    rear_df = df[df["has_rearing"] == True] if has_rearing else df

    for ax, (col, ylabel, title) in zip(axes, metrics):
        src = rear_df if col == "rear_pct_time" else df
        for xi, grp in zip(x_pos, groups):
            vals = src.loc[src["cohort"] == grp, col].dropna().values
            if len(vals) == 0:
                continue
            draw_dot_mean_sd(ax, xi, vals, COHORT_COLORS[grp], rng)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [COHORT_DISPLAY[g] for g in groups], fontsize=9, rotation=0, ha="center",
        )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=COHORT_COLORS[g], alpha=0.8, label=COHORT_DISPLAY[g])
        for g in COHORT_ORDER
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Plus Maze — Etolojik Davranislar", fontsize=11, fontweight="bold", y=0.99)

    out = FIGS / "ethological_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ── Kohort tablosu ────────────────────────────────────────────────────────────

def print_cohort_summary(df: pd.DataFrame) -> None:
    print("\n=== Kohort Ortalamalari ===")
    want = ["groom_bout_count", "groom_pct_time", "groom_mean_s",
            "rear_bout_count",  "rear_pct_time",  "rear_mean_s",
            "sap_bout_count",   "sap_pct_time",   "sap_mean_s"]
    cols = [c for c in want if c in df.columns]
    grp  = df.groupby("cohort")[cols].mean().round(2).reindex(COHORT_ORDER)
    print(grp.to_string())
    print()
    sig_cols = [c for c in ["groom_signal_pct", "sap_signal_pct"] if c in df.columns]
    if sig_cols:
        print("Ham sinyal yuzdeleri (esik kontrolu):")
        print(df.groupby("cohort")[sig_cols].mean().round(1)
              .reindex(COHORT_ORDER).to_string())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Plus Maze etolojik ozellikler hesaplaniyor...\n")
    print(
        f"Grooming : fpn_tight<{GROOMING_FPN_TIGHT}px  fpn_loose<{GROOMING_FPN_LOOSE}px  "
        f"vel<{GROOM_MAX_VEL}px/s  (4 dal: tight/loose/upright/occluded)\n"
        f"Rearing  : compact htdist<{REAR_COMPACT_HTDIST}px  |  "
        f"top fp_hp>{REAR_EXTEND_FPHP}  |  bot fp_hp<{REAR_BOTTOM_FPHP}  |  "
        f"near_end={NEAR_ARM_END_PX}px  (arm_coords.json gerekli)\n"
        f"SAP      : body_len>{SAP_BL_THRESH}px  vel<{SAP_MAX_VEL}px/s\n"
        f"Bout     : min={MIN_BOUT_FRAMES}kare ({MIN_BOUT_FRAMES/FPS:.1f}s)  "
        f"merge_gap={MERGE_GAP_FRAMES}kare ({MERGE_GAP_FRAMES/FPS:.2f}s)\n"
    )

    df = run_all()

    out_csv = REPORTS / "ethological_metrics_epm.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[ok] {out_csv}")

    print_cohort_summary(df)

    plot_behavior_panel(
        df, "groom_pct_time", "% Oturum Suresi",
        "Grooming — Kendini Temizleme",
        FIGS / "ethological_grooming.png",
    )

    has_rearing = bool(df.get("has_rearing", pd.Series(False)).any())
    if has_rearing:
        rear_df = df[df["has_rearing"] == True]
        plot_behavior_panel(
            rear_df, "rear_pct_time", "% Oturum Suresi",
            "Rearing — Ayaga Kalkma",
            FIGS / "ethological_rearing.png",
        )

    plot_behavior_panel(
        df, "sap_pct_time", "% Oturum Suresi",
        "SAP — Uzanma-Degerlendirme Durusu",
        FIGS / "ethological_sap.png",
    )
    plot_summary(df)

    print("\n[done] Tum ciktilar reports/ ve reports/figures/ altinda.")


if __name__ == "__main__":
    main()
