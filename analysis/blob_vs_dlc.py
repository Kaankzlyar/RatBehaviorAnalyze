"""
analysis/blob_vs_dlc.py
-----------------------
Validate a blob-based rearing/grooming detector (HSV-blob bounding box from
the professor's MATLAB pipeline) against the DLC-pose-based detector on
subjects where both pipelines have run (MA1, MA3, MA5, MA7).

Why this exists
---------------
MA2/4/6/8 only have ``_res.mat`` (no DLC pose, no video).  If a coarse
blob-based detector agrees well enough with the multi-bodypart DLC detector
(frame-level F1 >= ~0.65), we can apply it to MA2/4/6/8 and double the
sample size for the rearing/grooming classifier.  This script measures
that agreement and writes a markdown report.

Inputs
------
    data/DLCfiltered/Kare/{control,ASP,Greyfurt,ASP ve Greyfurt}/MA{N}-{R}_res.mat
        Variables: xc, yc           (centroid in pixels)
                   amn, amx          (blob row/y bounds)
                   bmn, bmx          (blob col/x bounds)
                   t                 (timestamps in seconds)
                   Center, Sides     (in/out of inner zone — unused here)

    data/DLCfiltered/{control,ASP,Greyfurt,ASP ve Greyfurt}/
        OpenFieldMA{N}_{R}/OpenFieldMA{N}_{R}_behavior_frames.csv
        Columns: frame, time_s, behaviour in {rearing, grooming, other}

Outputs
-------
    docs/BLOB_VS_DLC_REPORT.md
    docs/figures/blob_vs_dlc_{subject}_timeline.png

Usage
-----
    pip install scipy numpy pandas matplotlib scikit-learn
    python analysis/blob_vs_dlc.py --cohorts MA1 MA3
    python analysis/blob_vs_dlc.py --cohorts MA1 MA3 MA5 MA7

Tweak detection thresholds at the top of this file and re-run; the report
re-generates from scratch each time.
"""

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "DLCfiltered"
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"

COHORT_FOLDER = {
    "MA1": "control",
    "MA3": "ASP",
    "MA5": "Greyfurt",
    "MA7": "ASP ve Greyfurt",
}

# ── Detection thresholds ───────────────────────────────────────────────────────
FPS = 30.0

# Rearing — top-down projected area shrinks when the rat stands vertically,
# and the bounding box becomes squarer.
LOCOMOTING_VEL_THRESH = 60.0     # px/s — frames where the rat is clearly walking
REAR_AREA_FRAC        = 0.55     # rearing if blob_area < FRAC * locomoting baseline
REAR_ASPECT_MAX       = 1.6      # bbox aspect ratio (long/short side)

# Grooming — slow centroid AND stable blob shape AND not rearing.
GROOM_VEL_MAX         = 25.0     # px/s
GROOM_VEL_WINDOW      = 5        # frames — velocity smoothing
GROOM_AREA_CV_MAX     = 0.18     # rolling std/mean of blob_area
GROOM_STABLE_WINDOW   = 30       # ≈ 1 s

# Bout grouping — matches src/behavior_detection.py
INTER_BOUT_GAP   = 15
MIN_BOUT_FRAMES  = 10

CLASSES = ["other", "rearing", "grooming"]


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_blob_mat(path: pathlib.Path) -> dict:
    m = loadmat(str(path))
    return {
        "xc":  np.asarray(m["xc"]).flatten().astype(float),
        "yc":  np.asarray(m["yc"]).flatten().astype(float),
        "amn": np.asarray(m["amn"]).flatten().astype(float),
        "amx": np.asarray(m["amx"]).flatten().astype(float),
        "bmn": np.asarray(m["bmn"]).flatten().astype(float),
        "bmx": np.asarray(m["bmx"]).flatten().astype(float),
        "t":   np.asarray(m["t"]).flatten().astype(float),
    }


# ── Blob feature extraction ────────────────────────────────────────────────────
def blob_features(d: dict) -> pd.DataFrame:
    blob_w = d["bmx"] - d["bmn"]
    blob_h = d["amx"] - d["amn"]
    blob_area = blob_w * blob_h
    long  = np.maximum(blob_w, blob_h)
    short = np.maximum(np.minimum(blob_w, blob_h), 1e-6)
    aspect = long / short

    dx = np.diff(d["xc"], prepend=d["xc"][0])
    dy = np.diff(d["yc"], prepend=d["yc"][0])
    v_inst = np.sqrt(dx ** 2 + dy ** 2) * FPS
    v_smooth = (
        pd.Series(v_inst)
        .rolling(GROOM_VEL_WINDOW, min_periods=1, center=True)
        .mean()
        .values
    )

    return pd.DataFrame({
        "blob_w":    blob_w,
        "blob_h":    blob_h,
        "blob_area": blob_area,
        "aspect":    aspect,
        "vel":       v_smooth,
    })


# ── Detection rules ────────────────────────────────────────────────────────────
def detect_behaviors(feat: pd.DataFrame) -> np.ndarray:
    area = feat["blob_area"].values
    aspect = feat["aspect"].values
    vel = feat["vel"].values

    locomoting = vel > LOCOMOTING_VEL_THRESH
    if locomoting.sum() >= 30:
        baseline = float(np.nanmedian(area[locomoting]))
    else:
        baseline = float(np.nanmedian(area))

    rear = (area < REAR_AREA_FRAC * baseline) & (aspect < REAR_ASPECT_MAX)

    win = pd.Series(area).rolling(GROOM_STABLE_WINDOW, min_periods=10, center=True)
    cv = (win.std() / win.mean()).fillna(1.0).values
    groom = (vel < GROOM_VEL_MAX) & (cv < GROOM_AREA_CV_MAX) & ~rear

    labels = np.full(len(feat), "other", dtype=object)
    labels[rear]  = "rearing"
    labels[groom] = "grooming"

    labels = _smooth(labels, "rearing")
    labels = _smooth(labels, "grooming")
    return labels


def _smooth(labels: np.ndarray, target: str) -> np.ndarray:
    """Merge gaps <= INTER_BOUT_GAP, drop bouts < MIN_BOUT_FRAMES (per class)."""
    n = len(labels)
    is_target = labels == target

    bouts = []
    in_bout, start = False, 0
    for i in range(n):
        if is_target[i] and not in_bout:
            start, in_bout = i, True
        elif not is_target[i] and in_bout:
            bouts.append((start, i - 1))
            in_bout = False
    if in_bout:
        bouts.append((start, n - 1))

    merged = []
    for s, e in bouts:
        if merged and s - merged[-1][1] - 1 <= INTER_BOUT_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    final = [(s, e) for s, e in merged if e - s + 1 >= MIN_BOUT_FRAMES]

    out = np.where(labels == target, "other", labels)
    for s, e in final:
        out[s:e + 1] = target
    return out


# ── Pair discovery & alignment ─────────────────────────────────────────────────
def find_pairs(cohorts: list[str]) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    pairs = []
    for cohort in cohorts:
        folder = COHORT_FOLDER.get(cohort)
        if folder is None:
            print(f"  [skip] Unknown cohort: {cohort}")
            continue
        kare_dir = DATA / "Kare" / folder
        dlc_dir  = DATA / folder
        if not kare_dir.exists():
            print(f"  [skip] Missing folder: {kare_dir}")
            continue
        for mat in sorted(kare_dir.glob(f"{cohort}-*_res.mat")):
            run = mat.stem.split("-")[1].split("_")[0]
            subject = f"{cohort}_{run}"
            dlc_csv = dlc_dir / f"OpenField{subject}" / f"OpenField{subject}_behavior_frames.csv"
            if dlc_csv.exists():
                pairs.append((subject, mat, dlc_csv))
    return pairs


def align(blob_lab: np.ndarray, dlc_lab: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offset = int(round(float(t[0]) * FPS)) if len(t) else 0
    dlc_aligned = dlc_lab[offset:offset + len(blob_lab)]
    n = min(len(blob_lab), len(dlc_aligned))
    return blob_lab[:n], dlc_aligned[:n]


# ── Metrics & plotting ─────────────────────────────────────────────────────────
def per_subject_metrics(blob: np.ndarray, dlc: np.ndarray) -> dict:
    cm = confusion_matrix(dlc, blob, labels=CLASSES)
    prec, rec, f1, supp = precision_recall_fscore_support(
        dlc, blob, labels=["rearing", "grooming"], zero_division=0
    )
    return {
        "n_frames":  int(len(dlc)),
        "kappa":     float(cohen_kappa_score(dlc, blob)),
        "rear_p":    float(prec[0]),
        "rear_r":    float(rec[0]),
        "rear_f1":   float(f1[0]),
        "rear_n":    int(supp[0]),
        "groom_p":   float(prec[1]),
        "groom_r":   float(rec[1]),
        "groom_f1":  float(f1[1]),
        "groom_n":   int(supp[1]),
        "cm":        cm.tolist(),
    }


def plot_overlay(subject: str, blob: np.ndarray, dlc: np.ndarray, out: pathlib.Path) -> None:
    n = len(dlc)
    if n == 0:
        return
    t = np.arange(n) / FPS
    colors = {"rearing": "#E74C3C", "grooming": "#2ECC71"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 3.2), sharex=True)
    fig.suptitle(f"Blob vs DLC — {subject}", fontsize=11, y=1.02)

    for ax, labs, title in [(axes[0], dlc, "DLC pose"), (axes[1], blob, "Blob box")]:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)
        for cls in ("rearing", "grooming"):
            for i in np.where(labs == cls)[0]:
                ax.axvspan(t[i], t[i] + 1 / FPS, color=colors[cls], alpha=0.7, lw=0)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(0, t[-1])
    handles = [mpatches.Patch(color=colors[c], label=c) for c in ("rearing", "grooming")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Markdown report ────────────────────────────────────────────────────────────
def write_report(subjects: list[dict], cohorts: list[str], md_path: pathlib.Path) -> None:
    L: list[str] = []
    add = L.append

    add("# Blob-Based vs DLC-Based Behavior Detection — Validation Report")
    add("")
    add("> Auto-generated by `analysis/blob_vs_dlc.py`. Re-run to refresh.")
    add("")
    add(f"**Cohorts compared:** {', '.join(cohorts)}  ")
    add(f"**Subjects:** {len(subjects)} (only animals with both `_res.mat` and DLC `_behavior_frames.csv`)")
    add("")

    add("## Goal")
    add("")
    add("Quantify how well a coarse single-blob detector (HSV-thresholded bounding")
    add("box from `sonuc_kutu_final_v1.m`) agrees with the 12-bodypart DLC pose")
    add("detector for **rearing** and **grooming**.  If frame-level F1 is high")
    add("enough we apply the blob detector to MA2/4/6/8 (which only have")
    add("`_res.mat`) and double the rearing/grooming training set.  Otherwise we")
    add("drop those targets from the 24-sample classifier.")
    add("")

    add("## Methodology")
    add("")
    add("### Blob features (per frame)")
    add("")
    add("From each `_res.mat` we read `xc, yc, amn, amx, bmn, bmx` and derive:")
    add("")
    add("| Feature | Formula | Purpose |")
    add("|---|---|---|")
    add("| `blob_w`     | `bmx − bmn`              | horizontal extent (px) |")
    add("| `blob_h`     | `amx − amn`              | vertical extent (px) |")
    add("| `blob_area`  | `blob_w × blob_h`        | top-down projected area (px²) |")
    add("| `aspect`     | `max(w,h) / min(w,h)`    | shape elongation (1 = square) |")
    add("| `vel`        | `√(dxc² + dyc²) · fps`   | centroid speed (px/s, smoothed) |")
    add("")

    add("### Detection rules")
    add("")
    add("**Rearing** — when the rat stands vertically the top-down projection")
    add("compresses; both `blob_area` shrinks and the bounding box becomes")
    add("squarer.  We use a **per-subject locomoting baseline** so the threshold")
    add("self-calibrates to lighting / camera-distance differences:")
    add("")
    add("```")
    add(f"baseline = median(blob_area where vel > {LOCOMOTING_VEL_THRESH:.0f} px/s)")
    add(f"rearing  = (blob_area < {REAR_AREA_FRAC} · baseline) AND (aspect < {REAR_ASPECT_MAX})")
    add("```")
    add("")
    add("**Grooming** — slow centroid AND stable blob shape AND not rearing:")
    add("")
    add("```")
    add(f"area_cv  = (rolling std / rolling mean) of blob_area, ±{GROOM_STABLE_WINDOW//2} frames")
    add(f"grooming = (vel < {GROOM_VEL_MAX:.0f} px/s)")
    add(f"           AND (area_cv < {GROOM_AREA_CV_MAX})")
    add("           AND NOT rearing")
    add("```")
    add("")
    add(f"Bouts are then merged if separated by ≤ {INTER_BOUT_GAP} frames")
    add(f"(~{INTER_BOUT_GAP/FPS:.2f} s) and bursts shorter than {MIN_BOUT_FRAMES} frames")
    add(f"(~{MIN_BOUT_FRAMES/FPS:.2f} s) are dropped — same post-processing as")
    add("`src/behavior_detection.py`.")
    add("")

    add("### Comparison")
    add("")
    add("- Frame-level confusion matrix over `{other, rearing, grooming}`")
    add("- Per-class precision / recall / F1 for rearing and grooming")
    add("- Cohen's κ across all three classes (chance-corrected agreement)")
    add(f"- Frame alignment: blob's `t[0]` (s) → DLC frame `round(t[0] · {FPS:.0f})`,")
    add("  arrays truncated to common length")
    add("- DLC labels are the **reference** (validated against video ground truth")
    add("  on MA1_2 / MA5_1 / MA7_1).  F1 here is _agreement_, not absolute accuracy.")
    add("")
    add("**Decision thresholds**")
    add("")
    add("| Mean F1 (both classes) | Action |")
    add("|---|---|")
    add("| ≥ 0.65 (both) | apply blob detector to MA2/4/6/8 → 24-sample classifier |")
    add("| ≥ 0.65 (one only) | hybrid — blob for that class, drop the other |")
    add("| < 0.65 (both) | fall back to Strategy A — drop rearing/grooming targets |")
    add("")
    add("F1 ≥ 0.65 was chosen because it corresponds to \"substantial\" agreement")
    add("on Cohen's κ (0.61–0.80, Landis & Koch 1977) — defensible as a label")
    add("source for downstream classification.")
    add("")

    if not subjects:
        add("## Results")
        add("")
        add("_No subject pairs were processed.  Either no pairs were found for the")
        add("requested cohorts, or the script has not been run yet._")
        add("")
        _write_reproduce(add, cohorts)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(L), encoding="utf-8")
        return

    rear_f1 = np.array([s["rear_f1"]  for s in subjects])
    groom_f1 = np.array([s["groom_f1"] for s in subjects])
    kappa = np.array([s["kappa"] for s in subjects])

    add("## Population summary")
    add("")
    add(f"Mean Cohen's κ across {len(subjects)} subjects: **{kappa.mean():.3f}** (std {kappa.std():.3f})")
    add("")
    add("| Class | Mean F1 | Std | Min | Max |")
    add("|---|---|---|---|---|")
    add(f"| rearing  | {rear_f1.mean():.3f}  | {rear_f1.std():.3f}  | {rear_f1.min():.3f}  | {rear_f1.max():.3f}  |")
    add(f"| grooming | {groom_f1.mean():.3f} | {groom_f1.std():.3f} | {groom_f1.min():.3f} | {groom_f1.max():.3f} |")
    add("")

    rear_pass  = rear_f1.mean()  >= 0.65
    groom_pass = groom_f1.mean() >= 0.65
    if rear_pass and groom_pass:
        verdict = "✅ **Both classes pass F1 ≥ 0.65** — apply blob detector to MA2/4/6/8 to reach 24 samples for all targets."
    elif rear_pass or groom_pass:
        only = "rearing" if rear_pass else "grooming"
        verdict = (f"⚠ **Only {only} passes F1 ≥ 0.65** — hybrid path: use blob detector "
                   f"for {only} on MA2/4/6/8, drop the other target.")
    else:
        verdict = "❌ **Neither class passes F1 ≥ 0.65** — fall back to Strategy A: drop rearing & grooming targets, train anxiety_level + group on 24 samples × ~11 centroid-only features."
    add(f"**Verdict:** {verdict}")
    add("")

    add("## Per-subject results")
    add("")
    add("| Subject | Frames | κ | Rear P | Rear R | Rear F1 | Rear n_DLC | Groom P | Groom R | Groom F1 | Groom n_DLC |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in subjects:
        add(
            f"| {s['subject']} | {s['n_frames']} | {s['kappa']:.3f} | "
            f"{s['rear_p']:.2f} | {s['rear_r']:.2f} | {s['rear_f1']:.2f} | {s['rear_n']} | "
            f"{s['groom_p']:.2f} | {s['groom_r']:.2f} | {s['groom_f1']:.2f} | {s['groom_n']} |"
        )
    add("")

    add("### Confusion matrices")
    add("")
    add("Rows = DLC reference, columns = blob detector. Diagonal = agreement.")
    add("")
    for s in subjects:
        add(f"**{s['subject']}**")
        add("")
        add("|  | other | rearing | grooming |")
        add("|---|---|---|---|")
        for i, name in enumerate(CLASSES):
            row = s["cm"][i]
            add(f"| **{name}** | {row[0]} | {row[1]} | {row[2]} |")
        add("")
        rel = (FIGS / f"blob_vs_dlc_{s['subject']}_timeline.png").relative_to(DOCS)
        add(f"![{s['subject']} timeline](figures/{rel.name})")
        add("")

    _write_reproduce(add, cohorts)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(L), encoding="utf-8")


def _write_reproduce(add, cohorts: list[str]) -> None:
    add("## Reproduce")
    add("")
    add("```bash")
    add("pip install scipy numpy pandas matplotlib scikit-learn")
    add(f"python analysis/blob_vs_dlc.py --cohorts {' '.join(cohorts)}")
    add("```")
    add("")
    add("Detection thresholds live at the top of `analysis/blob_vs_dlc.py`")
    add("(`REAR_AREA_FRAC`, `REAR_ASPECT_MAX`, `GROOM_VEL_MAX`, `GROOM_AREA_CV_MAX`).")
    add("Adjust and re-run; this report regenerates from scratch.")
    add("")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", nargs="+", default=["MA1", "MA3"],
                    help="Cohorts to compare (default: MA1 MA3)")
    ap.add_argument("--out-md", default=str(DOCS / "BLOB_VS_DLC_REPORT.md"),
                    help="Output report path")
    args = ap.parse_args()

    pairs = find_pairs(args.cohorts)
    print(f"Found {len(pairs)} subject pair(s):")
    for s, b, d in pairs:
        print(f"  {s}:  {b.relative_to(ROOT)}  <->  {d.relative_to(ROOT)}")

    subjects: list[dict] = []
    for subject, mat_path, dlc_csv in pairs:
        print(f"\nProcessing {subject}...")
        blob_data = load_blob_mat(mat_path)
        feat = blob_features(blob_data)
        blob_lab = detect_behaviors(feat)

        dlc_lab = pd.read_csv(dlc_csv)["behaviour"].values

        blob_a, dlc_a = align(blob_lab, dlc_lab, blob_data["t"])
        m = per_subject_metrics(blob_a, dlc_a)
        m["subject"] = subject
        subjects.append(m)

        out_png = FIGS / f"blob_vs_dlc_{subject}_timeline.png"
        plot_overlay(subject, blob_a, dlc_a, out_png)

        print(f"  n={m['n_frames']}  kappa={m['kappa']:.3f}  "
              f"rear F1={m['rear_f1']:.2f}  groom F1={m['groom_f1']:.2f}")

    md_path = pathlib.Path(args.out_md)
    write_report(subjects, args.cohorts, md_path)
    print(f"\nReport written to {md_path}")


if __name__ == "__main__":
    main()
