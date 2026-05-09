"""
behavior_analysis.py
--------------------
Cross-cohort OFT behavioral analysis for the bipolar/anxiety rat model.

Reads the per-subject *_behavior_bouts.csv and *_oft_metrics.csv files and
produces:
  1. data/behavior_summary.csv    — per-subject metric table
  2. data/behavior_group_stats.csv — cohort means ± SD
  3. data/behavior_comparison.png  — multi-panel thesis figure

Metrics
-------
  Line Crossing : total_distance_px (locomotor activity proxy)
  Thigmotaxis   : pct_time_periphery (wall-hugging, anxiety index)
  Rearing       : count, total_s, mean_bout_s (vertical exploration)
  Grooming      : count, total_s, mean_bout_s, fragmentation_index
                  (fragmentation = n_bouts / total_s;  higher = more disrupted)

Biological framework (see thesis literature)
--------------------------------------------
  Bipolar/manic phase  : high locomotion, high rearing, fragmented/increased grooming,
                         low thigmotaxis (risk-taking, center-seeking)
  Depressive/anxiety   : low locomotion, low rearing, fragmented grooming,
                         high thigmotaxis (wall-hugging, center avoidance)
  Grapefruit treatment : metrics converge toward control (antioxidant restoration
                         of dopamine/serotonin balance disrupted by aspartame)

Groups (confirmed by folder structure)
---------------------------------------
  MA1 = Control
  MA3 = Aspartame           (bipolar/anxiety model)
  MA5 = Grapefruit only     (single-treatment arm)
  MA7 = Aspartame+Grapefruit(combined arm)

Usage
-----
  python analysis/behavior_analysis.py
  python analysis/behavior_analysis.py --dlc-dir data/DLCfiltered --out-dir data
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── cohort metadata ────────────────────────────────────────────────────────────
COHORT_MAP = {
    "MA1": "Control",
    "MA2": "Control",
    "MA3": "Aspartame",
    "MA4": "Aspartame",
    "MA5": "Grapefruit",
    "MA6": "Grapefruit",
    "MA7": "Aspartame+\nGrapefruit",
    "MA8": "Aspartame+\nGrapefruit",
}

GROUP_COLORS = {
    "Control":             "#4C72B0",   # blue
    "Aspartame":           "#DD8452",   # orange-red
    "Grapefruit":          "#55A868",   # green
    "Aspartame+\nGrapefruit": "#C44E52", # red
}

GROUP_ORDER = ["Control", "Aspartame", "Grapefruit", "Aspartame+\nGrapefruit"]

# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_DLC_DIR = "data/DLCfiltered"
DEFAULT_OUT_DIR = "data"


# ── data collection ────────────────────────────────────────────────────────────

def find_subject_files(dlc_dir: str) -> list[dict]:
    """Walk dlc_dir and collect paths to bouts + oft_metrics CSVs per subject."""
    subjects = []
    for root, dirs, files in os.walk(dlc_dir):
        for f in files:
            if f.endswith("_behavior_bouts.csv"):
                base = f.replace("_behavior_bouts.csv", "")
                m = re.search(r"(MA\d+)_(\d+)", base)
                if not m:
                    continue
                cohort = m.group(1)
                run    = m.group(2)
                group  = COHORT_MAP.get(cohort, cohort)
                bouts_path = os.path.join(root, f)
                metrics_path = os.path.join(root, f"{base}_oft_metrics.csv")
                subjects.append({
                    "subject_id": f"{cohort}_{run}",
                    "cohort": cohort,
                    "group": group,
                    "bouts_path": bouts_path,
                    "metrics_path": metrics_path if os.path.isfile(metrics_path) else None,
                })
    return sorted(subjects, key=lambda x: x["subject_id"])


def compute_behavior_metrics(sub: dict) -> dict:
    """Compute all behavioral metrics for one subject."""
    row = {
        "subject_id":  sub["subject_id"],
        "cohort":      sub["cohort"],
        "group":       sub["group"],
    }

    # ── session duration (from oft_metrics if available, else bouts max) ───
    session_s = None
    if sub["metrics_path"]:
        mdf = pd.read_csv(sub["metrics_path"])
        if not mdf.empty:
            session_s            = float(mdf["session_duration_s"].iloc[0])
            row["total_distance_px"] = float(mdf["total_distance_px"].iloc[0])
            row["mean_speed_px_s"]   = float(mdf["mean_speed_px_s"].iloc[0])
            row["pct_time_center"]   = float(mdf["pct_time_center"].iloc[0])
            row["pct_time_periphery"]= float(mdf["pct_time_periphery"].iloc[0])
            row["center_entries"]    = int(mdf["center_zone_entries"].iloc[0])
            row["spatial_entropy"]   = float(mdf["spatial_entropy_norm"].iloc[0])

    bouts = pd.read_csv(sub["bouts_path"])

    if session_s is None:
        session_s = float(bouts["end_s"].max()) if not bouts.empty else 1.0

    row["session_s"] = session_s

    for beh, prefix in [("rearing", "rear"), ("grooming", "groom")]:
        sub_b = bouts[bouts["behaviour"] == beh].copy()
        count     = len(sub_b)
        total_s   = float(sub_b["duration_s"].sum()) if count else 0.0
        pct       = total_s / session_s * 100 if session_s else 0.0
        mean_s    = float(sub_b["duration_s"].mean()) if count else 0.0
        max_s     = float(sub_b["duration_s"].max())  if count else 0.0
        # fragmentation index: more bouts per unit time = more disrupted
        frag      = count / total_s if total_s > 0 else 0.0

        row[f"{prefix}_count"]     = count
        row[f"{prefix}_total_s"]   = round(total_s, 2)
        row[f"{prefix}_pct"]       = round(pct, 2)
        row[f"{prefix}_mean_s"]    = round(mean_s, 3)
        row[f"{prefix}_max_s"]     = round(max_s, 2)
        row[f"{prefix}_frag_idx"]  = round(frag, 3)  # bouts / s

    return row


# ── statistics helpers ─────────────────────────────────────────────────────────

def group_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return mean ± SD per group (ordered by GROUP_ORDER)."""
    stats = df.groupby("group")[metric].agg(["mean", "std", "count"]).reindex(GROUP_ORDER)
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    return stats


# ── plotting ───────────────────────────────────────────────────────────────────

def _bar_with_points(ax, df: pd.DataFrame, metric: str, ylabel: str,
                     title: str, fmt_int: bool = False) -> None:
    """
    Bar chart (group mean) with individual data points and SD error bars.
    Drops groups with no data for the metric.
    """
    stats = group_stats(df, metric)
    groups = [g for g in GROUP_ORDER if g in stats.index and not np.isnan(stats.loc[g, "mean"])]
    if not groups:
        ax.set_visible(False)
        return

    x = np.arange(len(groups))
    colors = [GROUP_COLORS[g] for g in groups]
    means  = [stats.loc[g, "mean"] for g in groups]
    sds    = [stats.loc[g, "std"]  for g in groups]

    bars = ax.bar(x, means, color=colors, alpha=0.75, width=0.55,
                  edgecolor="white", linewidth=0.8, zorder=2)

    # SD error bars
    ax.errorbar(x, means, yerr=sds, fmt="none", color="#333333",
                capsize=5, capthick=1.2, linewidth=1.2, zorder=3)

    # individual data points
    for i, g in enumerate(groups):
        vals = df[df["group"] == g][metric].dropna().values
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(i + jitter, vals, color="white", edgecolors=colors[i],
                   s=45, linewidths=1.5, zorder=4)

    ax.set_xticks(x)
    short = [g.replace("Aspartame+\nGrapefruit", "Asp+\nGrap") for g in groups]
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(bottom=0)

    # value labels on bars
    for bar, mean in zip(bars, means):
        label = f"{mean:.0f}" if fmt_int else f"{mean:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01,
                label, ha="center", va="bottom", fontsize=7, color="#333333")


def plot_comparison(df: pd.DataFrame, out_path: str) -> None:
    """Produce the 3×3 multi-panel OFT comparison figure."""
    np.random.seed(42)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle(
        "Open Field Test — Behavioral Profile Comparison\n"
        "Control  vs  Aspartame  vs  Grapefruit  vs  Aspartame+Grapefruit",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.patch.set_facecolor("#FAFAFA")
    for ax in axes.flat:
        ax.set_facecolor("#F5F5F5")

    panels = [
        # (metric,              ylabel,                    title,                         fmt_int)
        ("total_distance_px",   "Distance (px)",           "Line Crossing\n(Total Distance)", True),
        ("pct_time_periphery",  "% Time",                  "Thigmotaxis\n(Periphery Time)", False),
        ("center_entries",      "Count",                   "Center Zone\nEntries",         True),
        ("rear_count",          "Count",                   "Rearing\nBout Count",          True),
        ("rear_total_s",        "Duration (s)",            "Rearing\nTotal Duration",      False),
        ("rear_mean_s",         "Mean Duration (s)",       "Rearing\nMean Bout Duration",  False),
        ("groom_count",         "Count",                   "Grooming\nBout Count",         True),
        ("groom_total_s",       "Duration (s)",            "Grooming\nTotal Duration",     False),
        ("groom_mean_s",        "Mean Duration (s)",       "Grooming\nMean Bout Duration\n(low = fragmented)", False),
    ]

    for ax, (metric, ylabel, title, fmt_int) in zip(axes.flat, panels):
        if metric in df.columns:
            _bar_with_points(ax, df, metric, ylabel, title, fmt_int)
        else:
            ax.set_visible(False)

    # legend
    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g.replace("\n", " "), alpha=0.8)
        for g in GROUP_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


def plot_grooming_fragmentation(df: pd.DataFrame, out_path: str) -> None:
    """
    Dedicated grooming fragmentation figure.
    Shows count vs total_s scatter + mean_bout_s bar per group.
    """
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Grooming Pattern Analysis — Fragmentation",
                 fontsize=11, fontweight="bold")
    fig.patch.set_facecolor("#FAFAFA")

    # left: scatter count vs total duration
    ax = axes[0]
    ax.set_facecolor("#F5F5F5")
    for g in GROUP_ORDER:
        sub = df[df["group"] == g]
        ax.scatter(sub["groom_total_s"], sub["groom_count"],
                   color=GROUP_COLORS[g], s=100,
                   edgecolors="white", linewidths=1.2,
                   label=g.replace("\n", " "), zorder=3)
        # label each point
        for _, r in sub.iterrows():
            ax.annotate(r["subject_id"],
                        (r["groom_total_s"], r["groom_count"]),
                        fontsize=6.5, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points", color="#444444")
    ax.set_xlabel("Total Grooming Time (s)", fontsize=9)
    ax.set_ylabel("Number of Bouts", fontsize=9)
    ax.set_title("Count vs Duration\n(upper-left = many short/fragmented bouts)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.8)

    # right: mean bout duration (fragmentation inverse)
    _bar_with_points(axes[1], df, "groom_mean_s",
                     "Mean Bout Duration (s)",
                     "Mean Grooming Bout Duration\n(shorter = more fragmented / disrupted)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── console report ─────────────────────────────────────────────────────────────

METRICS_DEF = [
    ("total_distance_px",  "Distance (px)",                "{:.0f}",  True),
    ("pct_time_periphery", "Periphery %",                  "{:.1f}",  False),
    ("center_entries",     "Center entries",               "{:.0f}",  True),
    ("rear_count",         "Rearing bouts",                "{:.0f}",  True),
    ("rear_total_s",       "Rearing total (s)",            "{:.1f}",  False),
    ("rear_mean_s",        "Rearing mean bout (s)",        "{:.2f}",  False),
    ("groom_count",        "Grooming bouts",               "{:.0f}",  True),
    ("groom_total_s",      "Grooming total (s)",           "{:.1f}",  False),
    ("groom_mean_s",       "Grooming mean bout (s)",       "{:.2f}",  False),
    ("groom_frag_idx",     "Grooming frag. (bouts/s)",     "{:.2f}",  False),
    ("spatial_entropy",    "Spatial entropy (0-1)",        "{:.3f}",  False),
]

# column headers use the short group labels (no newline)
_COL_LABELS = [g.replace("\n", " ") for g in GROUP_ORDER]


def _fmt_cell(mean: float, sd: float, fmt: str, is_int: bool) -> str:
    """Format 'mean +/- SD' for one table cell."""
    if np.isnan(mean):
        return "N/A"
    m_s = fmt.format(mean)
    s_s = fmt.format(sd) if not np.isnan(sd) else "0"
    return f"{m_s} +/- {s_s}"


def _table_row(label: str, cells: list[str], col_widths: list[int],
               label_w: int) -> str:
    row = "| " + label.ljust(label_w) + " |"
    for cell, w in zip(cells, col_widths):
        row += " " + cell.center(w) + " |"
    return row


def _divider(col_widths: list[int], label_w: int,
             left="+", mid="+", right="+", fill="-") -> str:
    parts = [fill * (label_w + 2)]
    for w in col_widths:
        parts.append(fill * (w + 2))
    return left + mid.join(parts) + right


def print_report(df: pd.DataFrame) -> None:
    label_w = max(len(m[1]) for m in METRICS_DEF)
    col_labels = _COL_LABELS

    # build cell strings first to determine column widths
    table_data: list[tuple[str, list[str]]] = []
    for col, label, fmt, is_int in METRICS_DEF:
        if col not in df.columns:
            continue
        stats = group_stats(df, col)
        cells = []
        for g in GROUP_ORDER:
            if g in stats.index:
                cells.append(_fmt_cell(stats.loc[g, "mean"], stats.loc[g, "std"],
                                       fmt, is_int))
            else:
                cells.append("N/A")
        table_data.append((label, cells))

    col_widths = [max(len(h), max(len(r[1][i]) for r in table_data))
                  for i, h in enumerate(col_labels)]

    top    = _divider(col_widths, label_w)
    sep    = _divider(col_widths, label_w)
    bottom = _divider(col_widths, label_w)

    header_row = _table_row("Metric", col_labels, col_widths, label_w)

    print(f"\n  OFT Behavioral Profile -- Cohort Means (mean +/- SD)")
    print(top)
    print(header_row)
    print(sep)
    for label, cells in table_data:
        print(_table_row(label, cells, col_widths, label_w))
        print(sep)
    # replace last sep with bottom (same in ASCII)
    print()

    # biological interpretation
    print("Interpretation (bipolar/anxiety framework):")
    print("-" * 60)
    asp  = df[df["group"] == "Aspartame"]
    ctrl = df[df["group"] == "Control"]
    if not asp.empty and not ctrl.empty:
        d_diff    = asp["total_distance_px"].mean() - ctrl["total_distance_px"].mean()
        r_diff    = asp["rear_count"].mean()         - ctrl["rear_count"].mean()
        thig_diff = asp["pct_time_periphery"].mean() - ctrl["pct_time_periphery"].mean()
        gm_diff   = asp["groom_mean_s"].mean()       - ctrl["groom_mean_s"].mean()
        print(f"  Aspartame vs Control:")
        print(f"    Distance      : {'+' if d_diff>=0 else ''}{d_diff:.0f} px  "
              f"({'hyperactive' if d_diff > 500 else 'hypolocomotive' if d_diff < -500 else 'similar to control'})")
        print(f"    Rearing       : {'+' if r_diff>=0 else ''}{r_diff:.1f} bouts  "
              f"({'increased' if r_diff > 1 else 'decreased' if r_diff < -1 else 'similar to control'})")
        print(f"    Thigmotaxis   : {'+' if thig_diff>=0 else ''}{thig_diff:.1f}%  "
              f"({'more anxious / thigmotactic' if thig_diff > 5 else 'less anxious' if thig_diff < -5 else 'similar'})")
        print(f"    Groom bout len: {'+' if gm_diff>=0 else ''}{gm_diff:.2f} s  "
              f"({'longer bouts' if gm_diff > 0.2 else 'shorter/fragmented' if gm_diff < -0.2 else 'similar'})")
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-cohort OFT behavioral analysis"
    )
    parser.add_argument("--dlc-dir", default=DEFAULT_DLC_DIR, dest="dlc_dir",
                        help=f"DLC data directory (default: {DEFAULT_DLC_DIR})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, dest="out_dir",
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning {args.dlc_dir} for subject files...")
    subjects = find_subject_files(args.dlc_dir)
    print(f"  Found {len(subjects)} subjects.\n")

    rows = []
    for sub in subjects:
        row = compute_behavior_metrics(sub)
        rows.append(row)
        print(f"  {sub['subject_id']} ({sub['group']})  "
              f"rear={row['rear_count']} ({row['rear_total_s']:.1f}s)  "
              f"groom={row['groom_count']} ({row['groom_total_s']:.1f}s)")

    df = pd.DataFrame(rows)

    # ── save summary CSV ───────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "behavior_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    # ── group stats CSV ────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    group_means = df.groupby("group")[numeric_cols].mean().round(3)
    group_sds   = df.groupby("group")[numeric_cols].std().round(3)
    group_stats_df = group_means.add_suffix("_mean").join(group_sds.add_suffix("_sd"))
    stats_path = os.path.join(args.out_dir, "behavior_group_stats.csv")
    group_stats_df.to_csv(stats_path)
    print(f"  Saved: {stats_path}")

    # ── console report ─────────────────────────────────────────────────────
    print_report(df)

    # ── figures ────────────────────────────────────────────────────────────
    print("Generating figures...")
    fig_path  = os.path.join(args.out_dir, "behavior_comparison.png")
    frag_path = os.path.join(args.out_dir, "grooming_fragmentation.png")
    plot_comparison(df, fig_path)
    plot_grooming_fragmentation(df, frag_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
