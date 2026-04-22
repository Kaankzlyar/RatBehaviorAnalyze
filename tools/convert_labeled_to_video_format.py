"""
convert_labeled_to_video_format.py
------------------------------------
Converts DLC labeled-data CSVs (x, y only, 3-column index) to the
standard DLC video-analysis format (x, y, likelihood=1.0, integer index).

Usage:
    python convert_labeled_to_video_format.py
    python convert_labeled_to_video_format.py --dry-run
    python convert_labeled_to_video_format.py --csv path/to/file.csv
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np


def is_labeled_format(csv_path: str) -> bool:
    """Return True if the CSV uses the labeled-data (no likelihood) format."""
    with open(csv_path, "r") as f:
        lines = [f.readline() for _ in range(3)]
    # coords row has only x/y — no 'likelihood' token
    return "likelihood" not in lines[2]


def convert(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=[0, 1, 2])
    scorer_name = df.columns.get_level_values(0).unique()[0]
    df.columns = df.columns.droplevel(0)
    bodyparts = df.columns.get_level_values(0).unique().tolist()

    # Build new multi-level columns: (scorer, bodypart, coord) with likelihood added
    new_cols = pd.MultiIndex.from_tuples(
        [(scorer_name, bp, coord)
         for bp in bodyparts
         for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )

    rows = []
    for i in range(len(df)):
        row = []
        for bp in bodyparts:
            row.append(float(df[bp]["x"].iloc[i]))
            row.append(float(df[bp]["y"].iloc[i]))
            row.append(1.0)  # all labeled points are fully valid
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=new_cols)
    out_df.index = range(len(out_df))
    out_df.index.name = None

    out_df.to_csv(out_path)
    print(f"  {len(out_df):4d} frames  ->  {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert labeled-data CSVs to video-analysis format")
    parser.add_argument("--csv", help="Single CSV to convert (default: all new-format CSVs in data/DLCfiltered)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without writing")
    parser.add_argument("--data-dir", default="../data/DLCfiltered",
                        help="Root folder to scan (default: ../data/DLCfiltered)")
    args = parser.parse_args()

    if args.csv:
        targets = [args.csv]
    else:
        root = os.path.abspath(args.data_dir)
        targets = []
        for dirpath, _, files in os.walk(root):
            for fname in files:
                if fname.endswith(".csv") and not any(
                    tag in fname for tag in ("_behavior", "_bouts", "_frames")
                ):
                    targets.append(os.path.join(dirpath, fname))
        targets.sort()

    converted = skipped = 0
    for csv_path in targets:
        if not is_labeled_format(csv_path):
            print(f"  SKIP (already video format): {csv_path}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  WOULD CONVERT: {csv_path}")
            converted += 1
            continue

        print(f"Converting: {csv_path}")
        convert(csv_path, csv_path)  # overwrite in place
        converted += 1

    print(f"\nDone. Converted: {converted}  Skipped: {skipped}")


if __name__ == "__main__":
    main()
