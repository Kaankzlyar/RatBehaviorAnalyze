# Strategy A pipeline — 24-sample anxiety_level + group classifier

A parallel pipeline that takes MA2/4/6/8 (HSV-blob `_res.mat` only) and the
existing MA1/3/5/7 DLC subjects to a 24-sample classifier for `anxiety_level`
and `group`.  **Does not modify any code under `analysis/`, `src/`, `data/`,
`models/`, or `reports/` outside this folder** — all generated outputs live
under `data/blob_pipeline/`, `models/blob_pipeline/`, and
`reports/blob_pipeline/`.

## Why this pipeline exists

`docs/BLOB_VS_DLC_REPORT.md` quantified that the HSV-blob detector cannot
reliably reproduce DLC's rearing/grooming labels (F1 = 0.13 / 0.25, n = 12).
We therefore drop the `rearing_profile` and `grooming_profile` classifier
targets and double the sample count for the centroid-derivable targets
(`anxiety_level`, `group`) by adding MA2/4/6/8 from blob data.

## Steps (run in order)

| # | Script | Inputs | Outputs |
|---|---|---|---|
| 01 | `01_convert_mat_to_csv.py` | `data/DLCfiltered/Kare/{folder}/MA*-*_res.mat`, `data/DLCfiltered/{folder}/OpenField...csv` (for sanity check) | `data/blob_pipeline/_arena_check.csv`, `data/blob_pipeline/csv_converted/{folder}/OpenFieldMA{N}_{R}/OpenFieldMA{N}_{R}.csv` (only with `--apply`) |
| 02 | _pending_ — `02_compute_metrics.py` | the converted CSVs + existing 12 DLC CSVs | `data/blob_pipeline/oft_metrics_24.csv` |
| 03 | _pending_ — `03_build_features.py` | `oft_metrics_24.csv` | `data/blob_pipeline/features/{features_raw_24,features_normalized_24,labels_24}.csv` |
| 04 | _pending_ — `04_train_classifiers.py` | features + labels | `models/blob_pipeline/classifier/*.pkl`, `reports/blob_pipeline/{model_comparison_24,loocv_predictions_24}.csv`, `reports/blob_pipeline/figures/*.png` |

Steps 02–04 will be added after step 01's verdict is reviewed.

## Step 01 — coordinate sanity check + conversion

The MATLAB blob pipeline (`sonuc_kutu_final_v1.m`) and DLC pose pipeline
process the same source videos but may use slightly different image
coordinates.  Step 01 must verify alignment before fixed arena bounds
can be reused for MA2/4/6/8.

```bash
pip install scipy numpy pandas
python analysis/blob_pipeline/01_convert_mat_to_csv.py
```

Outputs `data/blob_pipeline/_arena_check.csv` plus a stdout report with
per-subject xc/yc percentile ranges, and a verdict for the paired
MA1/3/5/7 subjects:

| Verdict | Worst \|offset\| | Action |
|---|---|---|
| GREEN  | < 20 px            | fixed arena `396 776 153 530` is safe — proceed to `--apply` |
| YELLOW | 20 ≤ \|offset\| < 50 px | recommend offset transform on MA2/4/6/8 before conversion |
| RED    | ≥ 50 px            | fixed arena will not work — step 02 must use per-subject auto-arena |

After reviewing the verdict, run with `--apply` to emit the DLC-style
CSVs under `data/blob_pipeline/csv_converted/`:

```bash
python analysis/blob_pipeline/01_convert_mat_to_csv.py --apply
```

If the verdict is YELLOW or RED, the script refuses to convert; pass
`--force` to override (not recommended without addressing the offset).

## Conventions

- All converted CSVs live under `data/blob_pipeline/csv_converted/`,
  isolated from `data/DLCfiltered/` so they cannot accidentally be
  picked up by the existing 12-subject DLC batch tools.
- Converted CSVs use the same DLC 3-row header layout (`scorer`,
  `bodyparts`, `coords`).  `body_center` is populated from `xc/yc`
  with `likelihood = 1.0`; the other 9 bodyparts have empty x/y
  (parsed as NaN) and `likelihood = 0`, so `oft_metrics.py`'s
  likelihood filter drops them automatically.
- Cohort → folder mapping mirrors the existing `Kare/` layout:
  control (MA1+MA2), ASP (MA3+MA4), Greyfurt (MA5+MA6),
  ASP ve Greyfurt (MA7+MA8).
