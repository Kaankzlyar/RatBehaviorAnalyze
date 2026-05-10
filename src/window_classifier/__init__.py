"""Window-level behavior classifier pipeline.

Stages:
    features      DLC pose CSV -> per-window feature parquet
    label_join    rule-based frame labels -> per-window label
    train         GroupKFold + LOGOCV training, model + reports
    infer         new DLC pose CSV -> per-frame predictions + bouts
"""
