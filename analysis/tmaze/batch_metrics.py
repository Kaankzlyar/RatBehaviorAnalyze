"""
batch_metrics.py
----------------
12 sıçanın ayrı *_plus_maze_metrics.csv dosyalarını tarayıp
data/plus_maze_metrics_all.csv'ye birleştirir.

Kullanım:
    python analysis/tmaze/batch_metrics.py

Çıktı:
    data/plus_maze_metrics_all.csv
"""

import pathlib
import pandas as pd

ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
DATA      = ROOT / "data" / "DLCfiltered"
OUT_PATH  = ROOT / "data" / "plus_maze_metrics_all.csv"

COHORT_MAP = {
    "control":         "Control",
    "ASP":             "Aspartame",
    "Greyfurt":        "Grapefruit",
    "ASP ve Greyfurt": "ASP+Greyfurt",
}


def collect() -> pd.DataFrame:
    frames = []
    for cohort_dir, cohort_label in COHORT_MAP.items():
        folder = DATA / cohort_dir
        if not folder.exists():
            print(f"[skip] {folder} bulunamadı")
            continue
        for subject_dir in sorted(folder.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith("PlusMaze"):
                continue
            csv_files = list(subject_dir.glob("*_plus_maze_metrics.csv"))
            if not csv_files:
                print(f"[skip] {subject_dir.name} — metrik CSV yok")
                continue
            df = pd.read_csv(csv_files[0])
            # cohort etiketi override — tutarlı olsun
            df["cohort"] = cohort_label
            frames.append(df)
            print(f"[ok]   {subject_dir.name}  ({cohort_label})")

    if not frames:
        raise RuntimeError("Hiç metrik dosyası bulunamadı.")
    return pd.concat(frames, ignore_index=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """EPM analizi için türetilmiş sütunlar ekle."""
    # Açık kol (yatay: left + right), kapalı kol (dikey: top + bottom)
    df["pct_open_arm"]   = df["pct_time_left"] + df["pct_time_right"]
    df["pct_closed_arm"] = df["pct_time_top"]  + df["pct_time_bottom"]

    # Açık kol girişi yüzdesi
    total = df["total_entries"].replace(0, float("nan"))
    df["pct_open_arm_entries"]   = (df["left_entries"] + df["right_entries"]) / total * 100
    df["pct_closed_arm_entries"] = (df["top_entries"]  + df["bottom_entries"]) / total * 100

    # Anksiyete indeksi: yüksek = düşük kaygı
    # (% açık kol süre + % açık kol giriş) / 2
    df["anxiety_index_epm"] = (df["pct_open_arm"] + df["pct_open_arm_entries"].fillna(0)) / 2

    return df


def main():
    print("Plus Maze metrik dosyaları taranıyor...\n")
    df = collect()
    df = add_derived(df)

    # Sütun sıralaması — meta önce, türetilmiş sonra
    meta_cols    = ["subject_id", "cohort_id", "cohort", "session",
                    "n_frames", "n_valid_frames", "session_duration_s"]
    derived_cols = ["pct_open_arm", "pct_closed_arm",
                    "pct_open_arm_entries", "pct_closed_arm_entries",
                    "anxiety_index_epm"]
    other_cols   = [c for c in df.columns
                    if c not in meta_cols + derived_cols]
    df = df[meta_cols + other_cols + derived_cols]

    df.to_csv(OUT_PATH, index=False)
    print(f"\n[done] {len(df)} sican -> {OUT_PATH}")
    print(f"       Sütun sayısı: {len(df.columns)}")
    print("\nKohort dağılımı:")
    print(df.groupby("cohort")[["pct_open_arm", "anxiety_index_epm",
                                 "total_entries"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
