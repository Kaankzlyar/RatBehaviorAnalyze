import deeplabcut
import glob
import os

CONFIG  = "C:/RatWork/models/dlc_model/rat_behavior-kaank-2026-04-01/config.yaml"
OUT_DIR = "C:/RatWork/data/dlc_output/part0_labeled"

os.makedirs(OUT_DIR, exist_ok=True)

videos = sorted(glob.glob("C:/RatWork/data/raw_videos/Part0/*.avi"))
print(f"Toplam {len(videos)} video bulundu:\n")
for v in videos:
    print(f"  {os.path.basename(v)}")

print("\n[1/2] Koordinatlar hesaplanıyor...")
deeplabcut.analyze_videos(CONFIG, videos, save_as_csv=True, destfolder=OUT_DIR)

print("\n[2/2] Etiketli videolar oluşturuluyor...")
deeplabcut.create_labeled_video(CONFIG, videos, destfolder=OUT_DIR)

print(f"\nTamamlandı! Çıktılar: {OUT_DIR}")
