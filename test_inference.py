import deeplabcut
import glob

config  = "C:/RatWork/models/dlc_model/rat_behavior-kaank-2026-04-01/config.yaml"
videos  = sorted(glob.glob("C:/RatWork/data/raw_videos/Part0/*.avi"))
out_dir = "C:/RatWork/data/dlc_output/part0"

print(f"{len(videos)} video bulundu.")
deeplabcut.analyze_videos(config, videos, save_as_csv=True, destfolder=out_dir)
deeplabcut.create_labeled_video(config, videos, destfolder=out_dir)
