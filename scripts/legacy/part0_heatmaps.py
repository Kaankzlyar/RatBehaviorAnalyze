import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Koordinatlar
coords_file = "C:/RatWork/data/part0_coordinates.csv"
df = pd.read_csv(coords_file)

# Çıktı klasörü
HEATMAP_DIR = "C:/RatWork/data/part0_heatmaps"
os.makedirs(HEATMAP_DIR, exist_ok=True)

# Arena parametreleri
ARENA_WIDTH = 1280
ARENA_HEIGHT = 720

print("Heatmaplar olusturuluyor...")

for video in sorted(df['video'].unique()):
    video_data = df[df['video'] == video]

    # Body center koordinatları
    x_coords = video_data['body_center_x'].dropna().values
    y_coords = video_data['body_center_y'].dropna().values

    if len(x_coords) < 100:
        print(f"[skip] {video}: Az veri")
        continue

    # 2D Histogram (Heatmap)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Piksel tabanlı grid
    h, xedges, yedges = np.histogram2d(
        x_coords, y_coords,
        bins=[int(ARENA_WIDTH/20), int(ARENA_HEIGHT/20)],  # 20px grid
        range=[[0, ARENA_WIDTH], [0, ARENA_HEIGHT]]
    )

    # Y eksenini çevir (görselde y artan yukarı gitmeli)
    im = ax.imshow(h.T, origin='lower', cmap='hot', extent=[0, ARENA_WIDTH, 0, ARENA_HEIGHT])

    ax.set_xlim(0, ARENA_WIDTH)
    ax.set_ylim(0, ARENA_HEIGHT)
    ax.set_xlabel('X (piksel)')
    ax.set_ylabel('Y (piksel)')
    ax.set_title(f'{video}\nOccupancy Heatmap')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frame Count (Kare Sayısı)')

    # Kaydet
    out_path = os.path.join(HEATMAP_DIR, f"{video}_heatmap.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[OK] {video}")

print(f"\nTamamlandi! Heatmaplar: {HEATMAP_DIR}")
