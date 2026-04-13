"""
dlc_train.py
------------
Step 3 & 4: Create training dataset, train the model, and evaluate it.
Run after labeling is complete in dlc_setup.py.

GPU: RTX 3060 6GB — uses PyTorch backend with CUDA.
     batch_size=8 is safe for 6GB VRAM with ResNet-50.
"""

import deeplabcut

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Paste your config path from dlc_setup.py output here
CONFIG_PATH = "C:/RatWork/models/dlc_model/rat_behavior-kaank-2026-04-01/config.yaml"

GPU_ID      = 0       # GPU index — 0 for your RTX 3060
SHUFFLE     = 1       # Training shuffle index
MAX_ITERS   = 50000   # Start with 50k; increase if test RMSE > 8px
EPOCHS      = 300     # 300 epochs optimal for ~240 labeled frames

# ─── CREATE TRAINING DATASET ──────────────────────────────────────────────────

def create_dataset(config_path: str) -> None:
    deeplabcut.create_training_dataset(
        config_path,
        num_shuffles=1,
        augmenter_type="imgaug",   # imgaug works well with PyTorch backend
    )
    print("Training dataset created.")

# ─── PATCH POSE CONFIG FOR RTX 3060 ──────────────────────────────────────────

def patch_pose_config(config_path: str, shuffle: int = 1) -> None:
    """
    Patch pytorch_config.yaml for DLC 3.0 (PyTorch backend).
    Sets epochs and batch_size for RTX 3060 6GB VRAM.
    """
    import os, glob, yaml

    project_dir = os.path.dirname(config_path)

    # DLC 3.0 PyTorch config
    pattern = os.path.join(
        project_dir, "dlc-models-pytorch", "**", "train", "pytorch_config.yaml"
    )
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        print("pytorch_config.yaml not found — skipping patch.")
        return

    pytorch_cfg_path = matches[0]
    with open(pytorch_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["train_settings"]["epochs"]     = EPOCHS
    cfg["train_settings"]["batch_size"] = 8

    with open(pytorch_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"pytorch_config.yaml patched: epochs={EPOCHS}, batch_size=8\n  → {pytorch_cfg_path}")

# ─── TRAIN ────────────────────────────────────────────────────────────────────

def train(config_path: str, shuffle: int = SHUFFLE, max_iters: int = MAX_ITERS) -> None:
    print(f"\nStarting training on GPU {GPU_ID} for {max_iters} iterations...")
    print("Monitor loss in terminal — training saves checkpoints every 5000 iters.\n")

    deeplabcut.train_network(
        config_path,
        shuffle=shuffle,
        trainingsetindex=0,
        gputouse=GPU_ID,
        max_snapshots_to_keep=5,
        autotune=False,
        displayiters=500,
        saveiters=5000,
        maxiters=max_iters,
    )
    print("\nTraining complete.")

# ─── EVALUATE ─────────────────────────────────────────────────────────────────

def evaluate(config_path: str, shuffle: int = SHUFFLE) -> None:
    deeplabcut.evaluate_network(
        config_path,
        plotting=True,        # saves prediction vs ground truth plots
    )
    # Target: Train RMSE < 5px, Test RMSE < 8px
    print("\nEvaluation plots saved to evaluation-results/ in your project folder.")
    print("Target: Train RMSE < 5px | Test RMSE < 8px")

# ─── REFINE (if evaluation is poor) ──────────────────────────────────────────

def extract_outliers_and_refine(config_path: str, videos: list[str]) -> None:
    """
    Use when RMSE is high on specific videos.
    Extracts low-confidence frames → re-label → merge → retrain.
    """
    # Extract uncertain frames from problem videos
    deeplabcut.extract_outlier_frames(
        config_path,
        videos,
        outlieralgorithm="uncertain",
        epsilon=20,
        automatic=True,
    )

    # Re-label the extracted frames
    print("Launching refine GUI — label the new outlier frames.")
    deeplabcut.refine_labels(config_path)

    # Merge back into main dataset
    deeplabcut.merge_datasets(config_path)
    print("Dataset merged. Re-run train() with higher max_iters.")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    create_dataset(CONFIG_PATH)
    patch_pose_config(CONFIG_PATH, shuffle=SHUFFLE)
    train(CONFIG_PATH, shuffle=SHUFFLE, max_iters=MAX_ITERS)
    evaluate(CONFIG_PATH, shuffle=SHUFFLE)

    print(f"\nNext step: run dlc_inference.py with CONFIG_PATH = '{CONFIG_PATH}'")
