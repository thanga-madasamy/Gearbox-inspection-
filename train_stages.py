import os
import shutil
from ultralytics import YOLO

# Project Base Directory
BASE_DIR = r"C:\Users\z0050910\dev\washer_bolt_project"

# Stage setup definitions
stages = [
    {
        "name": "stage_01",
        "yaml": os.path.join(BASE_DIR, "dataset_stage01.yaml"),
        "dataset_dir": os.path.join(BASE_DIR, "dataset_stage01"),
        "target_model": os.path.join(BASE_DIR, "models", "stage_01_best.pt")
    },
    {
        "name": "stage_02",
        "yaml": os.path.join(BASE_DIR, "dataset_stage02.yaml"),
        "dataset_dir": os.path.join(BASE_DIR, "dataset_stage02"),
        "target_model": os.path.join(BASE_DIR, "models", "stage_02_best.pt")
    },
    {
        "name": "stage_03",
        "yaml": os.path.join(BASE_DIR, "dataset_stage03.yaml"),
        "dataset_dir": os.path.join(BASE_DIR, "dataset_stage03"),
        "target_model": os.path.join(BASE_DIR, "models", "stage_03_best.pt")
    },
]

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

for stage in stages:
    yaml_path = stage["yaml"]
    dataset_dir = stage["dataset_dir"]

    # Convert Windows backslashes to forward slashes before injecting into f-string
    formatted_dataset_dir = dataset_dir.replace("\\", "/")

    # Auto-generate YAML config if missing
    if not os.path.exists(yaml_path):
        print(f"[INFO] Creating missing config file: {yaml_path}")
        yaml_content = (
            f"path: {formatted_dataset_dir}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"names:\n"
            f"  0: washer\n"
            f"  1: bolt\n"
        )
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

    print(f"\n==========================================")
    print(f" STARTING TRAINING FOR: {stage['name'].upper()}")
    print(f" Config: {yaml_path}")
    print(f" Dataset Directory: {dataset_dir}")
    print(f"==========================================\n")

    # Verify image dataset exists before training
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    if not os.path.exists(train_img_dir):
        print(f"[WARNING] Skipping {stage['name']} — dataset folder missing: {train_img_dir}")
        continue

    # Train model
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project=os.path.join(BASE_DIR, "runs", "detect"),
        name=stage["name"],
        exist_ok=True
    )

    # Copy trained weights to target directory
    best_weights_path = os.path.join(BASE_DIR, "runs", "detect", stage["name"], "weights", "best.pt")
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, stage["target_model"])
        print(f"[SUCCESS] Model saved to {stage['target_model']}")

print("\nWorkflow processing complete!")