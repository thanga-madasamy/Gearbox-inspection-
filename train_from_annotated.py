import os
import shutil
import random
from ultralytics import YOLO

# 1. Base Paths
BASE_DIR = r"C:\Users\z0050910\dev\washer_bolt_project"
DESKTOP_DIR = r"C:\Users\z0050910\OneDrive - ZF Friedrichshafen AG\Desktop\washer and bolting"

IMAGE_SRC_DIR = os.path.join(DESKTOP_DIR, "images_folder_annotated_images")
LABEL_SRC_DIR = os.path.join(DESKTOP_DIR, "images_folder_auto_labels")

STAGES = ["stage_01", "stage_02", "stage_03"]

def prepare_dataset_split(stage_name):
    """Pairs images and auto-generated labels, then splits into train/val sets."""
    dataset_dir = os.path.join(BASE_DIR, f"dataset_{stage_name}")
    
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    val_img_dir = os.path.join(dataset_dir, "images", "val")
    train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
    val_lbl_dir = os.path.join(dataset_dir, "labels", "val")

    # Clean existing dataset directory to avoid stale files
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect images
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(IMAGE_SRC_DIR) if f.lower().endswith(image_exts)]

    matched_pairs = []
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(LABEL_SRC_DIR, label_file)

        if os.path.exists(label_path):
            matched_pairs.append((img_name, label_file))

    print(f"[INFO] Found {len(images)} images and matched {len(matched_pairs)} label files from auto_labels.")

    if not matched_pairs:
        print(f"[ERROR] Could not match any label files in {LABEL_SRC_DIR}")
        return False

    random.seed(42)
    random.shuffle(matched_pairs)

    split_idx = int(len(matched_pairs) * 0.8)
    train_pairs = matched_pairs[:split_idx]
    val_pairs = matched_pairs[split_idx:]

    def copy_pairs(pairs, img_dest, lbl_dest):
        for img_file, lbl_file in pairs:
            shutil.copy(os.path.join(IMAGE_SRC_DIR, img_file), os.path.join(img_dest, img_file))
            shutil.copy(os.path.join(LABEL_SRC_DIR, lbl_file), os.path.join(lbl_dest, lbl_file))

    copy_pairs(train_pairs, train_img_dir, train_lbl_dir)
    copy_pairs(val_pairs, val_img_dir, val_lbl_dir)

    print(f"[SUCCESS] Prepared dataset: {len(train_pairs)} train, {len(val_pairs)} val samples.")
    return True


# 2. Main Execution Loop
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

for stage_name in STAGES:
    print(f"\n==========================================")
    print(f" PROCESSING DATASET & TRAINING: {stage_name.upper()}")
    print(f"==========================================\n")

    if not prepare_dataset_split(stage_name):
        continue

    # Create YAML config
    yaml_path = os.path.join(BASE_DIR, f"dataset_{stage_name}.yaml")
    formatted_dataset_dir = os.path.join(BASE_DIR, f"dataset_{stage_name}").replace("\\", "/")

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

    # Train YOLO
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project=os.path.join(BASE_DIR, "runs", "detect"),
        name=stage_name,
        exist_ok=True
    )

    # Save trained weights
    best_weights_path = os.path.join(BASE_DIR, "runs", "detect", stage_name, "weights", "best.pt")
    target_model_path = os.path.join(BASE_DIR, "models", f"{stage_name}_best.pt")

    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, target_model_path)
        print(f"[SUCCESS] Saved model weights to {target_model_path}")

print("\nAll stage models trained and exported successfully!")