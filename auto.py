# auto.py
import os
import glob
from ultralytics import YOLO

BOOTSTRAP_DATA_YAML = "data.yaml"
BOOTSTRAP_MODEL_NAME = "bootstrap_run1"

UNLABELED_IMAGES_DIR = "images"
AUTO_LABELS_OUTPUT_DIR = "images_auto_labels"
CONF_THRESHOLD = 0.25


def train_bootstrap():
    print("Training bootstrap model...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=BOOTSTRAP_DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=8,
        name=BOOTSTRAP_MODEL_NAME
    )
    print("Bootstrap training complete.\n")


def get_latest_weights():
    candidates = glob.glob("runs/detect/*/weights/best.pt")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def auto_annotate():
    model_path = get_latest_weights()

    if not model_path:
        print("ERROR: No trained model found. Training must have failed — check the log above for errors.")
        return

    print(f"Using model: {model_path}")
    model = YOLO(model_path)
    os.makedirs(AUTO_LABELS_OUTPUT_DIR, exist_ok=True)

    image_files = [
        f for f in os.listdir(UNLABELED_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print(f"No images found in {UNLABELED_IMAGES_DIR}")
        return

    print(f"Auto-annotating {len(image_files)} images...\n")

    for img_file in image_files:
        img_path = os.path.join(UNLABELED_IMAGES_DIR, img_file)
        results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(AUTO_LABELS_OUTPUT_DIR, label_file)

        with open(label_path, "w") as f:
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    x_center, y_center, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        print(f"  {img_file} -> {len(results[0].boxes)} objects detected")

    print(f"\nDone. Labels saved to: {AUTO_LABELS_OUTPUT_DIR}")


if __name__ == "__main__":
    train_bootstrap()
    auto_annotate()