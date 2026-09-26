# auto_annotate_remaining.py
import os
import sys
import glob
from ultralytics import YOLO

CONF_THRESHOLD = 0.25


def get_latest_weights():
    """Finds the most recently trained model automatically."""
    candidates = glob.glob("runs/detect/*/weights/best.pt")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def auto_annotate_remaining(images_dir, output_labels_dir, already_labeled_dirs=None):
    """
    Annotates only images that don't already have a label file.
    already_labeled_dirs: list of folders to check for existing labels
    (e.g. your manually labeled folder + previous auto-labels folder)
    """
    model_path = get_latest_weights()
    if not model_path:
        print("ERROR: No trained model found under runs/detect/*/weights/best.pt")
        print("Train a model first (run auto.py).")
        return

    print(f"Using model: {model_path}")
    model = YOLO(model_path)

    os.makedirs(output_labels_dir, exist_ok=True)

    if already_labeled_dirs is None:
        already_labeled_dirs = [output_labels_dir]

    # Build a set of base filenames (no extension) that ALREADY have a label
    already_labeled = set()
    for label_dir in already_labeled_dirs:
        if os.path.exists(label_dir):
            for f in os.listdir(label_dir):
                if f.lower().endswith(".txt"):
                    already_labeled.add(os.path.splitext(f)[0])

    print(f"Found {len(already_labeled)} images already labeled — these will be skipped.\n")

    # Find all images
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(image_extensions):
                base = os.path.splitext(f)[0]
                if base not in already_labeled:
                    image_files.append(os.path.join(root, f))

    if not image_files:
        print("No remaining images to annotate — everything is already labeled.")
        return

    print(f"Found {len(image_files)} REMAINING images to auto-annotate...\n")

    total_objects = 0
    for img_path in image_files:
        results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)

        img_filename = os.path.basename(img_path)
        label_file = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(output_labels_dir, label_file)

        num_objects = 0
        with open(label_path, "w") as f:
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    x_center, y_center, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    num_objects += 1

        total_objects += num_objects
        print(f"  {img_filename} -> {num_objects} objects detected")

    print(f"\nDone. {len(image_files)} new images processed, {total_objects} total objects detected.")
    print(f"Labels saved to: {output_labels_dir}")


if __name__ == "__main__":
    # Usage: python auto_annotate_remaining.py <images_folder> [output_labels_folder]
    if len(sys.argv) < 2:
        print("Usage: python auto_annotate_remaining.py <images_folder> [output_labels_folder]")
        sys.exit(1)

    images_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else images_dir.rstrip("/\\") + "_auto_labels"

    # Check against BOTH your manually labeled folder AND previous auto-labels
    # so nothing gets re-annotated
    already_labeled_dirs = [
        "project-13-at-2026-09-25-13-51-aaeb9e8a/labels",  # your manual labels
        output_dir,                                          # previous auto-labels output
    ]

    auto_annotate_remaining(images_dir, output_dir, already_labeled_dirs)