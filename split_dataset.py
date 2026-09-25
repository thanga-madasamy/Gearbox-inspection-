# split_dataset.py
import os
import shutil
import random

SOURCE_IMAGES = r"project-13-at-2026-09-25-13-51-aaeb9e8a/images"
SOURCE_LABELS = r"project-13-at-2026-09-25-13-51-aaeb9e8a/labels"

DEST = r"dataset_stage01"
VAL_RATIO = 0.2

def split_dataset():
    print(f"Source images: {os.path.abspath(SOURCE_IMAGES)}")
    print(f"Source exists: {os.path.exists(SOURCE_IMAGES)}")

    if not os.path.exists(SOURCE_IMAGES):
        print("ERROR: Source images folder not found. Stopping.")
        return False

    for split in ["train", "val"]:
        os.makedirs(f"{DEST}/images/{split}", exist_ok=True)
        os.makedirs(f"{DEST}/labels/{split}", exist_ok=True)

    images = [f for f in os.listdir(SOURCE_IMAGES) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images in source.")

    if len(images) == 0:
        print("ERROR: No images found in source folder. Stopping.")
        return False

    random.shuffle(images)
    val_count = max(1, int(len(images) * VAL_RATIO))
    val_images = images[:val_count]
    train_images = images[val_count:]

    def copy_pair(img_list, split):
        copied = 0
        for img in img_list:
            base = os.path.splitext(img)[0]
            label = base + ".txt"

            src_img = os.path.join(SOURCE_IMAGES, img)
            dst_img = f"{DEST}/images/{split}/{img}"
            shutil.copy(src_img, dst_img)

            label_src = os.path.join(SOURCE_LABELS, label)
            if os.path.exists(label_src):
                shutil.copy(label_src, f"{DEST}/labels/{split}/{label}")
                copied += 1
            else:
                print(f"  WARNING: no label found for {img}")
        return copied

    train_copied = copy_pair(train_images, "train")
    val_copied = copy_pair(val_images, "val")

    print(f"\nTrain: {len(train_images)} images ({train_copied} with labels)")
    print(f"Val: {len(val_images)} images ({val_copied} with labels)")

    # Verify files actually landed on disk
    val_dir = f"{DEST}/images/val"
    actual_val_files = os.listdir(val_dir)
    print(f"\nVerification — files actually in {val_dir}: {len(actual_val_files)}")
    for f in actual_val_files:
        size = os.path.getsize(os.path.join(val_dir, f))
        print(f"  {f} — {size} bytes")

    return len(actual_val_files) > 0


if __name__ == "__main__":
    success = split_dataset()
    if success:
        print("\n✅ Split completed and verified.")
    else:
        print("\n❌ Split failed — do not proceed to training yet.")