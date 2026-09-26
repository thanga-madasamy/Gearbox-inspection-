import os

LABEL_DIRS = [
    "dataset_stage01/labels/train",
    "dataset_stage01/labels/val",
]

def swap_classes(label_dir):
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(label_dir, fname)
        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            new_cls_id = 1 - cls_id  # swaps 0<->1
            new_lines.append(" ".join([str(new_cls_id)] + parts[1:]))

        with open(path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        print(f"Updated {fname}: {len(new_lines)} labels swapped")

for d in LABEL_DIRS:
    if os.path.exists(d):
        swap_classes(d)

print("Done. Now retrain using the updated data.yaml with washer=0, bolt=1.")