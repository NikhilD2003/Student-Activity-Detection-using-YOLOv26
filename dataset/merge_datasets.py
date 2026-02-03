import zipfile
import os
import shutil
import random
import yaml
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ZIP_BEHAVIOR = "Student Behavior.v1i.yolov11.zip"
ZIP_CLASSROOM = "student-classroom-activity.v6i.yolov11.zip"

OUT_DATASET = "dataset"

FINAL_CLASSES = [
    "Looking_Forward",
    "Raising_Hand",
    "Reading",
    "Sleeping",
    "Turning_Around",
    "phone",
    "sleep",
    "study",
]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

random.seed(42)

# ============================================================
# CLEAN WORKSPACE
# ============================================================

if os.path.exists(OUT_DATASET):
    shutil.rmtree(OUT_DATASET)

TMP = "tmp_extract"
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

# ============================================================
# UNZIP BOTH DATASETS
# ============================================================

paths = {}

for zip_path in [ZIP_BEHAVIOR, ZIP_CLASSROOM]:
    name = Path(zip_path).stem
    out = os.path.join(TMP, name)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)

    paths[name] = out

# ============================================================
# CLASS MAPS (FROM INSPECTION)
# ============================================================

behavior_classes = {
    0: "Looking_Forward",
    1: "Raising_Hand",
    2: "Reading",
    3: "Sleeping",
    4: "Turning_Around",
}

classroom_classes = {
    0: "phone",
    1: "sleep",
    2: "study",
}

FINAL_INDEX = {name: i for i, name in enumerate(FINAL_CLASSES)}

# ============================================================
# COLLECT FILES
# ============================================================

def collect_pairs(root):
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")

        if not os.path.exists(img_dir):
            continue

        for img in os.listdir(img_dir):
            lbl = img.rsplit(".", 1)[0] + ".txt"
            pairs.append((img_dir, lbl_dir, img, lbl))

    return pairs


all_items = []

for root in paths.values():
    all_items.extend(collect_pairs(root))

random.shuffle(all_items)

n = len(all_items)
n_train = int(TRAIN_RATIO * n)
n_val = int((TRAIN_RATIO + VAL_RATIO) * n)

splits = {
    "train": all_items[:n_train],
    "val": all_items[n_train:n_val],
    "test": all_items[n_val:],
}

# ============================================================
# CREATE MERGED FOLDERS
# ============================================================

for sp in splits:
    os.makedirs(os.path.join(OUT_DATASET, sp, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DATASET, sp, "labels"), exist_ok=True)

# ============================================================
# COPY + REMAP LABELS
# ============================================================

def remap_label(line, source):
    cls, x, y, w, h = line.split()
    cls = int(cls)

    if source == "behavior":
        name = behavior_classes[cls]
    else:
        name = classroom_classes[cls]

    new_id = FINAL_INDEX[name]

    return f"{new_id} {x} {y} {w} {h}\n"


for sp, items in splits.items():
    for img_dir, lbl_dir, img, lbl in items:

        src_img = os.path.join(img_dir, img)
        src_lbl = os.path.join(lbl_dir, lbl)

        dst_img = os.path.join(OUT_DATASET, sp, "images", img)
        dst_lbl = os.path.join(OUT_DATASET, sp, "labels", lbl)

        shutil.copy(src_img, dst_img)

        source = "behavior" if "Student Behavior" in img_dir else "classroom"

        with open(src_lbl) as f:
            lines = f.readlines()

        new_lines = [remap_label(l, source) for l in lines]

        with open(dst_lbl, "w") as f:
            f.writelines(new_lines)

# ============================================================
# WRITE YAML
# ============================================================

yaml_data = {
    "train": "../train/images",
    "val": "../val/images",
    "test": "../test/images",
    "nc": len(FINAL_CLASSES),
    "names": FINAL_CLASSES,
}

with open(os.path.join(OUT_DATASET, "merged_data.yaml"), "w") as f:
    yaml.dump(yaml_data, f)

print("\nMERGE + REMAP COMPLETE.")
print("Dataset written to:", OUT_DATASET)
