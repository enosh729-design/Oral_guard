import os, shutil, random, glob, collections
from pathlib import Path

output_base = r"C:\Users\enosh\oralguard\data\combined"
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_base, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_base, "labels", split), exist_ok=True)

KEEP_CLASSES = {0, 1, 2, 3}

def remap_label_file(src_label, dst_label, class_map):
    lines_out = []
    if not os.path.exists(src_label):
        open(dst_label, "w").close()
        return 0
    with open(src_label, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        if class_map is not None:
            if cls not in class_map:
                continue
            cls = class_map[cls]
        if cls not in KEEP_CLASSES:
            continue
        lines_out.append(f"{cls} {' '.join(parts[1:])}")
    with open(dst_label, "w") as f:
        f.write("\n".join(lines_out))
    return len(lines_out)

all_pairs = []

# Add original DENTEX data (train + val)
for split in ["train", "val"]:
    img_dir = os.path.join(r"C:\Users\enosh\oralguard\data\dentex\images", split)
    lbl_dir = os.path.join(r"C:\Users\enosh\oralguard\data\dentex\labels", split)
    if os.path.exists(img_dir):
        for img in glob.glob(os.path.join(img_dir, "*")):
            base = Path(img).stem
            lbl = os.path.join(lbl_dir, base + ".txt")
            all_pairs.append((img, lbl, None, "dentex"))

# Add roboflow caries (class 0 -> 0 = caries)
for split in ["train", "valid", "test"]:
    img_dir = os.path.join(r"C:\Users\enosh\oralguard\data\finetune\roboflow_caries", split, "images")
    lbl_dir = os.path.join(r"C:\Users\enosh\oralguard\data\finetune\roboflow_caries", split, "labels")
    if os.path.exists(img_dir):
        for img in glob.glob(os.path.join(img_dir, "*")):
            base = Path(img).stem
            lbl = os.path.join(lbl_dir, base + ".txt")
            all_pairs.append((img, lbl, {0: 0}, "caries"))

print(f"Total image-label pairs collected: {len(all_pairs)}")
source_counts = collections.Counter(p[3] for p in all_pairs)
for src, cnt in source_counts.items():
    print(f"  {src}: {cnt}")

random.seed(42)
random.shuffle(all_pairs)

n = len(all_pairs)
n_train = int(n * 0.80)
n_val = int(n * 0.15)

splits_dict = {
    "train": all_pairs[:n_train],
    "val": all_pairs[n_train:n_train + n_val],
    "test": all_pairs[n_train + n_val:],
}

class_counts = collections.Counter()
total_copied = 0

CLASS_NAMES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]

for split_name, pairs in splits_dict.items():
    for img_src, lbl_src, class_map, source in pairs:
        ext = Path(img_src).suffix
        unique = f"{split_name}_{total_copied:06d}"
        img_dst = os.path.join(output_base, "images", split_name, unique + ext)
        lbl_dst = os.path.join(output_base, "labels", split_name, unique + ".txt")
        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
            n_ann = remap_label_file(lbl_src, lbl_dst, class_map)
            if os.path.exists(lbl_dst):
                with open(lbl_dst) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_counts[int(parts[0])] += 1
            total_copied += 1

print(f"Total images merged: {total_copied}")
print(f"Train: {len(splits_dict['train'])}")
print(f"Val: {len(splits_dict['val'])}")
print(f"Test: {len(splits_dict['test'])}")
print("Class distribution:")
for cls_id in sorted(class_counts):
    name = CLASS_NAMES[cls_id] if cls_id < 4 else f"unknown_{cls_id}"
    print(f"  {cls_id} ({name}): {class_counts[cls_id]}")

yaml_content = """path: C:/Users/enosh/oralguard/data/combined
train: images/train
val: images/val
test: images/test
nc: 4
names: [caries, deep_caries, periapical_lesion, impacted_tooth]
"""
with open(os.path.join(output_base, "dental_combined.yaml"), "w") as f:
    f.write(yaml_content)
print("YAML config written")
