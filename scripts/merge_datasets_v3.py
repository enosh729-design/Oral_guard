import os, shutil, glob, yaml, random
from pathlib import Path

# Auto-detect class mappings from data.yaml files
def build_class_map(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    mapping = {}
    for i, name in enumerate(names):
        n = name.lower()
        if any(k in n for k in ['caries', 'decay', 'cavity', 'carie']):
            if any(k in n for k in ['deep', 'pulp', 'gross', 'abscess', 'involving']):
                mapping[i] = 1  # deep_caries
            else:
                mapping[i] = 0  # caries
        elif any(k in n for k in ['periapical', 'apical', 'lesion']):
            mapping[i] = 2  # periapical_lesion
        elif any(k in n for k in ['impacted', 'impaction', 'unerupted']):
            mapping[i] = 3  # impacted_tooth
    print(f'Class map for {os.path.basename(os.path.dirname(yaml_path))}:')
    for src, dst in mapping.items():
        print(f'  {names[src]} ({src}) -> OralGuard class {dst}')
    return mapping

output_base = r'C:\Users\enosh\oralguard\data\combined'
all_pairs = []

# Process each new dataset
new_datasets = [
    r'C:\Users\enosh\oralguard\data\finetune\celldetection_panoramic',
    r'C:\Users\enosh\oralguard\data\finetune\coded_ai_panoramic',
    r'C:\Users\enosh\oralguard\data\finetune\vzrad2',
    r'C:\Users\enosh\oralguard\data\finetune\panoramic_xray_caries',
]

for ds_path in new_datasets:
    yaml_files = glob.glob(
        os.path.join(ds_path, '**', 'data.yaml'), 
        recursive=True
    )
    if not yaml_files:
        print(f'No data.yaml found in {ds_path}, skipping')
        continue
    
    class_map = build_class_map(yaml_files[0])
    if not class_map:
        print(f'No matching classes in {ds_path}, skipping')
        continue

    for split in ['train', 'valid', 'val', 'test']:
        found_split = False
        for img_dir_pattern in [
            os.path.join(ds_path, split, 'images'),
            os.path.join(ds_path, 'images', split),
            os.path.join(ds_path, split),
        ]:
            if os.path.exists(img_dir_pattern):
                lbl_dir = img_dir_pattern.replace('images', 'labels')
                imgs = glob.glob(os.path.join(img_dir_pattern, '*.jpg'))
                imgs += glob.glob(os.path.join(img_dir_pattern, '*.png'))
                imgs += glob.glob(os.path.join(img_dir_pattern, '*.JPG'))
                imgs += glob.glob(os.path.join(img_dir_pattern, '*.PNG'))
                imgs = list(set(imgs))
                if len(imgs) > 0:
                    for img in imgs:
                        base = Path(img).stem
                        lbl = os.path.join(lbl_dir, base + '.txt')
                        all_pairs.append((img, lbl, class_map))
                    found_split = True
                    break

print(f'Total new image-label pairs: {len(all_pairs)}')

def remap_label(src, dst, class_map):
    lines_out = []
    if os.path.exists(src):
        with open(src) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                    if cls in class_map:
                        new_cls = class_map[cls]
                        lines_out.append(
                            f'{new_cls} {" ".join(parts[1:])}'
                        )
                except ValueError:
                    continue
    with open(dst, 'w') as f:
        f.write('\n'.join(lines_out))
    return len(lines_out)

random.seed(2026)
random.shuffle(all_pairs)
n = len(all_pairs)
splits = {
    'train': all_pairs[:int(n*0.80)],
    'val':   all_pairs[int(n*0.80):int(n*0.95)],
    'test':  all_pairs[int(n*0.95):]
}

added = 0
for split_name, pairs in splits.items():
    img_out = os.path.join(output_base, 'images', split_name)
    lbl_out = os.path.join(output_base, 'labels', split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_src, lbl_src, class_map in pairs:
        ext = Path(img_src).suffix
        uid = f'v3_{added:07d}'
        if os.path.exists(img_src):
            shutil.copy2(img_src, os.path.join(img_out, uid+ext))
            remap_label(
                lbl_src,
                os.path.join(lbl_out, uid+'.txt'),
                class_map
            )
            added += 1

print(f'Added {added} new images to combined dataset')

# Count updated class distribution
counts = {0:0, 1:0, 2:0, 3:0}
all_labels = glob.glob(
    os.path.join(output_base, 'labels', 'train', '*.txt')
)
for lbl in all_labels:
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    cls = int(parts[0])
                    if cls in counts:
                        counts[cls] += 1
                except ValueError:
                    continue

names = ['caries','deep_caries','periapical_lesion','impacted_tooth']
total = sum(counts.values()) or 1
print('Updated class distribution in train split:')
for i, name in enumerate(names):
    print(f'  {name}: {counts[i]} ({counts[i]/total*100:.1f}%)')
print(f'Total annotations: {total}')
