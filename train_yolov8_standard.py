"""
Standard Ultralytics YOLOv8 training on the PCB dataset (single run, time-boxed).

- Pretrained yolov8s.pt, no custom modules, no HPO, no k-fold.
- Labels are converted (polygon -> bbox) into a separate prepared copy under
  runs/; the original dataset/images and dataset/labels are never modified.
- Ultralytics `time=` caps training wall-clock and fits the LR schedule to it.
"""
import csv
import os
import sys
import time
from collections import Counter

import torch
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, 'dataset')
RUN_ROOT = os.path.join(ROOT, 'runs', 'yolov8s_standard')
PREP_DIR = os.path.join(RUN_ROOT, 'dataset')

CLASS_NAMES = {0: 'Open', 1: 'Short', 2: 'Mousebite', 3: 'Spur', 4: 'Missing-hole', 5: 'Copper'}

MODEL = 'yolov8s.pt'
IMG_SIZE = 640
BATCH_SIZE = 16
TIME_HOURS = 0.75   # training budget; leaves headroom for setup + final validation
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def to_bbox_lines(src_path):
    """Return label lines as 'cls cx cy w h'; polygons are converted to their bounding box."""
    out = []
    with open(src_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id, v = parts[0], [float(x) for x in parts[1:]]
            if len(v) == 4:
                cx, cy, w, h = v
            else:
                xs, ys = v[0::2], v[1::2]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def prepare_dataset():
    """Symlink images and write normalized labels into PREP_DIR; return (images, dominant classes)."""
    img_src, lbl_src = os.path.join(SRC_DIR, 'images'), os.path.join(SRC_DIR, 'labels')
    img_dst, lbl_dst = os.path.join(PREP_DIR, 'images'), os.path.join(PREP_DIR, 'labels')
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    strat, counts = [], Counter()
    for img in imgs:
        dst = os.path.join(img_dst, img)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(img_src, img), dst)
        stem = os.path.splitext(img)[0]
        lines = to_bbox_lines(os.path.join(lbl_src, stem + '.txt'))
        with open(os.path.join(lbl_dst, stem + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        classes = [int(l.split()[0]) for l in lines]
        counts.update(classes)
        strat.append(Counter(classes).most_common(1)[0][0] if classes else -1)

    print(f"Prepared {len(imgs)} images; annotations per class: {dict(sorted(counts.items()))}")
    return imgs, strat


def main():
    t_start = time.time()
    imgs, strat = prepare_dataset()
    train, val = train_test_split(imgs, test_size=0.2, random_state=42, stratify=strat)

    img_dir = os.path.join(PREP_DIR, 'images')
    for name, split in [('train', train), ('val', val)]:
        with open(os.path.join(RUN_ROOT, f'{name}.txt'), 'w') as f:
            f.write('\n'.join(os.path.join(img_dir, i) for i in split))
    data_yaml = os.path.join(RUN_ROOT, 'data.yaml')
    with open(data_yaml, 'w') as f:
        yaml.dump({'path': RUN_ROOT, 'train': 'train.txt', 'val': 'val.txt', 'names': CLASS_NAMES}, f)
    print(f"Train: {len(train)}, Val: {len(val)}, Device: {DEVICE}")

    model = YOLO(MODEL)
    t_train = time.time()
    model.train(
        data=data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE,
        time=TIME_HOURS, epochs=100, patience=100, deterministic=False,
        project=RUN_ROOT, name='train', exist_ok=True, plots=True,
    )
    train_seconds = time.time() - t_train

    save_dir = str(model.trainer.save_dir)
    best_pt = os.path.join(save_dir, 'weights', 'best.pt')
    # best.pt's own 'epoch' field is reset to -1 when Ultralytics finalizes it,
    # so take the best epoch from results.csv (Ultralytics fitness: 0.1*mAP50 + 0.9*mAP50-95)
    with open(os.path.join(save_dir, 'results.csv')) as f:
        rows = list(csv.DictReader(f))
    fit = [0.1 * float(r['metrics/mAP50(B)']) + 0.9 * float(r['metrics/mAP50-95(B)']) for r in rows]
    best_epoch = int(rows[fit.index(max(fit))]['epoch'])
    epochs_done = len(rows)

    best = YOLO(best_pt)
    m = best.val(data=data_yaml, split='val', imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE,
                 project=RUN_ROOT, name='val_best', exist_ok=True, plots=True)

    lines = [
        f"Training time:      {train_seconds:.0f} s ({train_seconds / 60:.1f} min)",
        f"Total script time:  {time.time() - t_start:.0f} s",
        f"Epochs completed:   {epochs_done}",
        f"Best epoch:         {best_epoch}",
        f"mAP50:              {m.box.map50:.4f}",
        f"mAP50-95:           {m.box.map:.4f}",
        f"Precision:          {m.box.mp:.4f}",
        f"Recall:             {m.box.mr:.4f}",
        "Per-class (P / R / mAP50 / mAP50-95):",
    ]
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        lines.append(f"  {CLASS_NAMES[int(c)]:13s} {p:.4f} / {r:.4f} / {ap50:.4f} / {ap:.4f}")
    lines.append(f"best.pt:            {best_pt}")
    report = '\n'.join(lines)
    print("\n=== FINAL REPORT ===\n" + report)
    with open(os.path.join(RUN_ROOT, 'final_report.txt'), 'w') as f:
        f.write(report + '\n')


if __name__ == '__main__':
    sys.exit(main())
