"""
Continue the standard YOLOv8s PCB run for 20 more epochs, starting from its last.pt weights.

True `resume=True` is impossible here: Ultralytics stripped optimizer/EMA/epoch from
last.pt when the first run finished. So this is a new 20-epoch run initialized from
last.pt with the same data split and training settings. Outputs go to
runs/yolov8s_standard/continued_20ep/ and nothing from the first run is overwritten.
"""
import csv
import os
import sys
import time

import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_RUN = os.path.join(ROOT, 'runs', 'yolov8s_standard')
START_WEIGHTS = os.path.join(BASE_RUN, 'train', 'weights', 'last.pt')
DATA_YAML = os.path.join(BASE_RUN, 'data.yaml')   # same train.txt / val.txt split
OUT_DIR = os.path.join(BASE_RUN, 'continued_20ep')

CLASS_NAMES = {0: 'Open', 1: 'Short', 2: 'Mousebite', 3: 'Spur', 4: 'Missing-hole', 5: 'Copper'}
EXTRA_EPOCHS = 20
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def main():
    if os.path.exists(os.path.join(OUT_DIR, 'train')):
        sys.exit(f"{OUT_DIR}/train already exists; refusing to overwrite.")

    model = YOLO(START_WEIGHTS)
    t_train = time.time()
    model.train(
        data=DATA_YAML, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE,
        epochs=EXTRA_EPOCHS, patience=EXTRA_EPOCHS + 1, deterministic=False,
        # Previous run's 'auto' optimizer resolved to these; pin them so they can't change
        optimizer='AdamW', lr0=0.001, momentum=0.9,
        project=OUT_DIR, name='train', exist_ok=False, plots=True,
    )
    train_seconds = time.time() - t_train

    save_dir = str(model.trainer.save_dir)
    best_pt = os.path.join(save_dir, 'weights', 'best.pt')
    # best.pt's 'epoch' field is reset when finalized; use results.csv (fitness: 0.1*mAP50 + 0.9*mAP50-95)
    with open(os.path.join(save_dir, 'results.csv')) as f:
        rows = list(csv.DictReader(f))
    fit = [0.1 * float(r['metrics/mAP50(B)']) + 0.9 * float(r['metrics/mAP50-95(B)']) for r in rows]
    best_epoch = int(rows[fit.index(max(fit))]['epoch'])

    m = YOLO(best_pt).val(data=DATA_YAML, split='val', imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE,
                          project=OUT_DIR, name='val_best', exist_ok=True, plots=True)

    lines = [
        f"Additional training time: {train_seconds:.0f} s ({train_seconds / 60:.1f} min)",
        f"Additional epochs:        {len(rows)}",
        f"Best epoch (of these):    {best_epoch}  (= epoch {5 + best_epoch} overall)",
        f"mAP50:                    {m.box.map50:.4f}",
        f"mAP50-95:                 {m.box.map:.4f}",
        f"Precision:                {m.box.mp:.4f}",
        f"Recall:                   {m.box.mr:.4f}",
        "Per-class (P / R / mAP50 / mAP50-95):",
    ]
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        lines.append(f"  {CLASS_NAMES[int(c)]:13s} {p:.4f} / {r:.4f} / {ap50:.4f} / {ap:.4f}")
    lines.append(f"best.pt: {best_pt}")
    lines.append(f"last.pt: {os.path.join(save_dir, 'weights', 'last.pt')}")
    report = '\n'.join(lines)
    print("\n=== FINAL REPORT ===\n" + report)
    with open(os.path.join(OUT_DIR, 'final_report.txt'), 'w') as f:
        f.write(report + '\n')


if __name__ == '__main__':
    sys.exit(main())
