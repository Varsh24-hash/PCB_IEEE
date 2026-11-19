"""
Evaluate detection accuracy of the continued YOLOv8s model on the 760-image validation set.
No training: loads best.pt, runs Ultralytics val (P/R/mAP) and computes box-level accuracy.

Accuracy (per class and overall) = TP / (TP + FP + FN), at IoU >= 0.5 with class match,
using the confidence threshold that maximizes mean F1 (the one Ultralytics uses for P/R).
  TP: prediction matched one-to-one to an unmatched same-class GT box with IoU >= 0.5
      (greedy, highest confidence first)
  FP: any unmatched prediction (wrong location or wrong class)
  FN: any unmatched GT box
Detection has no true negatives, so this is the standard detection "accuracy" (a.k.a. CSI/Jaccard).
"""
import json
import os

import numpy as np
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, 'runs', 'yolov8s_standard')
BEST_PT = os.path.join(RUN, 'continued_20ep', 'train', 'weights', 'best.pt')
DATA_YAML = os.path.join(RUN, 'data.yaml')
OUT_DIR = os.path.join(RUN, 'continued_20ep', 'accuracy')

CLASS_NAMES = {0: 'Open', 1: 'Short', 2: 'Mousebite', 3: 'Spur', 4: 'Missing-hole', 5: 'Copper'}
IMG_SIZE = 640
IOU_MATCH = 0.5
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def iou_matrix(a, b):
    """IoU between xyxy boxes a (N,4) and b (M,4)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_gt(img_path, w, h):
    lbl = img_path.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}').rsplit('.', 1)[0] + '.txt'
    rows = [l.split() for l in open(lbl) if l.strip()] if os.path.exists(lbl) else []
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, dtype=float)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return a[:, 0].astype(int), np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(BEST_PT)

    # 1) Standard Ultralytics validation (P, R at max-F1 confidence; mAP50; mAP50-95)
    m = model.val(data=DATA_YAML, split='val', imgsz=IMG_SIZE, batch=16, device=DEVICE,
                  project=OUT_DIR, name='val', exist_ok=True, plots=False)
    conf_thr = float(m.box.px[int(m.box.f1_curve.mean(0).argmax())])

    # 2) Box-level TP/FP/FN at that confidence threshold
    with open(os.path.join(RUN, 'val.txt')) as f:
        val_imgs = [l.strip() for l in f if l.strip()]
    tp, fp, fn = (np.zeros(len(CLASS_NAMES), int) for _ in range(3))
    # One image per call: a list source is run as a single batch (760 images -> MPS OOM)
    for img in val_imgs:
        res = model.predict(img, imgsz=IMG_SIZE, conf=conf_thr, device=DEVICE, verbose=False)[0]
        h, w = res.orig_shape
        g_cls, g_box = load_gt(res.path, w, h)
        p_cls = res.boxes.cls.cpu().numpy().astype(int)
        p_box = res.boxes.xyxy.cpu().numpy()
        order = np.argsort(-res.boxes.conf.cpu().numpy())
        ious = iou_matrix(p_box, g_box) if len(p_box) and len(g_box) else np.zeros((len(p_box), len(g_box)))
        g_used = np.zeros(len(g_cls), bool)
        for i in order:
            cand = np.where((g_cls == p_cls[i]) & ~g_used & (ious[i] >= IOU_MATCH))[0]
            if len(cand):
                g_used[cand[ious[i, cand].argmax()]] = True
                tp[p_cls[i]] += 1
            else:
                fp[p_cls[i]] += 1
        for c in g_cls[~g_used]:
            fn[c] += 1

    acc = tp / np.maximum(tp + fp + fn, 1)
    results = {
        'confidence_threshold': round(conf_thr, 4),
        'iou_match': IOU_MATCH,
        'overall': {
            'accuracy_pct': round(100 * tp.sum() / (tp.sum() + fp.sum() + fn.sum()), 2),
            'macro_accuracy_pct': round(100 * acc.mean(), 2),
            'TP': int(tp.sum()), 'FP': int(fp.sum()), 'FN': int(fn.sum()),
            'precision_recomputed': round(tp.sum() / (tp.sum() + fp.sum()), 4),
            'recall_recomputed': round(tp.sum() / (tp.sum() + fn.sum()), 4),
            'ultralytics_precision': round(float(m.box.mp), 4),
            'ultralytics_recall': round(float(m.box.mr), 4),
            'mAP50': round(float(m.box.map50), 4),
            'mAP50-95': round(float(m.box.map), 4),
        },
        'per_class': {},
    }
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        c = int(c)
        results['per_class'][CLASS_NAMES[c]] = {
            'accuracy_pct': round(100 * acc[c], 2), 'TP': int(tp[c]), 'FP': int(fp[c]), 'FN': int(fn[c]),
            'precision': round(p, 4), 'recall': round(r, 4), 'mAP50': round(ap50, 4), 'mAP50-95': round(ap, 4),
        }

    print(json.dumps(results, indent=2))
    with open(os.path.join(OUT_DIR, 'accuracy_report.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
