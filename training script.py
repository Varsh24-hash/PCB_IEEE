import os
import sys
import glob
import shutil
import zipfile
import yaml
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.nn.modules import block

# ==============================================================================
# 0. DATASET SETUP HELPERS
# ==============================================================================
# --- Class Map (6 PCB Defect Types from PKU-Market / Synthetic) ---
CLASS_NAMES = {
    0: 'Open',
    1: 'Short',
    2: 'Mousebite',
    3: 'Spur',
    4: 'Missing-hole',
    5: 'Copper',
}
NUM_CLASSES = len(CLASS_NAMES)

# --- Label normalization: convert polygon/OBB labels to bounding boxes ---
def polygon_to_bbox(coords):
    """Convert a list of (x, y) polygon vertices to YOLO bbox (cx, cy, w, h)."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def normalize_label_file(src_path, dst_path):
    """
    Read a YOLO label file that may contain mixed formats:
      - Standard bbox:  class cx cy w h          (5 fields)
      - Polygon/OBB:    class x1 y1 x2 y2 ...   (odd number of fields > 5)
    Write a normalized file where every line is: class cx cy w h
    """
    lines_out = []
    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip empty or malformed lines
            cls_id = parts[0]
            values = [float(v) for v in parts[1:]]

            if len(values) == 4:
                # Already standard YOLO bbox format
                lines_out.append(f"{cls_id} {values[0]} {values[1]} {values[2]} {values[3]}")
            elif len(values) >= 6 and len(values) % 2 == 0:
                # Polygon format: pairs of (x, y) coordinates
                coords = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
                cx, cy, w, h = polygon_to_bbox(coords)
                lines_out.append(f"{cls_id} {cx} {cy} {w} {h}")
            else:
                # Fallback: treat first 4 values as bbox
                lines_out.append(f"{cls_id} {values[0]} {values[1]} {values[2]} {values[3]}")

    with open(dst_path, 'w') as f:
        f.write('\n'.join(lines_out))
        if lines_out:
            f.write('\n')


def setup_dataset(dataset_path=None, zip_name='dataset.zip', extract_to='temp_dataset'):
    """
    Locate or extract the dataset. Supports:
      1. Direct path to a folder containing images/ and labels/
      2. Zip file extraction (legacy)

    Returns the path to the dataset root (parent of images/ and labels/).
    """
    # --- Option 1: Direct path provided ---
    if dataset_path and os.path.isdir(dataset_path):
        img_dir = os.path.join(dataset_path, 'images')
        lbl_dir = os.path.join(dataset_path, 'labels')
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            print(f"✅ Dataset found at: {dataset_path}")
            print(f"   Images: {len(os.listdir(img_dir))}, Labels: {len(os.listdir(lbl_dir))}")
            return dataset_path

    # --- Option 2: Zip extraction ---
    cwd = os.getcwd()
    zip_path = os.path.join(cwd, zip_name)
    extract_path = os.path.join(cwd, extract_to)

    if os.path.exists(extract_path):
        print(f"✅ Dataset already extracted to: {extract_path}")
    elif os.path.exists(zip_path):
        print(f"⏳ Extracting '{zip_name}'... (This may take a moment)")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction complete.")
    else:
        raise FileNotFoundError(
            f"❌ No dataset found. Provide a valid path or place '{zip_name}' in {cwd}."
        )

    # Walk to find images/ folder
    for root, dirs, files in os.walk(extract_path):
        if 'images' in dirs:
            img_dir = os.path.join(root, 'images')
            if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(img_dir)):
                print(f"📂 Dataset root found at: {root}")
                return root

    raise FileNotFoundError("❌ Could not find an 'images' folder with image files.")


def normalize_labels_if_needed(dataset_root):
    """
    Check if any label files have polygon/OBB format (>5 fields per line).
    If so, normalize all labels to standard YOLO bbox format.
    Creates a backup of original labels before normalization.

    This is required for this dataset: files mixing 5-field bbox lines with
    polygon lines are misread by Ultralytics (it reshapes every line as a
    polygon, turning "cx cy w h" into two bogus vertices).
    """
    labels_dir = os.path.join(dataset_root, 'labels')
    needs_normalization = False

    # Scan every file; a partial scan can miss polygon lines
    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(labels_dir, fname)
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 5:
                    needs_normalization = True
                    break
        if needs_normalization:
            break

    if not needs_normalization:
        print("✅ Labels already in standard YOLO bbox format.")
        return

    print("⏳ Normalizing labels from polygon/OBB to YOLO bbox format...")

    # Backup original labels
    backup_dir = os.path.join(dataset_root, 'labels_original_backup')
    if not os.path.exists(backup_dir):
        shutil.copytree(labels_dir, backup_dir)
        print(f"   📋 Original labels backed up to: {backup_dir}")

    # Normalize in-place
    count = 0
    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(labels_dir, fname)
        normalize_label_file(fpath, fpath)
        count += 1

    print(f"✅ Normalized {count} label files.")


def get_dominant_class(label_path):
    """
    Get the most frequent class ID in a label file for stratified splitting.
    Returns -1 for empty label files (background images).
    """
    if not os.path.exists(label_path):
        return -1
    with open(label_path, 'r') as f:
        classes = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.append(int(parts[0]))
    if not classes:
        return -1  # background / no annotations
    # Return the most common class in this image
    return Counter(classes).most_common(1)[0][0]


# ==============================================================================
# 1. DEFINE & REGISTER CUSTOM ATTENTION MODULE
# ==============================================================================
class FeatureGatedAttention(nn.Module):
    """
    Feature-Gated Attention (FGA) module.
    Applies channel-wise gating using global average pooling and
    a lightweight SE-style bottleneck (squeeze-excitation).
    """
    def __init__(self, c1, c2=None, kernel_size=1, stride=1, p=None, g=1, act=True):
        super().__init__()
        # parse_model passes YAML args verbatim to unregistered modules, so
        # only c1 arrives; output channels must equal input for x * gate
        c2 = c2 or c1
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduction = 16
        c_inner = max(1, c1 // reduction)
        self.gate_layers = nn.Sequential(
            nn.Conv2d(c1, c_inner, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_inner, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate_layers(self.gap(x))

# Register FGA so YOLO can find it when parsing the model YAML
# (parse_model resolves module names via globals() of ultralytics.nn.tasks)
setattr(tasks, 'FeatureGatedAttention', FeatureGatedAttention)
setattr(block, 'FeatureGatedAttention', FeatureGatedAttention)

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
IMG_SIZE = 640
BATCH_SIZE = 16
N_FOLDS = 5

# Ultralytics never auto-selects Apple MPS (it falls back to CPU), so pick explicitly
DEVICE = '0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# --- Phase 1: Hyperparameter Optimization ---
PROXY_EPOCHS = 100     # Short training epochs for HPO trials
HPO_TRIALS = 50        # Number of genetic algorithm iterations

# --- Phase 2: Definitive Training ---
FULL_EPOCHS = 200
PATIENCE = 25

# Robust augmentation hyperparameters (frozen across all experiments)
ROBUST_HYPS = {
    'degrees': 10.0, 'translate': 0.1, 'scale': 0.5, 'shear': 2.0,
    'perspective': 0.0005, 'flipud': 0.0, 'fliplr': 0.5,
    'mosaic': 1.0, 'mixup': 0.15, 'copy_paste': 0.1,
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
}

# Model architecture YAML (must match the file in this folder)
MODEL_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'fga_yolov12l_custom.yaml')

# Absolute output root: Ultralytics nests relative `project` paths under
# its own runs_dir/<task>/, which breaks hard-coded result paths
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')

# ==============================================================================
# 3. PHASE 1: HYPERPARAMETER OPTIMIZATION (Genetic Algorithm / Evolutionary)
# ==============================================================================
def run_phase_1_hpo(dataset_root, subset_fraction=0.3):
    """
    Phase 1: Outer Objective — Find optimal hyperparameters (λ*).

    Uses a smaller proxy model (YOLOv12-Medium equivalent = yolov8m) on a
    subset of data with shorter training to efficiently search the
    hyperparameter space via YOLO's built-in evolutionary strategy.

    Returns:
        dict: Best hyperparameters found (lr0, momentum, weight_decay, etc.)
    """
    print("\n" + "=" * 60)
    print("PHASE 1: HYPERPARAMETER OPTIMIZATION (Evolutionary Search)")
    print("=" * 60)

    # Use a subset of data for efficiency
    images_path = os.path.join(dataset_root, 'images')
    all_imgs = sorted([f for f in os.listdir(images_path)
                       if f.endswith(('.jpg', '.png', '.jpeg'))])

    subset_size = max(200, int(len(all_imgs) * subset_fraction))
    np.random.seed(42)
    subset_imgs = list(np.random.choice(all_imgs, size=min(subset_size, len(all_imgs)),
                                         replace=False))

    # Create subset directories
    hpo_dir = os.path.join(RUNS_DIR, 'hpo_subset')
    hpo_img_dir = os.path.join(hpo_dir, 'images')
    hpo_lbl_dir = os.path.join(hpo_dir, 'labels')
    os.makedirs(hpo_img_dir, exist_ok=True)
    os.makedirs(hpo_lbl_dir, exist_ok=True)

    for img_name in subset_imgs:
        src_img = os.path.join(dataset_root, 'images', img_name)
        dst_img = os.path.join(hpo_img_dir, img_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)

        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl = os.path.join(dataset_root, 'labels', lbl_name)
        dst_lbl = os.path.join(hpo_lbl_dir, lbl_name)
        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)

    # Split subset into train/val for HPO
    train_imgs, val_imgs = train_test_split(subset_imgs, test_size=0.2, random_state=42)

    # Write split files
    with open(os.path.join(hpo_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join([os.path.abspath(os.path.join(hpo_img_dir, img))
                           for img in train_imgs]))
    with open(os.path.join(hpo_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join([os.path.abspath(os.path.join(hpo_img_dir, img))
                           for img in val_imgs]))

    # HPO data YAML
    hpo_yaml_path = os.path.join(hpo_dir, 'data.yaml')
    with open(hpo_yaml_path, 'w') as f:
        yaml.dump({
            'path': os.path.abspath(hpo_dir),
            'train': 'train.txt',
            'val': 'val.txt',
            'names': CLASS_NAMES,
            'nc': NUM_CLASSES,
        }, f)

    print(f"📊 HPO subset: {len(train_imgs)} train, {len(val_imgs)} val images")
    print(f"🔬 Running evolutionary search ({HPO_TRIALS} iterations × {PROXY_EPOCHS} epochs)...\n")

    # Use a lighter proxy model for speed during HPO
    # (Use yolov8m as proxy — smaller, faster iterations)
    proxy_model = YOLO('yolov8m.pt')

    # YOLO's built-in tune() runs the genetic algorithm / evolutionary HPO
    try:
        result_dir = proxy_model.tune(
            data=hpo_yaml_path,
            epochs=PROXY_EPOCHS,
            iterations=HPO_TRIALS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=os.path.join(RUNS_DIR, 'hpo'),
            name='evolve',
            optimizer='AdamW',
            plots=True,
            val=True,
            **ROBUST_HYPS,
        )

        # Tuner writes best_hyperparameters.yaml into its (possibly
        # auto-incremented) tune dir: runs/hpo/evolve, evolve2, ...
        best_yamls = glob.glob(os.path.join(RUNS_DIR, 'hpo', 'evolve*',
                                            'best_hyperparameters.yaml'))
        if best_yamls:
            with open(max(best_yamls, key=os.path.getmtime), 'r') as yf:
                tuned = yaml.safe_load(yf) or {}
            best_hyps = {
                'lr0': float(tuned.get('lr0', 0.0091)),
                'momentum': float(tuned.get('momentum', 0.937)),
                'weight_decay': float(tuned.get('weight_decay', 0.0005)),
            }
            print(f"\n✅ Phase 1 Complete — Optimal Config (λ*): {best_hyps}")
            return best_hyps
    except Exception as e:
        print(f"⚠️  HPO tune() failed: {e}")
        print("   Falling back to default hyperparameters.")

    # Fallback defaults (from the paper)
    best_hyps = {'lr0': 0.0091, 'momentum': 0.937, 'weight_decay': 0.0005}
    print(f"📌 Using default hyperparameters: {best_hyps}")
    return best_hyps


# ==============================================================================
# 4. PHASE 2: DEFINITIVE TRAINING — 5-FOLD STRATIFIED CV WITH FGA-YOLOv12
# ==============================================================================
def run_phase_2_cv(dataset_root, best_hyps):
    """
    Phase 2: Inner Objective — Train FGA-YOLOv12-L with 5-Fold Stratified CV.

    Uses the optimal hyperparameters (λ*) from Phase 1.
    Tracks the best generalization model (Ω*) across all folds.

    Args:
        dataset_root: Path to the dataset (parent of images/ and labels/)
        best_hyps: Optimal hyperparameters dict from Phase 1
    """
    print("\n" + "=" * 60)
    print("PHASE 2: DEFINITIVE TRAINING (5-Fold Stratified CV)")
    print("=" * 60)

    # --- Gather all images ---
    images_path = os.path.join(dataset_root, 'images')
    labels_path = os.path.join(dataset_root, 'labels')
    all_imgs = sorted([f for f in os.listdir(images_path)
                       if f.endswith(('.jpg', '.png', '.jpeg'))])

    if len(all_imgs) == 0:
        print(f"❌ ERROR: No images found in {images_path}.")
        return

    print(f"📊 Total images: {len(all_imgs)}")

    # --- Build stratification labels (dominant class per image) ---
    strat_labels = []
    for img_name in all_imgs:
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(labels_path, lbl_name)
        strat_labels.append(get_dominant_class(lbl_path))

    strat_labels = np.array(strat_labels)
    X = np.array(all_imgs)

    # Use Stratified K-Fold for balanced class distribution across folds
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    best_score = 0.0
    best_model_path = None

    for fold_idx, (train_full_idx, test_idx) in enumerate(skf.split(X, strat_labels)):
        print(f"\n{'─' * 50}")
        print(f"  📁 Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'─' * 50}")

        # --- A. Data Splits ---
        X_test = X[test_idx]
        X_train_full = X[train_full_idx]
        strat_train_full = strat_labels[train_full_idx]

        # 90% train, 10% internal validation (from the non-test portion)
        X_train, X_val, _, _ = train_test_split(
            X_train_full, strat_train_full,
            test_size=0.10, random_state=42,
            stratify=strat_train_full
        )

        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # --- B. Setup Fold Directory & Split Files ---
        fold_dir = os.path.join(RUNS_DIR, 'folds', f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        for name, data in [('train', X_train), ('val', X_val), ('test', X_test)]:
            with open(os.path.join(fold_dir, f'{name}.txt'), 'w') as f:
                f.write('\n'.join([
                    os.path.abspath(os.path.join(images_path, img))
                    for img in data
                ]))

        # Create data YAML with all 6 classes
        fold_yaml_path = os.path.join(fold_dir, 'data.yaml')
        with open(fold_yaml_path, 'w') as f:
            yaml.dump({
                'path': os.path.abspath(fold_dir),
                'train': 'train.txt',
                'val': 'val.txt',
                'test': 'test.txt',
                'names': CLASS_NAMES,
                'nc': NUM_CLASSES,
            }, f)

        # --- C. Initialize Custom FGA-YOLOv12-L Model ---
        print(f"   🔧 Loading FGA-YOLOv12-L architecture from: {MODEL_YAML}")
        model = YOLO(MODEL_YAML)

        # Transfer learning: load COCO-pretrained weights
        # Mismatch in final detection head is expected (6 vs 80 classes)
        try:
            model.load('yolov12l.pt')
            print("   ✅ Loaded YOLOv12-L COCO pretrained weights (transfer learning)")
        except Exception as e:
            try:
                model.load('yolov8l.pt')
                print("   ⚠️  YOLOv12l.pt not found, loaded YOLOv8l.pt as fallback")
            except Exception:
                print(f"   ⚠️  Could not load pretrained weights: {e}")
                print("       Training from scratch.")

        # --- D. Train (Inner Objective: minimize L_train + R_gate) ---
        final_args = {**best_hyps, **ROBUST_HYPS}

        model.train(
            data=fold_yaml_path,
            epochs=FULL_EPOCHS,
            patience=PATIENCE,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            project=os.path.join(RUNS_DIR, f'fold_{fold_idx + 1}'),
            name='fga_yolov12l',
            verbose=True,
            save=True,
            save_period=50,          # Save checkpoint every 50 epochs
            plots=True,             # Save training plots
            optimizer='AdamW',
            **final_args
        )

        # --- E. Evaluate on Test Split ---
        metrics = model.val(data=fold_yaml_path, split='test', device=DEVICE)
        fold_map = metrics.box.map  # mAP@50-95
        fold_map50 = metrics.box.map50  # mAP@50

        fold_scores.append(fold_map)
        print(f"\n   📈 Fold {fold_idx + 1} Results:")
        print(f"      mAP@50:    {fold_map50:.4f}")
        print(f"      mAP@50-95: {fold_map:.4f}")

        # Per-class results
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            print("      Per-class mAP@50-95:")
            for cls_idx, cls_map in enumerate(metrics.box.maps):
                if cls_idx < NUM_CLASSES:
                    print(f"        {CLASS_NAMES[cls_idx]:15s}: {cls_map:.4f}")

        # --- F. Track Best Generalization Model (Ω*) ---
        if fold_map > best_score:
            best_score = fold_map
            # Save best model weights
            # Use the trainer's actual save_dir (name may be auto-incremented)
            best_weights = os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')
            if os.path.exists(best_weights):
                best_model_dir = os.path.join(RUNS_DIR, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                best_model_path = os.path.join(best_model_dir, 'best.pt')
                shutil.copy2(best_weights, best_model_path)
                print(f"   🏆 New best model! (mAP@50-95: {fold_map:.4f})")
                print(f"      Saved to: {best_model_path}")

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — FINAL RESULTS")
    print("=" * 60)
    print(f"  Fold Scores (mAP@50-95): {[f'{s:.4f}' for s in fold_scores]}")
    print(f"  Mean mAP@50-95:          {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  Best Single Fold:        {max(fold_scores):.4f}")
    if best_model_path:
        print(f"  Best Model (Ω*):         {best_model_path}")
    print("=" * 60)

    return fold_scores, best_model_path


# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # ── Dataset Path ──
    # Option 1: Pass as command-line argument
    #   python "training script.py" /path/to/dataset
    # Option 2: Edit this default path
    # Option 3: Place dataset.zip in the current working directory
    DEFAULT_DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'dataset'
    )

    dataset_input = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_PATH

    try:
        # 1. Setup Dataset
        dataset_path = setup_dataset(dataset_path=dataset_input)

        # 2. Normalize labels (polygon/OBB → standard YOLO bbox)
        normalize_labels_if_needed(dataset_path)

        # 3. Phase 1: Hyperparameter Optimization (find λ*)
        print("\n🔬 Starting Phase 1: Hyperparameter Optimization...")
        best_hyps = run_phase_1_hpo(dataset_path, subset_fraction=0.3)

        # 4. Phase 2: Definitive 5-Fold CV Training (find Ω*)
        print("\n🚀 Starting Phase 2: Definitive Training with FGA-YOLOv12-L...")
        fold_scores, best_model = run_phase_2_cv(dataset_path, best_hyps)

        # 5. Optional: Run explainability analysis
        if best_model and os.path.exists(best_model):
            print(f"\n🔍 Best model saved at: {best_model}")
            print("   Load with: model = YOLO('runs/best_model/best.pt')")
            print("   Predict:   results = model.predict('your_image.jpg')")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
