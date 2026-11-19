# Hierarchical Optimization Framework for Robust PCB Defect Detection via Adaptive YOLOv12 Architecture

Official code and dataset for the paper published in **IEEE Access, vol. 14, 2026** (DOI: [10.1109/ACCESS.2026.3663419](https://doi.org/10.1109/ACCESS.2026.3663419)).

This repository contains an FGA-integrated YOLOv12-Large detector for printed circuit board (PCB) defects, a bilevel genetic-algorithm hyperparameter search, a synthetic defect generation pipeline, and the hybrid dataset used in the paper.

---

## Highlights

- **Bilevel optimization:** a genetic algorithm (outer loop) tunes the training regime over 50 mini-trials with a YOLOv12-Medium proxy, and the inner loop trains the network weights and the FGA gates.
- **Feature-Gated Attention (FGA):** a squeeze-and-excitation style channel gate (r = 16, Sigmoid) added after the neck at P3, P4 and P5, just before the detection head.
- **Synthetic defect pipeline:** polygon-mask patch library, layout-aware placement and Poisson blending. 950 of the 3,800 images (25%) are synthetic.
- **Rigorous evaluation:** 5-fold cross-validation, ablation, cross-dataset validation, t-SNE and DBI analysis, and ReliefF + SVM feature validation.

## Results at a glance

| Metric (5-fold average) | Value |
|---|---|
| mAP@0.50 | **0.811** |
| mAP@0.50:0.95 | **0.636** |
| Precision | 0.885 |
| Recall | 0.761 |
| F1-score | 0.738 |
| Inference | 19.1 ms/img (about 52 FPS), 265.8 GFLOPs |

### Ablation: vanilla vs FGA (YOLOv12-Large)

| Parameter | Vanilla | FGA (ours) |
|---|---|---|
| mAP@0.50 | 0.786 | **0.811** |
| mAP@0.50:0.95 | 0.581 | **0.636** |
| Precision | 0.842 | **0.885** |
| Recall | 0.735 | **0.761** |
| F1-score | 0.712 | **0.738** |

The FGA head adds about 0.7 ms per image (+3.8% latency) and 3.4 GFLOPs (+1.3%).

### Comparison with baseline architectures (same combined dataset)

| Model | mAP@0.50:0.95 | mAP@0.50 | Inference (ms/img) | Params (M) | GFLOPs |
|---|---|---|---|---|---|
| Faster R-CNN (ResNet101) | 0.452 | 0.681 | 55.4 | 60.2 | 245.8 |
| YOLOv5l | 0.533 | 0.745 | 11.8 | 46.5 | 109.1 |
| YOLOv8l | 0.565 | 0.768 | 12.5 | 43.7 | 165.2 |
| YOLOv10l | 0.558 | 0.759 | 10.2 | 41.5 | 155.7 |
| YOLOv11l | 0.575 | 0.772 | 16.5 | 62.1 | 240.1 |
| **YOLOv12l** (baseline for FGA) | **0.581** | **0.786** | 18.4 | 73.2 | 262.4 |

### Per-class performance

| Defect class | mAP@0.50:0.95 | mAP@0.50 | Precision | F1 |
|---|---|---|---|---|
| Missing hole | 0.812 | 0.945 | 0.931 | 0.906 |
| Mouse bite | 0.588 | 0.791 | 0.844 | 0.710 |
| Open circuit | 0.551 | 0.755 | 0.915 | 0.674 |
| Short circuit | 0.703 | 0.882 | 0.865 | 0.798 |
| Spur | 0.579 | 0.764 | 0.882 | 0.686 |
| Spurious copper | 0.583 | 0.719 | 0.872 | 0.658 |
| **Overall** | **0.636** | **0.811** | **0.885** | **0.738** |

### 5-fold cross-validation

| Fold | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|---|
| 1 | 0.881 | 0.762 | 0.735 | 0.802 | 0.631 |
| 2 | 0.892 | 0.755 | 0.741 | 0.815 | 0.640 |
| 3 | 0.879 | 0.768 | 0.732 | 0.808 | 0.635 |
| 4 | 0.885 | 0.761 | 0.739 | 0.809 | 0.636 |
| 5 | 0.888 | 0.759 | 0.745 | 0.821 | 0.642 |
| **Average** | **0.885** | **0.761** | **0.738** | **0.811** | **0.636** |

### Cross-dataset validation (mAP@0.50)

The framework was trained from scratch on each public benchmark and compared with the numbers reported by the reference papers.

| Reference | Model and dataset | Reported | Ours |
|---|---|---|---|
| Zhao & Jiang, 2025 | WWBi, HRIPCB | 96.6 | **99.4** |
| Selvam et al., 2025 | DefXpert, HRIPCB | 99.0 | **99.4** |
| Zhou et al., 2025 | DFW, DeepPCB | 95.7 | **97.1** |
| Zheng et al., 2022 | MobileNet-V2, PKU-Market | 99.6 | **99.7** |

---

## Repository structure

> Adjust this section to match your repo. The layout below is a suggestion, so rename folders and files to what you actually pushed.

```
.
├── dataset/                 # Hybrid PCB dataset (YOLO format, see below)
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── synthetic_generation/    # Defect patch library, placement rules, Poisson blending
├── hpo/                     # Bilevel GA hyperparameter optimization (50 trials)
├── training/                # FGA-YOLOv12 definition and 5-fold training script
├── analysis/                # t-SNE, DBI, ReliefF, SVM, error analysis
├── results/                 # Metrics, curves, confusion matrix, figures
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset is a hybrid of the public Kaggle [PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects) dataset, additional publicly sourced PCB images with digitally created defects, and synthetic samples generated with the patch-based pipeline. It was versioned on Roboflow and annotated with bounding boxes.

- **Total images:** 3,800 (950 synthetic, 25%)
- **Annotated instances:** about 8,780
- **Classes (6):** `missing_hole`, `mouse_bite`, `open_circuit`, `short_circuit`, `spur`, `spurious_copper`
- **Input resolution:** 640 x 640

| Class | Images (before aug.) | Instances (before aug.) | Images (after aug.) | Instances (after aug.) |
|---|---|---|---|---|
| Missing hole | 364 | 786 | 779 | 1683 |
| Mouse bite | 272 | 648 | 582 | 1388 |
| Open circuit | 287 | 684 | 615 | 1466 |
| Short circuit | 319 | 670 | 683 | 1435 |
| Spur | 266 | 628 | 569 | 1345 |
| Spurious copper | 267 | 683 | 572 | 1463 |
| **Total** | **3800** | | | **8780** |

Synthetic images and instances after augmentation:

| Class | Synthetic images | Synthetic instances |
|---|---|---|
| Missing hole | 261 | 585 |
| Mouse bite | 86 | 635 |
| Open circuit | 128 | 293 |
| Short circuit | 112 | 229 |
| Spur | 89 | 169 |
| Spurious copper | 274 | 178 |
| **Total** | **950** | **2192** |

**Preprocessing:** auto-orientation, grayscale conversion, adaptive contrast equalization, resize to 640 x 640, pixel scaling to [0, 1] by the Ultralytics pipeline.

**Splits:** 5-fold cross-validation. In each iteration one fold (20%) is a strictly unseen test set. The remaining four folds are split 90% training and 10% internal validation.

### Synthetic defect generation

1. **Defect patch library:** real defects are cropped with polygon masks (not rectangles), each with a binary mask.
2. **Layout-aware placement:** for example, short circuits are placed between closely spaced tracks, and missing copper along traces.
3. **Blending:** Poisson blending and alpha feathering.
4. **Annotation:** all synthetic samples were manually checked and annotated.

Validation against real defects: LPIPS = 0.12, SSIM = 0.83, GLCM texture deviation under 5%, with strong real/synthetic overlap in t-SNE space. Adding synthetic data to the baseline YOLOv12 raised recall from 67.2% to 73.5% and F1 from 69.4% to 71.2%.

---

## Method

### Hierarchical (bilevel) optimization

```
min over λ   L_val( W*(λ), λ )
s.t.  W*(λ) = argmin over W  [ L_train(W, λ) + R_gate(W) ]
```

- **Outer loop (stability search):** a (μ, λ) evolutionary strategy runs 50 iterations. Each iteration trains a YOLOv12-Medium proxy for up to 100 epochs with early termination of weak trials, and scores it by validation mAP@0.50:0.95. The top 20% of configurations become parents and are mutated with Gaussian noise.
- **Inner loop (feature adaptation):** trains the network weights and FGA gates under the fixed regime λ*.

Best hyperparameters found (λ*):

| Hyperparameter | Search range | Final value |
|---|---|---|
| Initial learning rate (lr0) | 1e-4 to 1e-1 (log-uniform) | 0.0091 |
| Momentum | 0.700 to 0.980 | 0.937 |
| Weight decay | 1e-5 to 1e-3 (log-uniform) | 0.0005 |
| Warmup epochs | 0 to 5 | 3 |
| Mosaic | 0.0 to 1.0 | 1 |
| Mixup | 0.0 to 0.5 | 0.15 |
| HSV hue | 0.0 to 0.1 | 0.015 |
| HSV saturation | 0.0 to 0.9 | 0.7 |
| Rotation (degrees) | 0 to 45 | 0.0 (disabled) |

### Feature-Gated Attention (FGA)

For a feature map `X` of size C x H x W:

1. **Squeeze:** global average pooling gives a channel descriptor `z` (size C).
2. **Excitation:** `α = σ(W2 · δ(W1 · z))`, with reduction ratio r = 16, ReLU `δ` and Sigmoid `σ`.
3. **Scale:** `X̃_c = X_c · α_c` (channel-wise gating).

The FGA module is inserted after the final aggregation layers of the three detection scales (P3 80x80, P4 40x40, P5 20x20), just before the detection head. This gives a spatial-then-channel pipeline: R-ELAN backbone, then Area Attention (spatial), then FGA (channel).

### Definitive training

- Model: YOLOv12-Large, initialized from COCO pre-trained weights
- Up to 200 epochs per fold, early stopping with patience 25
- On-the-fly augmentation (Mosaic, MixUp, HSV, geometric transforms)
- Best checkpoint by validation mAP@0.50:0.95 is saved

---

## Getting started

### Environment

The experiments were run on Kaggle with an NVIDIA T4 x 2 GPU instance.

- Python 3.10
- PyTorch 2.1, CUDA 12.1
- Ultralytics (YOLOv12 implementation)
- Roboflow (dataset management)

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Data

```bash
# Dataset config (edit paths to match your setup)
cat dataset/data.yaml
```

### Hyperparameter optimization

```bash
# TODO: replace with your actual script name and arguments
python hpo/run_ga.py --data dataset/data.yaml --trials 50 --epochs 100 --model yolo12m.pt
```

### Training (5-fold, FGA-YOLOv12-L)

```bash
# TODO: replace with your actual script name and arguments
python training/train_fga_yolov12.py --data dataset/data.yaml --folds 5 --epochs 200 --patience 25
```

### Evaluation and analysis

```bash
# TODO: replace with your actual script names
python analysis/evaluate.py --weights <path/to/best.pt> --data dataset/data.yaml
python analysis/tsne_dbi.py --weights <path/to/best.pt>
python analysis/relieff_svm.py --weights <path/to/best.pt>
```

### Inference

```bash
# TODO: replace with your actual script name
python predict.py --weights <path/to/best.pt> --source <image-or-folder>
```

---

## Additional analyses in the paper

- **Feature disentanglement:** Davies-Bouldin Index drops from 2.5 (early backbone) to 0.76 (Neck-Large). The baseline YOLOv12 scores 1.52 against 0.76 for the FGA-optimized model.
- **ReliefF + SVM:** an SVM trained on only the top 20 ReliefF features (about 4% of 512) reaches 94.2% accuracy, 93.8% precision, 94.5% recall and 94.1% F1.
- **Error analysis:** localization error is the dominant failure mode, and background false positives point to the value of hard-negative mining.
- **Compute cost:** the one-time offline search costs about 42 extra GPU-hours on top of roughly 12 GPU-hours for standard training (about 4.5x).

## Limitations and future work

- Manual annotation by several annotators introduces minor label inconsistency.
- Public PCB datasets are limited in size and diversity.
- Future work will cover human-in-the-loop evaluation with different user groups, including defect visualization and trust thresholds.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{rao2026hierarchical,
  title   = {Hierarchical Optimization Framework for Robust {PCB} Defect Detection via Adaptive {YOLOv12} Architecture},
  author  = {Rao, Aditi P and Aekabote, Neetu and Vaishnavi, G Tanushri and S, Varshini and Praveen, Bellamkonda Likhith and Panigrahy, Parth Sarathi and Patra, Swarna Mayee},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {28754--28774},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3663419}
}
```

## Authors

Aditi P Rao, Neetu Aekabote, G Tanushri Vaishnavi, Varshini S, Bellamkonda Likhith Praveen, Parth Sarathi Panigrahy and Swarna Mayee Patra, RV College of Engineering, Bengaluru, India.

Corresponding authors: Parth Sarathi Panigrahy (parthsarathip@rvce.edu.in) and Swarna Mayee Patra (swarnamp@rvce.edu.in).

## License

The paper is published under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. <!-- TODO: add a LICENSE file for the code and dataset and state it here -->

## Acknowledgements

This work builds on the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO framework and the Kaggle [PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects) dataset. Cross-domain validation used HRIPCB, DeepPCB and PKU-Market.
