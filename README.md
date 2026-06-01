# Blood Cell Classification MobileNetV3Large + Transfer Learning

**8-class image classification of blood cell types from microscopy images.**  
Final result: **98.5% accuracy / F1** on the test set **top 8 out of 300+** participants.

*Developed as part of the Artificial Neural Networks and Deep Learning course, Politecnico di Milano.*

---

## Problem

Given ~23,000 RGB microscopy images (96×96 px), classify each image into one of 8 blood cell types:

| Label | Cell type |
|-------|-----------|
| 0 | Basophil |
| 1 | Eosinophil |
| 2 | Erythroblast |
| 3 | Immature Granulocytes |
| 4 | Lymphocyte |
| 5 | Monocyte |
| 6 | Neutrophil |
| 7 | Platelet |

Key challenges: **class imbalance** (some types are 10× rarer than others) and **morphological similarity** between classes 3, 5, and 6, which share overlapping visual features at certain maturation stages.

---

## Project Structure

```
blood_cell_classification/
│
├── train.py              # Main entry point — runs the full pipeline end-to-end
├── inference.py          # Model class used for competition submission (with TTA)
│
├── src/
│   ├── config.py         # All hyperparameters and constants in one place
│   ├── data_loader.py    # Load, clean, split, and visualise the dataset
│   ├── dataset_balancer.py  # Oversampling, CutMix, tf.data pipelines
│   ├── model.py          # Architecture definition and two-phase training logic
│   └── evaluation.py     # TTA inference, metrics, confusion matrix, plots
│
├── data/
│   └── training_set.npz  # Raw dataset (not included in the repo)
│
└── outputs/              # Saved models and training artefacts
```

---

## Architecture

```
Input (96 × 96 × 3)
       │
       ▼
Online Augmentation  [RandomFlip, RandomTranslation, RandomRotation]
(disabled automatically at inference time)
       │
       ▼
MobileNetV3Large  ── pretrained on ImageNet, Global Max Pooling
       │
       ▼
Dense(128) → ReLU
       │
       ▼
Dense(8, softmax) + L2 regularisation
       │
       ▼
Class probabilities (8 types)
```

**Why MobileNetV3Large?**  
It offers the best accuracy / parameter trade-off for 96×96 images. Depthwise Separable Convolutions reduce parameters ~9× vs standard Conv2D. The Squeeze-and-Excitation modules learn to weight feature maps by their relevance — particularly effective for fine-grained cell differentiation.

---

## Training Strategy

### Phase 1 — Feature Extraction (backbone frozen)

The MobileNetV3Large weights are locked. Only the Dense head is trained.

- **Optimizer:** Lion (lr = 1e-4 effective)  
  Lion applies only the *sign* of the gradient, making updates constant in scale and faster than Adam for small-head training.
- **Loss:** Categorical Crossentropy
- **Early stopping:** patience = 10, monitors `val_accuracy`, restores best weights

### Phase 2 — Selective Fine-Tuning

The first 124 backbone layers remain frozen (low-level universal features).  
Conv2D and DepthwiseConv2D layers in the deeper part are unfrozen.  
BatchNormalization layers remain frozen throughout (running statistics from ImageNet are preserved to avoid destabilising normalisation).

- **Optimizer:** Adam (lr = 5e-5) — very low rate to avoid overwriting ImageNet representations
- **Early stopping:** patience = 11

---

## Class Balancing

The training set is imbalanced. Two strategies are applied before training:

**1. Oversampling with augmentation**  
Every class is brought to 2,500 samples by resampling existing images and applying random translation, rotation, sharpening, and flipping.

**2. CutMix for the hard classes (3, 5, 6)**  
CutMix cuts a rectangular patch from one image and pastes it onto another, mixing labels in proportion to the pasted area. This forces the model to look at the full image rather than a single localised pattern — critical for distinguishing morphologically similar cell types.

---

## Test-Time Augmentation (TTA)

At inference time, 5 augmented copies of each image are generated.  
The softmax outputs from the original + 5 copies are averaged before taking the argmax.  
This reduces prediction variance at no extra training cost.

```
Original image → P0
Augmented copy 1 → P1
...
Augmented copy 5 → P5

Final prediction = argmax( mean([P0, P1, ..., P5]) )
```

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 98.5% |
| Precision (weighted) | 98.5% |
| Recall (weighted) | 98.5% |
| F1 Score (weighted) | 98.5% |
| Competition ranking | **Top 8 / 300+** |

The most confused pair was Immature Granulocytes ↔ Neutrophil, which are biologically related and share nucleus shape features at certain maturation stages.

---

## How to Run

**1. Install dependencies**
```bash
pip install tensorflow keras-cv scikit-learn matplotlib seaborn
```

**2. Place the dataset**
```
data/training_set.npz
```

**3. Train**
```bash
cd blood_cell_classification
python train.py
```

The script runs the full pipeline and saves the final model to `outputs/`.

**4. Inference**
```python
from inference import Model

model = Model("outputs/BloodCell_MobileNetV3L_98.5.keras")
labels = model.predict(X_test)  # X_test: (N, 96, 96, 3), uint8
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Transfer learning from ImageNet | Dataset too small (~23k) to train from scratch without severe overfitting |
| Global Max Pooling | Preserves the strongest activation per feature map; better than Average Pooling when localised patterns are discriminative |
| Lion optimizer (Phase 1) | Faster convergence for small-head training; sign-based updates are more stable when backbone is frozen |
| Adam with lr=5e-5 (Phase 2) | Low enough to adapt high-level features without destroying ImageNet representations |
| CutMix only for classes 3, 5, 6 | These classes are morphologically similar; standard augmentation is insufficient |
| BatchNorm frozen during fine-tuning | Running statistics from ImageNet pre-training are more reliable than re-estimated stats on a small domain-specific dataset |
| TTA at inference | Free variance reduction; equivalent to a lightweight ensemble |
