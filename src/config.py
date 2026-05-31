"""
config.py
---------
Central configuration file for the Blood Cell Classification project.
All hyperparameters, paths, and constants are defined here so that
changing a value in one place propagates across the entire codebase.
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Dataset ───────────────────────────────────────────────────────────────────

# Path to the .npz file containing 'images' and 'labels' arrays.
# Adjust this to wherever training_set.npz lives on your machine / Drive.
DATA_PATH = "data/training_set.npz"

# The 8 blood-cell categories the model must distinguish.
# Key = integer label stored in the .npz file.
# Value = human-readable name used in plots and reports.
CLASSES = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature Granulocytes",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet",
}
NUM_CLASSES = len(CLASSES)

# ── Dataset split sizes ───────────────────────────────────────────────────────

# Number of samples held out as the final, untouched test set.
TEST_SIZE = 1_000

# Number of samples held out from the remaining data for validation.
# Validation drives early stopping and learning-rate scheduling.
VAL_SIZE = 2_000

# ── Class balancing ───────────────────────────────────────────────────────────

# Every class in the training set is oversampled (with augmentation) up to
# this number of examples before the training loop begins.
TARGET_SAMPLES_PER_CLASS = 2_500

# Classes that receive an additional CutMix augmentation pass on top of the
# standard oversampling.  These were identified as the most morphologically
# similar and therefore the most often confused at inference time.
HARD_CLASSES = [3, 5, 6]

# Number of CutMix samples generated per hard class.
CUTMIX_SAMPLES_PER_HARD_CLASS = 1_500

# Number of augmented samples added per hard class to the *validation* set
# so that val_accuracy estimates for these rare classes are more stable.
VAL_AUGMENT_HARD_CLASSES = 300

# ── Training hyperparameters ──────────────────────────────────────────────────

BATCH_SIZE = 32
EPOCHS_PHASE1 = 300   # Maximum epochs for phase 1 (head training, backbone frozen)
EPOCHS_PHASE2 = 100   # Maximum epochs for phase 2 (selective fine-tuning)

# Phase 1 uses Lion, which is more aggressive than Adam, so the effective
# learning rate is divided by 10 relative to a standard Adam run.
LR_PHASE1 = 1e-3      # Passed to Lion as lr/10 internally → effective 1e-4

# Phase 2 uses Adam with a very small learning rate to avoid overwriting the
# pretrained representations in the backbone.
LR_PHASE2 = 5e-5

# L2 regularisation coefficient applied to the final Dense layer.
L2_LAMBDA = 5e-3

# ── Early stopping & LR scheduling ───────────────────────────────────────────

# Number of epochs without improvement before training stops.
ES_PATIENCE = 10

# Number of epochs without improvement before the learning rate is reduced.
LR_PATIENCE = 7

# Multiplicative factor applied to the learning rate on plateau.
LR_FACTOR = 0.1

# Minimum relative improvement considered as "progress".
LR_MIN_DELTA = 0.0015

# Absolute floor for the learning rate (scheduler will never go below this).
LR_MIN = 1e-5

# ── Model / backbone ─────────────────────────────────────────────────────────

# Input resolution expected by MobileNetV3Large.
# The dataset images are already 96×96×3, so no resizing is needed.
IMAGE_SHAPE = (96, 96, 3)

# Number of backbone layers that remain frozen during fine-tuning.
# Layers 0-123 capture low-level features (edges, textures) that are
# universal and do not need domain adaptation.
N_FROZEN_LAYERS = 124

# Size of the intermediate Dense layer inserted between the backbone output
# and the softmax classifier.
DENSE_UNITS = 128

# ── Test-Time Augmentation (TTA) ──────────────────────────────────────────────

# Number of augmented copies generated per test image.
# The final prediction is the mean of the N_TTA+1 softmax outputs
# (one original + N_TTA augmented).
N_TTA = 5

# ── Output paths ─────────────────────────────────────────────────────────────

MODEL_PHASE1_PATH = "outputs/phase1_model.keras"
MODEL_FINAL_PATH  = "outputs/BloodCell_MobileNetV3L_{acc}.keras"  # {acc} filled at runtime
