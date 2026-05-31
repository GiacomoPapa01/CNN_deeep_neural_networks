"""
train.py
--------
Entry point for the full training pipeline.  Run this script to go from
raw data to a saved model file:

    python train.py

What this script does (in order)
---------------------------------
  1. Set up reproducibility (seeds, TF verbosity)
  2. Load and clean the dataset
  3. Split into train / val / test
  4. Balance the training set (oversampling + CutMix)
  5. Build tf.data pipelines
  6. Phase 1 — train only the classification head (backbone frozen)
  7. Phase 2 — selectively fine-tune the deep backbone layers
  8. Evaluate on the test set with TTA
  9. Save the final model

All hyperparameters are defined in config.py; nothing is hardcoded here.
"""

import os
import random
import logging
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk

# ── Seed everything before any TF/Keras objects are created ──────────────────
from config import (
    SEED, EPOCHS_PHASE1, EPOCHS_PHASE2,
    MODEL_PHASE1_PATH, MODEL_FINAL_PATH,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── Project modules ───────────────────────────────────────────────────────────
from data_loader      import load_dataset, remove_outliers, split_dataset
from dataset_balancer import (balance_training_set, augment_validation_hard_classes,
                               make_train_dataset, make_val_dataset)
from model            import build_backbone, build_model, unfreeze_for_fine_tuning, get_callbacks
from evaluation       import evaluate, plot_training_curves, plot_confusion_matrix, plot_per_class_f1


def main():
    print("=" * 60)
    print("  Blood Cell Classification — Training Pipeline")
    print("=" * 60)
    print(f"\nTensorFlow : {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}\n")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("── Step 1: Loading data ────────────────────────────────────")
    X, y = load_dataset()

    # ── Step 2: Remove outliers ───────────────────────────────────────────────
    print("\n── Step 2: Removing outliers ───────────────────────────────")
    X, y = remove_outliers(X, y)

    # ── Step 3: Split ─────────────────────────────────────────────────────────
    print("\n── Step 3: Splitting dataset ───────────────────────────────")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # ── Step 4: Balance training set ─────────────────────────────────────────
    print("\n── Step 4: Balancing training set ─────────────────────────")
    X_train_bal, y_train_oh = balance_training_set(X_train, y_train)
    X_val_aug,   y_val_oh   = augment_validation_hard_classes(X_val, y_val)

    # One-hot encode the test labels (no augmentation applied to test set)
    y_test_oh = tfk.utils.to_categorical(y_test, num_classes=8).astype("float32")

    # ── Step 5: tf.data pipelines ─────────────────────────────────────────────
    print("\n── Step 5: Building tf.data pipelines ─────────────────────")
    train_ds = make_train_dataset(X_train_bal, y_train_oh)
    val_ds   = make_val_dataset(X_val_aug,   y_val_oh)

    # ── Step 6: Phase 1 — head training ──────────────────────────────────────
    print("\n── Step 6: Phase 1 — Training classification head ─────────")
    backbone = build_backbone()
    model    = build_model(backbone)

    history1 = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE1,
        validation_data=val_ds,
        callbacks=get_callbacks(phase=1),
        verbose=1,
    ).history

    best_val1 = round(max(history1["val_accuracy"]) * 100, 2)
    print(f"\nPhase 1 best val accuracy: {best_val1}%")
    model.save(MODEL_PHASE1_PATH)
    print(f"Phase 1 model saved to: {MODEL_PHASE1_PATH}")

    plot_training_curves(history1, phase=1)

    # ── Step 7: Phase 2 — fine-tuning ─────────────────────────────────────────
    print("\n── Step 7: Phase 2 — Selective fine-tuning ────────────────")

    # Reload the best phase-1 weights before unfreezing
    model = tfk.models.load_model(MODEL_PHASE1_PATH)
    model = unfreeze_for_fine_tuning(model)

    history2 = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE2,
        validation_data=val_ds,
        callbacks=get_callbacks(phase=2),
        verbose=1,
    ).history

    best_val2 = round(max(history2["val_accuracy"]) * 100, 2)
    final_model_path = MODEL_FINAL_PATH.format(acc=best_val2)
    model.save(final_model_path)
    print(f"\nPhase 2 best val accuracy: {best_val2}%")
    print(f"Final model saved to: {final_model_path}")

    plot_training_curves(history2, phase=2)

    # ── Step 8: Evaluate on test set ──────────────────────────────────────────
    print("\n── Step 8: Evaluation on test set ─────────────────────────")
    metrics, y_pred, y_true = evaluate(model, X_test, y_test_oh)

    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_f1(y_true, y_pred)

    print("\nPipeline complete.")
    return model, metrics


if __name__ == "__main__":
    main()
