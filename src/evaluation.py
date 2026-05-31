"""
evaluation.py
-------------
All post-training analysis in one place:

  tta_predict(model, X)
      Run inference with Test-Time Augmentation (TTA): generates N_TTA
      augmented versions of each image, averages the softmax outputs,
      and returns the argmax class prediction.

  evaluate(model, X_test, y_test_oh)
      Compute accuracy, precision, recall, and F1 (weighted) on the test set
      using TTA predictions.

  plot_training_curves(history, phase)
      Plot loss and accuracy curves from a history dict.

  plot_confusion_matrix(y_true, y_pred)
      Heatmap of the confusion matrix with class names on both axes.

  plot_per_class_f1(y_true, y_pred)
      Bar chart of the F1 score for each individual class.

  show_misclassified(X_test, y_true, y_pred, cls_true, cls_pred)
      Display images that were predicted as cls_pred when the true label
      is cls_true (useful for inspecting the hard class confusions).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

from config import CLASSES, N_TTA, SEED


# ── TTA inference ─────────────────────────────────────────────────────────────

# Augmentation pipeline used at test time.
# Deliberately lighter than the training augmentation to avoid distorting
# images so much that the predictions become unreliable.
_tta_aug = tfk.Sequential([
    tfkl.RandomFlip("horizontal_and_vertical"),
    tfkl.RandomRotation(0.5),
    tfkl.RandomContrast(0.35),
    tfkl.RandomTranslation(0.2, 0.2),
], name="tta_augmentation")


def tta_predict(model: tfk.Model, X: np.ndarray) -> np.ndarray:
    """
    Predict class labels using Test-Time Augmentation (TTA).

    Why TTA?
    --------
    During training the model saw each image in many augmented forms.
    At inference time, if we feed only the original image, the model
    may be sensitive to small variations in orientation or contrast.
    TTA addresses this by:
      1. Predicting on the original image.
      2. Generating N_TTA augmented copies and predicting on each.
      3. Averaging all N_TTA+1 softmax probability vectors.
      4. Returning the argmax of the averaged vector.

    This is equivalent to an ensemble of N_TTA+1 slightly different
    versions of the same model, reducing variance at no extra training cost.

    Parameters
    ----------
    model : trained tfk.Model
    X     : test images, shape (N, H, W, C), values in [0, 255]

    Returns
    -------
    y_pred : np.ndarray, shape (N,), integer class labels
    """
    # Normalise to [0,1] for the augmentation layer, then back to [0,255]
    X_norm = X / 255.0

    # Original prediction — shape (1, N, NUM_CLASSES)
    preds = model.predict(X, verbose=0)[np.newaxis, ...]

    for _ in range(N_TTA):
        X_aug = _tta_aug(X_norm).numpy() * 255.0          # augmented batch
        p     = model.predict(X_aug, verbose=0)[np.newaxis, ...]
        preds = np.concatenate([preds, p], axis=0)        # stack along axis 0

    # Average softmax probabilities across the N_TTA+1 forward passes
    mean_preds = preds.mean(axis=0)                       # shape (N, NUM_CLASSES)
    return np.argmax(mean_preds, axis=1)


# ── Quantitative evaluation ───────────────────────────────────────────────────

def evaluate(model: tfk.Model, X_test: np.ndarray, y_test_oh: np.ndarray) -> dict:
    """
    Run TTA inference and compute standard classification metrics.

    All metrics use the 'weighted' average, which weights each class by its
    frequency in the test set.  This gives a more realistic picture of
    overall performance when classes are imbalanced.

    Parameters
    ----------
    model      : trained tfk.Model
    X_test     : test images, shape (N, H, W, C), values in [0, 255]
    y_test_oh  : one-hot labels, shape (N, NUM_CLASSES)

    Returns
    -------
    metrics : dict with keys accuracy, precision, recall, f1
    """
    y_pred = tta_predict(model, X_test)
    y_true = np.argmax(y_test_oh, axis=1)

    metrics = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall"   : recall_score(y_true, y_pred,    average="weighted", zero_division=0),
        "f1"       : f1_score(y_true, y_pred,         average="weighted", zero_division=0),
    }

    print("\n── Test Set Results (with TTA) ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("─────────────────────────────────────────────────\n")

    return metrics, y_pred, y_true


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, phase: int = 1):
    """
    Plot loss and accuracy curves from a Keras history dict.

    The dashed line is the training metric; the solid line is validation.
    Looking at the gap between the two is the primary diagnostic for
    overfitting: a widening gap means the model is memorising training data.

    Parameters
    ----------
    history : dict returned by model.fit().history
    phase   : integer used only in the plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(history["loss"],     label="Train", color="#E07B39", alpha=0.7, linestyle="--")
    ax1.plot(history["val_loss"], label="Val",   color="#4D61E2", linewidth=1.8)
    ax1.set_title(f"Phase {phase} — Loss (Categorical CE)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history["accuracy"],     label="Train", color="#E07B39", alpha=0.7, linestyle="--")
    ax2.plot(history["val_accuracy"], label="Val",   color="#4D61E2", linewidth=1.8)
    ax2.set_title(f"Phase {phase} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Display a heatmap of the confusion matrix.

    The diagonal shows correct predictions; off-diagonal cells reveal
    which pairs of classes the model confuses most often.
    The most common confusion in this project is between
    Immature Granulocytes (3), Monocyte (5) and Neutrophil (6),
    which are morphologically similar at certain maturation stages.

    Parameters
    ----------
    y_true : 1-D array of true integer labels
    y_pred : 1-D array of predicted integer labels
    """
    labels = list(CLASSES.values())
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Test Set (with TTA)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Bar chart of the F1 score for each individual class.

    Per-class F1 is more informative than the weighted average when
    diagnosing specific failure modes: a class with F1 < 0.90 needs
    investigation (more data, targeted augmentation, or loss weighting).

    Parameters
    ----------
    y_true : 1-D array of true integer labels
    y_pred : 1-D array of predicted integer labels
    """
    labels      = list(CLASSES.values())
    per_class   = f1_score(y_true, y_pred, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, per_class, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 — Test Set")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


def show_misclassified(X_test: np.ndarray,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       cls_true: int,
                       cls_pred: int,
                       max_show: int = 10):
    """
    Display images where the true label is cls_true but the model predicted
    cls_pred.

    This is a targeted diagnostic tool for the hard-class confusions.
    For example, calling show_misclassified(..., cls_true=3, cls_pred=6)
    shows Immature Granulocytes that were mistaken for Neutrophils.
    Inspecting these images often reveals that the model struggles with
    cells at specific maturation stages where the two types look similar.

    Parameters
    ----------
    X_test   : test images, shape (N, H, W, C)
    y_true   : true integer labels, shape (N,)
    y_pred   : predicted integer labels, shape (N,)
    cls_true : integer label of the ground-truth class
    cls_pred : integer label of the predicted (wrong) class
    max_show : maximum number of images to display
    """
    idxs = [i for i in range(len(y_true))
             if y_true[i] == cls_true and y_pred[i] == cls_pred]

    if not idxs:
        print(f"No misclassifications found: true={CLASSES[cls_true]}, pred={CLASSES[cls_pred]}")
        return

    show = idxs[:max_show]
    fig, axes = plt.subplots(1, len(show), figsize=(2 * len(show), 3))
    if len(show) == 1:
        axes = [axes]

    for ax, idx in zip(axes, show):
        ax.imshow(np.clip(X_test[idx], 0, 255).astype(np.uint8))
        ax.set_title(f"True: {CLASSES[cls_true]}\nPred: {CLASSES[cls_pred]}", fontsize=7)
        ax.axis("off")

    plt.suptitle(f"Misclassified: {CLASSES[cls_true]} → {CLASSES[cls_pred]} "
                 f"({len(idxs)} total)", fontsize=10)
    plt.tight_layout()
    plt.show()
