"""
data_loader.py
--------------
Handles everything related to loading and preparing the raw dataset:

  1. load_dataset()      – reads the .npz file, returns X and y arrays
  2. remove_outliers()   – drops two non-medical images accidentally included
  3. split_dataset()     – stratified train / val / test split
  4. plot_samples()      – visualise a random grid of images with their labels
  5. plot_class_dist()   – bar chart of class frequencies

No augmentation or balancing is performed here; those steps live in
dataset_balancer.py so each responsibility is isolated.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from config import CLASSES, DATA_PATH, TEST_SIZE, VAL_SIZE, SEED
from sklearn.model_selection import train_test_split


# ── 1. Loading ────────────────────────────────────────────────────────────────

def load_dataset(path: str = DATA_PATH):
    """
    Load images and labels from a .npz archive.

    The archive is expected to have two arrays:
      - 'images': uint8 array of shape (N, H, W, C)
      - 'labels': integer array of shape (N, 1)

    Returns
    -------
    X : np.ndarray, shape (N, 96, 96, 3)  – raw pixel values in [0, 255]
    y : np.ndarray, shape (N, 1)           – integer class labels
    """
    data = np.load(path)
    X = data["images"]
    y = data["labels"]
    print(f"Loaded dataset  →  X: {X.shape}  y: {y.shape}")
    return X, y


# ── 2. Outlier removal ────────────────────────────────────────────────────────

def remove_outliers(X: np.ndarray, y: np.ndarray):
    """
    Remove the two non-medical images that were accidentally left in the
    dataset during data collection.

    The outliers are a frame from the movie Shrek (index 13 000) and a frame
    from the Rick & Morty series (index 13 753).  Both are identified by
    exact pixel-level matching so the function is robust even if the dataset
    is shuffled beforehand.

    Parameters
    ----------
    X, y : raw arrays returned by load_dataset()

    Returns
    -------
    X_clean, y_clean : arrays with outliers removed
    """
    # Store the pixel content of the two known outlier images
    outlier_a = X[13_000].copy()   # Shrek frame
    outlier_b = X[13_753].copy()   # Rick & Morty frame

    # Identify every index that matches either outlier (exact comparison)
    to_remove = [
        i for i, img in enumerate(X)
        if np.array_equal(img, outlier_a) or np.array_equal(img, outlier_b)
    ]

    X_clean = np.delete(X, to_remove, axis=0)
    y_clean = np.delete(y, to_remove, axis=0)

    print(f"Removed {len(to_remove)} outlier images  →  {X_clean.shape[0]} remaining")
    return X_clean, y_clean


# ── 3. Dataset split ──────────────────────────────────────────────────────────

def split_dataset(X: np.ndarray, y: np.ndarray,
                  test_size: int = TEST_SIZE,
                  val_size: int  = VAL_SIZE,
                  seed: int      = SEED):
    """
    Perform a two-step stratified split:

      Step 1  →  carve out a fixed-size test set (never touched during training)
      Step 2  →  split the remainder into train and validation

    Stratification ensures that the class imbalance in the full dataset is
    preserved proportionally in every split.

    Parameters
    ----------
    X, y        : cleaned arrays
    test_size   : number of samples reserved for the test set
    val_size    : number of samples reserved for validation
    seed        : random state for reproducibility

    Returns
    -------
    X_train, X_val, X_test : image arrays
    y_train, y_val, y_test : label arrays (still integer, not one-hot)
    """
    # Step 1: isolate the test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Step 2: split the remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_size,
        random_state=seed,
        stratify=y_tv,
    )

    print(f"Split  →  train: {len(X_train):,}  |  val: {len(X_val):,}  |  test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 4. Visualisation helpers ──────────────────────────────────────────────────

def plot_samples(X: np.ndarray, y: np.ndarray, n: int = 10, title: str = "Random samples"):
    """
    Display n randomly selected images in a single row with their class label
    as the subplot title.

    Parameters
    ----------
    X     : image array, values in [0, 255]
    y     : integer label array, shape (N,) or (N, 1)
    n     : number of images to show
    title : figure super-title
    """
    y_flat = y.flatten()
    idxs   = random.sample(range(len(X)), n)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    for ax, idx in zip(axes, idxs):
        ax.imshow(np.clip(X[idx], 0, 255).astype(np.uint8))
        ax.set_title(CLASSES[int(y_flat[idx])], fontsize=7)
        ax.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y: np.ndarray, title: str = "Class distribution"):
    """
    Bar chart showing how many samples belong to each class.

    Useful for visualising the imbalance before and after oversampling.

    Parameters
    ----------
    y     : integer label array (any shape; will be flattened)
    title : plot title
    """
    labels = list(CLASSES.values())
    unique, counts = np.unique(y.flatten(), return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, counts, color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()
