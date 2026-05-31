"""
dataset_balancer.py
-------------------
Addresses the class imbalance present in the training set through a
two-stage strategy:

  Stage 1 – Offline oversampling
      Every class is brought to TARGET_SAMPLES_PER_CLASS examples by
      randomly resampling existing images and applying a strong
      augmentation pipeline (translation, rotation, sharpening, flip).

  Stage 2 – CutMix for the hard classes
      The three most morphologically similar classes (Immature Granulocytes,
      Monocyte, Neutrophil — labels 3, 5, 6) receive an additional
      CutMix augmentation pass.  CutMix cuts a rectangular patch from one
      image and pastes it onto another, mixing the one-hot labels in
      proportion to the pasted area.  This forces the model to extract
      discriminative features from the entire image rather than a single
      salient region.

  Stage 3 – Online augmentation via tf.data
      During training, every batch is further perturbed on-the-fly using
      RandAugment + flip + rotation to maximise the effective dataset size.

Public API
----------
  balance_training_set(X_train, y_train)  →  X_bal, y_bal (one-hot)
  augment_validation_hard_classes(...)    →  X_val, y_val (one-hot)
  make_train_dataset(X, y_oh)             →  tf.data.Dataset
  make_val_dataset(X, y_oh)              →  tf.data.Dataset
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import keras_cv as kcv
from sklearn.utils import shuffle as sk_shuffle

from config import (
    SEED, NUM_CLASSES, TARGET_SAMPLES_PER_CLASS,
    HARD_CLASSES, CUTMIX_SAMPLES_PER_HARD_CLASS,
    VAL_AUGMENT_HARD_CLASSES, BATCH_SIZE,
)


# ── Shared augmentation layers (instantiated once, reused everywhere) ─────────

# Used during offline oversampling: aggressive spatial distortions.
_offline_aug = tf.keras.Sequential([
    tfkl.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
    tfkl.RandomRotation(0.8),
    kcv.layers.RandomSharpness(factor=1.0, value_range=(0, 1)),
    tfkl.RandomFlip(mode="horizontal_and_vertical"),
], name="offline_augmentation")

# Used online inside the tf.data pipeline: lighter but diverse.
_rand_augment = kcv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=2,
    magnitude=0.4,
    magnitude_stddev=0.15,
    rate=0.8,
    geometric=True,
    seed=SEED,
)
_flip_layer     = tfkl.RandomFlip()
_rotation_layer = tfkl.RandomRotation(factor=0.5)

# CutMix layer for the hard-class extra augmentation pass.
_cutmix_layer = kcv.layers.CutMix(alpha=1.0, seed=SEED)


# ── Stage 1: Offline oversampling ─────────────────────────────────────────────

def _oversample_class(X_train, y_train, cls, n_missing):
    """
    Generate n_missing new examples for class `cls` by:
      1. Randomly resampling (with replacement) from existing class examples.
      2. Applying the offline augmentation pipeline to each selected image.

    Pixel values are clipped back to [0, 255] after augmentation because
    some transforms (e.g. sharpening) can push values slightly out of range.

    Parameters
    ----------
    X_train  : image array, values in [0, 255]
    y_train  : integer label array, shape (N, 1)
    cls      : integer class index to oversample
    n_missing: number of new samples to generate

    Returns
    -------
    X_new : np.ndarray of shape (n_missing, H, W, C)
    y_new : np.ndarray of shape (n_missing, 1)
    """
    class_idxs = np.where(y_train.flatten() == cls)[0]
    selected   = np.array(random.choices(X_train[class_idxs], k=n_missing))

    # Normalise to [0,1] for the augmentation layer, then rescale back
    augmented = np.clip(_offline_aug(selected / 255.0) * 255.0, 0, 255)

    labels = np.full((n_missing, 1), cls, dtype=np.int64)
    return augmented, labels


def balance_training_set(X_train: np.ndarray, y_train: np.ndarray):
    """
    Run Stage 1 (oversampling) and Stage 2 (CutMix) on the training set.

    After this function the training set is balanced: every class has
    TARGET_SAMPLES_PER_CLASS + CUTMIX_SAMPLES_PER_HARD_CLASS (for hard classes)
    examples.

    Labels are returned in one-hot format (float32) because CutMix produces
    soft (fractional) labels that cannot be represented as integers.

    Parameters
    ----------
    X_train : shape (N, 96, 96, 3), values in [0, 255]
    y_train : shape (N, 1), integer labels

    Returns
    -------
    X_bal   : shape (N_bal, 96, 96, 3)
    y_bal   : shape (N_bal, NUM_CLASSES), float32 one-hot (possibly soft)
    """
    # ── Stage 1: bring every class to TARGET_SAMPLES_PER_CLASS ───────────────
    _, count_train = np.unique(y_train.flatten(), return_counts=True)

    for cls in range(NUM_CLASSES):
        n_missing = TARGET_SAMPLES_PER_CLASS - count_train[cls]
        if n_missing <= 0:
            # Class already has enough samples; nothing to do
            continue
        X_new, y_new = _oversample_class(X_train, y_train, cls, n_missing)
        X_train = np.concatenate([X_train, X_new], axis=0)
        y_train = np.concatenate([y_train, y_new], axis=0)

    X_train, y_train = sk_shuffle(X_train, y_train, random_state=SEED)

    # Convert integer labels to one-hot before CutMix (which needs soft labels)
    y_oh = tfk.utils.to_categorical(y_train, num_classes=NUM_CLASSES).astype(np.float32)

    # ── Stage 2: CutMix for hard classes ─────────────────────────────────────
    # Hard classes are the most often confused because their cell morphologies
    # overlap (e.g. Immature Granulocytes vs Neutrophil vs Monocyte).
    # CutMix diversifies their representations without changing the class ratio.
    for cls in HARD_CLASSES:
        class_idxs = np.where(y_oh[:, cls] == 1.0)[0].tolist()

        # Sample without replacement up to the requested number
        k = min(CUTMIX_SAMPLES_PER_HARD_CLASS, len(class_idxs))
        chosen = random.sample(class_idxs, k=k)

        X_sel = tf.convert_to_tensor(X_train[chosen], dtype=tf.float32)
        y_sel = tf.convert_to_tensor(y_oh[chosen],    dtype=tf.float32)

        # CutMix expects and returns a dict with keys 'images' and 'labels'
        output  = _cutmix_layer({"images": X_sel, "labels": y_sel})
        X_augm  = output["images"].numpy()
        y_augm  = output["labels"].numpy()

        X_train = np.concatenate([X_train, X_augm], axis=0)
        y_oh    = np.concatenate([y_oh,    y_augm], axis=0)

    # Final shuffle to mix oversampled and CutMix samples uniformly
    X_bal, y_bal = sk_shuffle(X_train, y_oh, random_state=SEED)

    print(f"Balanced training set  →  {X_bal.shape[0]:,} samples")
    return X_bal, y_bal


# ── Validation augmentation for hard classes ──────────────────────────────────

def augment_validation_hard_classes(X_val: np.ndarray, y_val: np.ndarray):
    """
    Add VAL_AUGMENT_HARD_CLASSES augmented copies per hard class to the
    validation set.

    Motivation: hard classes are underrepresented in the validation set,
    which makes val_accuracy estimates for those classes noisy and
    unreliable for early stopping.  Adding augmented examples stabilises
    the estimate without introducing data leakage (validation images are
    augmented differently from training images).

    Parameters
    ----------
    X_val, y_val : integer-labelled validation arrays

    Returns
    -------
    X_val_aug, y_val_oh : augmented val set with one-hot labels (float32)
    """
    for cls in HARD_CLASSES:
        class_idxs = np.where(y_val.flatten() == cls)[0]
        selected   = np.array(random.choices(X_val[class_idxs],
                                              k=VAL_AUGMENT_HARD_CLASSES))
        augmented  = np.clip(_offline_aug(selected / 255.0) * 255.0, 0, 255)
        new_labels = np.full((VAL_AUGMENT_HARD_CLASSES, 1), cls, dtype=np.int64)
        X_val = np.concatenate([X_val, augmented],  axis=0)
        y_val = np.concatenate([y_val, new_labels], axis=0)

    X_val, y_val = sk_shuffle(X_val, y_val, random_state=SEED)
    y_val_oh = tfk.utils.to_categorical(y_val, num_classes=NUM_CLASSES).astype(np.float32)

    print(f"Augmented val set  →  {X_val.shape[0]:,} samples")
    return X_val, y_val_oh


# ── Stage 3: tf.data pipelines ────────────────────────────────────────────────

def _augment_batch(image, label):
    """
    Online augmentation applied to every training batch inside the tf.data
    pipeline.

    Using tf.data.Dataset.map() ensures augmentation runs in parallel on
    CPU while the GPU is busy with the forward/backward pass, so it does
    not slow down training.

    Transformations applied:
      - RandAugment: randomly selects 2 operations from a pool of ~14
        (rotations, translations, colour jitter, sharpness, etc.)
      - RandomFlip: horizontal and/or vertical
      - RandomRotation: up to 180 degrees

    Parameters
    ----------
    image : tf.Tensor of shape (batch, H, W, C), float32
    label : tf.Tensor of shape (batch, NUM_CLASSES), float32

    Returns
    -------
    augmented image and unchanged label
    """
    image = _rand_augment(image)
    image = _flip_layer(image)
    image = _rotation_layer(image)
    return image, label


def make_train_dataset(X: np.ndarray, y_oh: np.ndarray) -> tf.data.Dataset:
    """
    Build the tf.data pipeline for training.

    Order of operations:
      1. Create dataset from in-memory arrays
      2. Shuffle with a buffer large enough to randomise across classes
      3. Batch (drop_remainder=False to keep all samples)
      4. Apply online augmentation to each batch
      5. Prefetch: start preparing the next batch while the GPU trains on
         the current one (AUTOTUNE lets TF choose the optimal buffer size)

    Parameters
    ----------
    X    : balanced image array, shape (N, H, W, C), values in [0, 255]
    y_oh : one-hot label array, shape (N, NUM_CLASSES), float32

    Returns
    -------
    tf.data.Dataset ready for model.fit()
    """
    ds = (
        tf.data.Dataset.from_tensor_slices((X, y_oh))
        .shuffle(buffer_size=1_000, seed=SEED)
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


def make_val_dataset(X: np.ndarray, y_oh: np.ndarray) -> tf.data.Dataset:
    """
    Build the tf.data pipeline for validation.

    No shuffling or augmentation is applied: the validation set must be
    deterministic so that val_accuracy comparisons across epochs are meaningful.

    Parameters
    ----------
    X    : validation image array, shape (N, H, W, C)
    y_oh : one-hot label array, shape (N, NUM_CLASSES), float32

    Returns
    -------
    tf.data.Dataset ready for model.fit(validation_data=...)
    """
    ds = (
        tf.data.Dataset.from_tensor_slices((X, y_oh))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds
