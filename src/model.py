"""
model.py
--------
Defines the full classification model and the two-phase training strategy.

Architecture overview
---------------------
  Input (96×96×3)
      │
      ▼
  Online augmentation  [RandomFlip, RandomTranslation, RandomRotation]
  (active only during training; bypassed at inference)
      │
      ▼
  MobileNetV3Large backbone  (pre-trained on ImageNet, output = 1280-dim vector)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Depthwise Separable Convolutions + Inverted Residual Bottlenecks       │
  │  + Squeeze-and-Excitation (SE) modules + Hard-Swish activations         │
  │  → learned with ImageNet weights; captures universal visual features    │
  └─────────────────────────────────────────────────────────────────────────┘
      │  Global Max Pooling  (preserves the most activated feature per map)
      ▼
  Dense(128) → ReLU
      │
      ▼
  Dense(8, softmax) + L2 regularisation
      │
      ▼
  Predicted class probabilities (8 blood-cell types)

Two-phase training
------------------
  Phase 1 – Feature extraction
      The backbone is frozen (trainable=False).  Only the two Dense layers
      at the top are updated.  Optimizer: Lion (lr/10).
      Rationale: Lion is more aggressive than Adam; using it on a small head
      while keeping the backbone frozen converges quickly without risk.

  Phase 2 – Selective fine-tuning
      The backbone is partially unfrozen: only Conv2D and DepthwiseConv2D
      layers beyond layer N_FROZEN_LAYERS are set trainable.
      BatchNormalization layers remain frozen (using their running statistics
      from ImageNet pre-training) to avoid destabilising normalisation.
      Optimizer: Adam (lr=5e-5) — small step size to preserve learned
      representations while adapting high-level features to the medical domain.

Public API
----------
  build_backbone()                    → frozen MobileNetV3Large
  build_model(backbone)               → compiled Keras model (phase 1)
  unfreeze_for_fine_tuning(model)     → model ready for phase 2
  get_callbacks(phase)                → list of Keras callbacks
"""

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

from config import (
    SEED, IMAGE_SHAPE, NUM_CLASSES, DENSE_UNITS, L2_LAMBDA,
    LR_PHASE1, LR_PHASE2, N_FROZEN_LAYERS,
    ES_PATIENCE, LR_PATIENCE, LR_FACTOR, LR_MIN_DELTA, LR_MIN,
    MODEL_PHASE1_PATH,
)


# ── Backbone ──────────────────────────────────────────────────────────────────

def build_backbone() -> tfk.Model:
    """
    Load MobileNetV3Large pre-trained on ImageNet and freeze all layers.

    Key configuration choices
    -------------------------
    include_top=False
        We discard the original 1000-class ImageNet head and attach our own
        8-class head on top.

    pooling='max'
        Global Max Pooling is applied after the last convolutional block,
        reducing the 3-D feature tensor to a 1-D vector (1280 dimensions).
        Max pooling preserves the strongest activation per feature map, which
        works well for tasks where a single salient pattern (e.g. cell nucleus
        shape) is the key discriminator.

    dropout_rate=0.01
        A minimal internal dropout is kept for light regularisation.

    weights='imagenet'
        The model starts from representations learned on ~1.2 M natural images.
        Although blood-cell images look different from everyday photographs,
        low-level features (edges, colour gradients, textures) transfer well
        to any image domain.

    Returns
    -------
    backbone : frozen Keras model
    """
    backbone = tfk.applications.MobileNetV3Large(
        input_shape=IMAGE_SHAPE,
        dropout_rate=0.01,
        include_top=False,
        weights="imagenet",
        pooling="max",
    )
    # Freeze the entire backbone so that only the new head is trained in phase 1
    backbone.trainable = False
    return backbone


# ── Full model ────────────────────────────────────────────────────────────────

def build_model(backbone: tfk.Model) -> tfk.Model:
    """
    Attach a custom classification head on top of the frozen backbone and
    compile the model for phase 1 training.

    Head architecture
    -----------------
      Dense(128) → ReLU → Dense(8, softmax) + L2

    L2 regularisation is applied only to the final layer's weights.
    This penalises large weights, reducing overfitting on the small head
    while the much larger backbone remains frozen.

    Optimizer: Lion
    ---------------
    Lion (EvoLved Sign Momentum, Chen et al. 2023) was derived through
    automated program search.  Unlike Adam, it applies only the *sign* of
    the gradient (not its magnitude), making updates constant in scale.
    It is memory-efficient and converges faster on small fine-tuning tasks.
    The learning rate is set to LR_PHASE1/10 because Lion's constant-size
    steps are more aggressive than Adam's and require a lower base rate.

    Parameters
    ----------
    backbone : frozen MobileNetV3Large returned by build_backbone()

    Returns
    -------
    model : compiled tfk.Model
    """
    tf.random.set_seed(SEED)
    regulariser = tfk.regularizers.l2(L2_LAMBDA)

    inputs = tfk.Input(IMAGE_SHAPE, name="input")

    # Lightweight online augmentation applied only during training.
    # Placing augmentation *inside* the model (rather than in the tf.data
    # pipeline) means it is automatically disabled during model.predict().
    x = tf.keras.Sequential([
        tfkl.RandomFlip("horizontal and vertical"),
        tfkl.RandomTranslation(0.2, 0.2),
        tfkl.RandomRotation(0.5),
    ], name="online_augmentation")(inputs)

    # Feature extraction through the frozen backbone
    x = backbone(x)

    # Classification head
    x       = tfkl.Dense(DENSE_UNITS, name="dense")(x)
    x       = tfkl.ReLU(name="relu")(x)
    outputs = tfkl.Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=regulariser,
        name="output",
    )(x)

    model = tfk.Model(inputs, outputs, name="BloodCell_Classifier")

    # Lion optimizer: more efficient than Adam for small heads on frozen backbones
    optimizer = tfk.optimizers.Lion(LR_PHASE1 / 10, weight_decay=None, use_ema=False)

    model.compile(
        loss=tfk.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


# ── Phase 2: selective fine-tuning ────────────────────────────────────────────

def unfreeze_for_fine_tuning(model: tfk.Model) -> tfk.Model:
    """
    Partially unfreeze the backbone and recompile the model for phase 2.

    Unfreezing strategy
    -------------------
    1. Make the entire backbone trainable.
    2. Re-freeze the first N_FROZEN_LAYERS layers.
       These capture low-level features (edges, gradients, simple textures)
       that are already optimal for any image domain; updating them would
       only introduce instability.
    3. Among the deeper layers, freeze BatchNormalization layers.
       Their running mean/variance statistics were calibrated on ImageNet;
       updating them with a small medical-imaging dataset would corrupt the
       normalisation and degrade performance.
    4. Only Conv2D and DepthwiseConv2D layers in the deeper part of the
       backbone are left trainable.  These capture high-level, domain-specific
       patterns (cell shapes, staining patterns) that benefit from adaptation.

    Optimizer: Adam (lr = LR_PHASE2 = 5e-5)
    ----------------------------------------
    The very low learning rate is critical: large gradient steps would
    overwrite the ImageNet representations still encoded in the backbone,
    erasing the benefit of transfer learning.

    Parameters
    ----------
    model : phase-1 model (loaded from disk with best phase-1 weights)

    Returns
    -------
    model : same model, recompiled with phase-2 settings
    """
    backbone = model.get_layer("MobileNetV3Large")
    backbone.trainable = True

    # Step 1: freeze all backbone layers unconditionally
    for layer in backbone.layers:
        layer.trainable = False

    # Step 2: unfreeze Conv2D / DepthwiseConv2D in the deeper part
    for i, layer in enumerate(backbone.layers):
        is_conv = isinstance(layer, (tf.keras.layers.Conv2D,
                                     tf.keras.layers.DepthwiseConv2D))
        is_deep = i >= N_FROZEN_LAYERS
        if is_conv and is_deep:
            layer.trainable = True

    trainable = sum(int(tf.size(w)) for w in model.trainable_variables)
    print(f"Trainable parameters after unfreeze: {trainable:,}")

    # Recompile with Adam and a very conservative learning rate
    model.compile(
        loss=tfk.losses.CategoricalCrossentropy(),
        optimizer=tfk.optimizers.Adam(LR_PHASE2),
        metrics=["accuracy"],
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(phase: int = 1) -> list:
    """
    Return the list of Keras callbacks appropriate for the given training phase.

    Both phases use:
      - EarlyStopping: stops training when val_accuracy stops improving for
        ES_PATIENCE epochs and restores the best weights found so far.
        restore_best_weights=True is critical here because ReduceLROnPlateau
        can cause oscillations in the final epochs; we always want the peak.
      - ReduceLROnPlateau: multiplies the learning rate by LR_FACTOR when
        val_accuracy does not improve by at least LR_MIN_DELTA for
        LR_PATIENCE consecutive epochs.  This lets the model escape plateaus
        without manual learning-rate tuning.

    Phase 2 uses a more lenient EarlyStopping patience (11 epochs) to give
    the fine-tuning stage more room to explore before giving up.

    Parameters
    ----------
    phase : 1 or 2

    Returns
    -------
    list of tfk.callbacks instances
    """
    patience = ES_PATIENCE if phase == 1 else 11

    early_stopping = tfk.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tfk.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_delta=LR_MIN_DELTA,
        min_lr=LR_MIN,
        verbose=0,
    )

    return [early_stopping, reduce_lr]
