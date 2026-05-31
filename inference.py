"""
inference.py
------------
Production inference class used for competition submission.

The Model class wraps the trained .keras model and applies Test-Time
Augmentation (TTA) at inference time to improve prediction robustness.

Usage (standalone)
------------------
    from inference import Model

    model  = Model("outputs/BloodCell_MobileNetV3L_98.5.keras")
    labels = model.predict(X_test)   # X_test: (N, 96, 96, 3), uint8

Usage (competition submission)
-------------------------------
The competition grader imports this file and calls Model().predict(X).
The model filename must match the .keras file included in the zip archive.

How TTA works here
------------------
  1. The input batch X (values in [0, 255]) is normalised to [0, 1].
  2. The original batch is fed to the network → softmax output P0.
  3. N_TTA augmented copies are generated and fed → P1, P2, ..., P_N.
  4. The N+1 probability matrices are stacked and averaged → P_mean.
  5. The predicted class is argmax(P_mean) for each sample.

This effectively creates an ensemble of N+1 slightly different "views"
of each image, reducing the variance of the final prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

from config import N_TTA, MODEL_FINAL_PATH


class Model:
    """
    Wrapper around the trained MobileNetV3Large classifier.

    Parameters
    ----------
    model_path : path to the saved .keras file.
                 Defaults to the path produced by train.py.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Fall back to the most recent trained model
            model_path = MODEL_FINAL_PATH

        # Load the complete model (architecture + weights + compilation state)
        self.neural_network = tfk.models.load_model(model_path)

        # Augmentation pipeline applied at test time.
        # The transforms are milder than training augmentation to avoid
        # distorting the image structure beyond what the model was trained on.
        self.tta_augmentation = tfk.Sequential([
            tfkl.RandomFlip("horizontal_and_vertical"),
            tfkl.RandomRotation(0.5),
            tfkl.RandomContrast(0.35),
            tfkl.RandomTranslation(0.2, 0.2),
        ], name="tta_augmentation")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a batch of images.

        Parameters
        ----------
        X : np.ndarray, shape (N, 96, 96, 3)
            Raw uint8 images in [0, 255].

        Returns
        -------
        predictions : np.ndarray, shape (N,)
            Integer class label for each image (0–7).
        """
        # Normalise to [0, 1] for the augmentation layer
        X_norm = X / 255.0

        # Original forward pass — shape (1, N, 8) after unsqueezing
        preds = self.neural_network.predict(X, verbose=0)[np.newaxis, ...]

        for _ in range(N_TTA):
            # Augment in [0,1] space, then rescale back to [0,255] for the model
            X_aug   = self.tta_augmentation(X_norm).numpy() * 255.0
            p       = self.neural_network.predict(X_aug, verbose=0)[np.newaxis, ...]
            preds   = np.concatenate([preds, p], axis=0)

        # Average the N_TTA+1 softmax probability vectors
        mean_preds = np.mean(preds, axis=0)         # shape (N, 8)

        # Return the class with the highest average probability
        return np.argmax(mean_preds, axis=1)
