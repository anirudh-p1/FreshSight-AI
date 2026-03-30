"""
FreshSight AI – CNN Freshness Model

Defines the convolutional neural network (CNN) architecture used to assign a
freshness score (0–100) to an image of produce.

Architecture overview
---------------------
A lightweight CNN with three convolutional blocks followed by fully-connected
classification heads.  In production the model would be trained on a large
dataset of produce images labelled by freshness stage (fresh, moderate,
near_expiry, spoiled).  The weights would then be frozen and deployed to the
edge computing units mounted behind each shelf section.

This module provides:
  • ``FreshnessModel``   – the CNN class (implemented in pure NumPy so no
                           deep-learning framework dependency is required to
                           run the system; swap the forward() body for a real
                           PyTorch / TensorFlow inference call when trained
                           weights are available).
  • ``FreshnessPredictor`` – thin wrapper that loads a model, preprocesses an
                             image and returns a freshness score 0-100.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from freshsight.config import (
    CNN_INPUT_SIZE,
    CNN_NUM_CLASSES,
    FRESHNESS_STAGES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    """Raw output produced by the CNN for a single image."""
    class_probabilities: np.ndarray   # shape (NUM_CLASSES,)
    freshness_stage: str              # e.g. "fresh"
    freshness_score: float            # 0–100


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalise an RGB image for CNN input.

    Parameters
    ----------
    image : PIL.Image.Image
        Raw camera frame.

    Returns
    -------
    np.ndarray
        Float32 array of shape (1, height, width, 3) with pixel values in
        [0, 1].
    """
    img = image.convert("RGB")
    img = img.resize(CNN_INPUT_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Channel-wise normalisation (ImageNet mean/std as a reasonable default)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr[np.newaxis, ...]   # add batch dimension


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

class FreshnessModel:
    """
    Convolutional neural network for produce freshness classification.

    The network outputs a probability distribution over the four freshness
    stages defined in ``config.FRESHNESS_STAGES``:
        0 → fresh  (score 85–100)
        1 → moderate (score 60–84)
        2 → near_expiry (score 35–59)
        3 → spoiled (score 0–34)

    A continuous freshness score is derived from the weighted mean of the
    stage midpoints, scaled to [0, 100].

    Notes
    -----
    The ``forward()`` method below uses a deterministic NumPy simulation
    (seeded from the pixel content) to stand in for real CNN inference.  In
    production, replace the body of ``forward()`` with a call to your
    framework's session / model.predict() using pre-trained weights.
    """

    # Stage midpoints used to compute the scalar freshness score
    _STAGE_ORDER = ["fresh", "moderate", "near_expiry", "spoiled"]
    _STAGE_MIDPOINTS = np.array(
        [
            (hi + lo) / 2.0
            for stage in _STAGE_ORDER
            for lo, hi in [FRESHNESS_STAGES[stage]]
        ],
        dtype=np.float64,
    )

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.weights_path = weights_path
        self._loaded = False
        self._load_weights()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        """
        Load pre-trained weights from ``weights_path``.

        If no path is provided (e.g. during development / testing) the model
        falls back to the simulation mode.
        """
        if self.weights_path and os.path.isfile(self.weights_path):
            logger.info("Loading CNN weights from %s", self.weights_path)
            # In production: load real weights here (e.g. torch.load / tf.saved_model.load)
            self._loaded = True
        else:
            logger.warning(
                "No pre-trained weights found – running in simulation mode. "
                "Train the model and provide a weights file for production use."
            )
            self._loaded = False

    # ------------------------------------------------------------------
    # Forward pass (inference)
    # ------------------------------------------------------------------

    def forward(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run inference on a pre-processed image batch.

        Parameters
        ----------
        image_array : np.ndarray
            Float32 array of shape (batch, height, width, channels).

        Returns
        -------
        np.ndarray
            Probability distribution of shape (batch, NUM_CLASSES).
        """
        if self._loaded:
            # Production path: call framework inference here
            raise NotImplementedError(
                "Replace this with your trained model's inference call."
            )

        # Simulation path – deterministic output seeded from image content so
        # the same image always returns the same score.
        batch_size = image_array.shape[0]
        probs = np.zeros((batch_size, CNN_NUM_CLASSES), dtype=np.float64)
        for i in range(batch_size):
            digest = hashlib.md5(image_array[i].tobytes()).hexdigest()
            seed = int(digest[:8], 16)
            rng = np.random.default_rng(seed)
            raw = rng.dirichlet(alpha=[4.0, 3.0, 2.0, 1.0])
            probs[i] = raw
        return probs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_array: np.ndarray) -> list[ModelOutput]:
        """
        Run inference and return structured outputs for each image in the batch.

        Parameters
        ----------
        image_array : np.ndarray
            Pre-processed batch, shape (batch, height, width, channels).

        Returns
        -------
        list[ModelOutput]
        """
        probs = self.forward(image_array)
        outputs: list[ModelOutput] = []
        for prob_vec in probs:
            stage_idx = int(np.argmax(prob_vec))
            stage_name = self._STAGE_ORDER[stage_idx]
            score = float(np.dot(prob_vec, self._STAGE_MIDPOINTS))
            score = float(np.clip(score, 0.0, 100.0))
            outputs.append(
                ModelOutput(
                    class_probabilities=prob_vec,
                    freshness_stage=stage_name,
                    freshness_score=score,
                )
            )
        return outputs


# ---------------------------------------------------------------------------
# High-level predictor (used by the freshness engine)
# ---------------------------------------------------------------------------

class FreshnessPredictor:
    """
    End-to-end predictor: accepts a raw PIL Image, returns a freshness score.

    Usage
    -----
    >>> predictor = FreshnessPredictor()
    >>> score = predictor.predict_from_image(pil_image)
    >>> print(score)  # e.g. 78.4
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self._model = FreshnessModel(weights_path=weights_path)

    def predict_from_image(self, image: Image.Image) -> ModelOutput:
        """
        Pre-process *image* and return a :class:`ModelOutput`.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        ModelOutput
        """
        arr = preprocess_image(image)
        results = self._model.predict(arr)
        return results[0]

    def predict_from_path(self, image_path: str) -> ModelOutput:
        """
        Load an image from *image_path*, pre-process it and return a
        :class:`ModelOutput`.
        """
        image = Image.open(image_path)
        return self.predict_from_image(image)
