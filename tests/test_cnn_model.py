"""
Tests for freshsight.model.cnn_model
"""

import numpy as np
import pytest
from PIL import Image

from freshsight.model.cnn_model import (
    FreshnessModel,
    FreshnessPredictor,
    ModelOutput,
    preprocess_image,
)
from freshsight.config import CNN_INPUT_SIZE, CNN_NUM_CLASSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width=300, height=200, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_output_shape(self):
        img = _make_image()
        arr = preprocess_image(img)
        h, w = CNN_INPUT_SIZE[1], CNN_INPUT_SIZE[0]
        assert arr.shape == (1, h, w, 3)

    def test_output_dtype(self):
        img = _make_image()
        arr = preprocess_image(img)
        assert arr.dtype == np.float32

    def test_normalised_values_are_reasonable(self):
        """After normalisation, values should not all be in [0,1] but in a wider range."""
        img = _make_image()
        arr = preprocess_image(img)
        # After ImageNet normalisation the range typically spans [-3, 3]
        assert arr.min() < 0 or arr.max() > 1  # normalised away from [0,1]

    def test_converts_non_rgb_image(self):
        """Grayscale image should be converted to RGB before processing."""
        gray = Image.fromarray(
            np.zeros((100, 100), dtype=np.uint8), mode="L"
        )
        arr = preprocess_image(gray)
        assert arr.shape[-1] == 3


# ---------------------------------------------------------------------------
# FreshnessModel
# ---------------------------------------------------------------------------

class TestFreshnessModel:
    def setup_method(self):
        self.model = FreshnessModel()  # simulation mode (no weights)

    def test_forward_returns_correct_shape(self):
        img = _make_image()
        arr = preprocess_image(img)
        probs = self.model.forward(arr)
        assert probs.shape == (1, CNN_NUM_CLASSES)

    def test_forward_probabilities_sum_to_one(self):
        img = _make_image()
        arr = preprocess_image(img)
        probs = self.model.forward(arr)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_forward_is_deterministic(self):
        """Same image must always produce the same output."""
        img = _make_image(seed=7)
        arr = preprocess_image(img)
        p1 = self.model.forward(arr)
        p2 = self.model.forward(arr)
        np.testing.assert_array_equal(p1, p2)

    def test_different_images_give_different_outputs(self):
        arr1 = preprocess_image(_make_image(seed=1))
        arr2 = preprocess_image(_make_image(seed=2))
        p1 = self.model.forward(arr1)
        p2 = self.model.forward(arr2)
        assert not np.allclose(p1, p2)

    def test_predict_returns_model_output_list(self):
        arr = preprocess_image(_make_image())
        outputs = self.model.predict(arr)
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert isinstance(outputs[0], ModelOutput)

    def test_freshness_score_in_valid_range(self):
        for seed in range(10):
            arr = preprocess_image(_make_image(seed=seed))
            outputs = self.model.predict(arr)
            score = outputs[0].freshness_score
            assert 0.0 <= score <= 100.0, f"Score {score} out of range for seed {seed}"

    def test_freshness_stage_is_valid(self):
        valid_stages = {"fresh", "moderate", "near_expiry", "spoiled"}
        for seed in range(10):
            arr = preprocess_image(_make_image(seed=seed))
            outputs = self.model.predict(arr)
            assert outputs[0].freshness_stage in valid_stages

    def test_batch_predict(self):
        """Model should handle a batch of multiple images."""
        imgs = [_make_image(seed=i) for i in range(3)]
        # Stack preprocessed images into a batch
        arrays = [preprocess_image(img) for img in imgs]
        batch = np.concatenate(arrays, axis=0)
        outputs = self.model.predict(batch)
        assert len(outputs) == 3


# ---------------------------------------------------------------------------
# FreshnessPredictor
# ---------------------------------------------------------------------------

class TestFreshnessPredictor:
    def setup_method(self):
        self.predictor = FreshnessPredictor()

    def test_predict_from_image_returns_model_output(self):
        img = _make_image()
        result = self.predictor.predict_from_image(img)
        assert isinstance(result, ModelOutput)

    def test_score_between_0_and_100(self):
        img = _make_image()
        result = self.predictor.predict_from_image(img)
        assert 0.0 <= result.freshness_score <= 100.0

    def test_predict_from_path(self, tmp_path):
        img = _make_image()
        path = str(tmp_path / "test.png")
        img.save(path)
        result = self.predictor.predict_from_path(path)
        assert isinstance(result, ModelOutput)
        assert 0.0 <= result.freshness_score <= 100.0
