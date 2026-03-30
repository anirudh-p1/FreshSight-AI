"""
Tests for freshsight.core.freshness_engine
"""

import numpy as np
import pytest
from PIL import Image

from freshsight.core.freshness_engine import FreshnessEngine, ItemAssessment
from freshsight.model.cnn_model import FreshnessPredictor, ModelOutput
from freshsight.config import CATEGORY_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


class _StubPredictor:
    """Predictable predictor that always returns a fixed score."""

    def __init__(self, score: float, stage: str = "moderate"):
        self._score = score
        self._stage = stage

    def predict_from_image(self, image) -> ModelOutput:
        return ModelOutput(
            class_probabilities=np.array([0.1, 0.7, 0.15, 0.05]),
            freshness_stage=self._stage,
            freshness_score=self._score,
        )

    def predict_from_path(self, path) -> ModelOutput:
        return self.predict_from_image(None)


# ---------------------------------------------------------------------------
# FreshnessEngine._decide_action
# ---------------------------------------------------------------------------

class TestDecideAction:
    thresholds = CATEGORY_THRESHOLDS["strawberry"]  # full=85, discount=70, foodbank=40

    def test_action_1_at_full_price_threshold(self):
        assert FreshnessEngine._decide_action(85, self.thresholds) == 1

    def test_action_1_above_full_price_threshold(self):
        assert FreshnessEngine._decide_action(100, self.thresholds) == 1

    def test_action_2_just_below_full_price(self):
        assert FreshnessEngine._decide_action(84, self.thresholds) == 2

    def test_action_2_at_discount_threshold(self):
        assert FreshnessEngine._decide_action(70, self.thresholds) == 2

    def test_action_2_just_above_foodbank(self):
        assert FreshnessEngine._decide_action(41, self.thresholds) == 2

    def test_action_3_just_below_foodbank(self):
        assert FreshnessEngine._decide_action(39, self.thresholds) == 3

    def test_action_3_at_zero(self):
        assert FreshnessEngine._decide_action(0, self.thresholds) == 3

    def test_root_vegetable_thresholds(self):
        t = CATEGORY_THRESHOLDS["root_vegetable"]  # full=75, discount=50, foodbank=25
        assert FreshnessEngine._decide_action(75, t) == 1
        assert FreshnessEngine._decide_action(50, t) == 2
        assert FreshnessEngine._decide_action(24, t) == 3


class TestGetThresholds:
    def test_known_category_returns_its_thresholds(self):
        t = FreshnessEngine._get_thresholds("strawberry")
        assert t["discount"] == 70

    def test_unknown_category_falls_back_to_default(self):
        t = FreshnessEngine._get_thresholds("purple_carrot")
        assert t == CATEGORY_THRESHOLDS["default"]


# ---------------------------------------------------------------------------
# FreshnessEngine.assess_image
# ---------------------------------------------------------------------------

class TestAssessImage:
    def _engine_with_score(self, score: float) -> FreshnessEngine:
        return FreshnessEngine(predictor=_StubPredictor(score))

    def test_returns_item_assessment_dataclass(self):
        engine = self._engine_with_score(90)
        result = engine.assess_image(
            _make_image(), "item-001", "strawberry"
        )
        assert isinstance(result, ItemAssessment)

    def test_assessment_contains_correct_item_id(self):
        engine = self._engine_with_score(90)
        result = engine.assess_image(_make_image(), "item-999", "strawberry")
        assert result.item_id == "item-999"

    def test_assessment_contains_correct_category(self):
        engine = self._engine_with_score(90)
        result = engine.assess_image(_make_image(), "item-001", "banana")
        assert result.category == "banana"

    def test_action_1_when_score_is_high(self):
        engine = self._engine_with_score(95)
        result = engine.assess_image(_make_image(), "item-001", "strawberry")
        assert result.recommended_action == 1

    def test_action_2_when_score_in_discount_band(self):
        engine = self._engine_with_score(75)  # between 70 and 85 for strawberry
        result = engine.assess_image(_make_image(), "item-001", "strawberry")
        assert result.recommended_action == 2

    def test_action_3_when_score_below_foodbank_threshold(self):
        engine = self._engine_with_score(30)  # below 40 for strawberry
        result = engine.assess_image(_make_image(), "item-001", "strawberry")
        assert result.recommended_action == 3

    def test_camera_id_propagated(self):
        engine = self._engine_with_score(90)
        result = engine.assess_image(
            _make_image(), "item-001", "strawberry", camera_id="cam-A1"
        )
        assert result.camera_id == "cam-A1"

    def test_quantity_propagated(self):
        engine = self._engine_with_score(90)
        result = engine.assess_image(
            _make_image(), "item-001", "strawberry", quantity=15
        )
        assert result.quantity == 15

    def test_thresholds_included_in_assessment(self):
        engine = self._engine_with_score(90)
        result = engine.assess_image(_make_image(), "item-001", "strawberry")
        assert "full_price" in result.thresholds
        assert "discount" in result.thresholds
        assert "foodbank" in result.thresholds

    def test_class_probabilities_included(self):
        engine = self._engine_with_score(75)
        result = engine.assess_image(_make_image(), "item-001", "apple")
        assert isinstance(result.class_probabilities, list)
        assert len(result.class_probabilities) == 4

    def test_assess_from_path(self, tmp_path):
        img = _make_image()
        path = str(tmp_path / "img.png")
        img.save(path)
        engine = FreshnessEngine()
        result = engine.assess_from_path(path, "item-001", "banana")
        assert isinstance(result, ItemAssessment)
        assert 0 <= result.freshness_score <= 100
