"""
Tests for freshsight.config
"""

import pytest
from freshsight.config import (
    CATEGORY_THRESHOLDS,
    FRESHNESS_STAGES,
    CNN_INPUT_SIZE,
    CNN_NUM_CLASSES,
    MAX_DISCOUNT_PCT,
    MIN_DISCOUNT_PCT,
    FOODBANK_COLLECTION_WINDOW_HOURS,
)


class TestCategoryThresholds:
    def test_default_category_exists(self):
        assert "default" in CATEGORY_THRESHOLDS

    def test_all_categories_have_required_keys(self):
        required_keys = {"full_price", "discount", "foodbank"}
        for category, thresholds in CATEGORY_THRESHOLDS.items():
            assert required_keys == set(thresholds.keys()), (
                f"Category '{category}' is missing required threshold keys"
            )

    def test_threshold_ordering(self):
        """full_price > discount > foodbank for every category."""
        for category, t in CATEGORY_THRESHOLDS.items():
            assert t["full_price"] > t["discount"], (
                f"{category}: full_price threshold must be > discount threshold"
            )
            assert t["discount"] > t["foodbank"], (
                f"{category}: discount threshold must be > foodbank threshold"
            )

    def test_strawberry_discount_threshold_is_70(self):
        assert CATEGORY_THRESHOLDS["strawberry"]["discount"] == 70

    def test_root_vegetable_discount_threshold_is_50(self):
        assert CATEGORY_THRESHOLDS["root_vegetable"]["discount"] == 50

    def test_thresholds_in_valid_score_range(self):
        for category, t in CATEGORY_THRESHOLDS.items():
            for key, value in t.items():
                assert 0 <= value <= 100, (
                    f"{category}.{key} = {value} is outside 0–100"
                )


class TestFreshnessStages:
    def test_four_stages_defined(self):
        assert len(FRESHNESS_STAGES) == 4

    def test_expected_stage_names(self):
        assert set(FRESHNESS_STAGES.keys()) == {
            "fresh", "moderate", "near_expiry", "spoiled"
        }

    def test_stage_ranges_are_valid(self):
        for stage, (lo, hi) in FRESHNESS_STAGES.items():
            assert 0 <= lo <= 100, f"{stage}: lower bound out of range"
            assert 0 <= hi <= 100, f"{stage}: upper bound out of range"
            assert lo <= hi, f"{stage}: lower bound > upper bound"

    def test_fresh_stage_covers_top_scores(self):
        lo, hi = FRESHNESS_STAGES["fresh"]
        assert hi == 100

    def test_spoiled_stage_starts_at_zero(self):
        lo, hi = FRESHNESS_STAGES["spoiled"]
        assert lo == 0


class TestModelSettings:
    def test_input_size_is_tuple_of_two_positive_ints(self):
        assert isinstance(CNN_INPUT_SIZE, tuple)
        assert len(CNN_INPUT_SIZE) == 2
        assert all(isinstance(v, int) and v > 0 for v in CNN_INPUT_SIZE)

    def test_num_classes_matches_stages(self):
        assert CNN_NUM_CLASSES == len(FRESHNESS_STAGES)


class TestPricingSettings:
    def test_discount_pct_range(self):
        assert 0 < MIN_DISCOUNT_PCT < MAX_DISCOUNT_PCT < 1.0

    def test_collection_window_is_positive(self):
        assert FOODBANK_COLLECTION_WINDOW_HOURS > 0
