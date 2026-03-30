"""
Tests for freshsight.model.pricing_model
"""

import pytest
from freshsight.model.pricing_model import PricingModel, PricingResult, SalesRecord
from freshsight.config import MAX_DISCOUNT_PCT, MIN_DISCOUNT_PCT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return PricingModel()


@pytest.fixture
def model_minimal_data():
    """Model fitted on very few records → falls back to heuristic."""
    history = [
        SalesRecord("strawberry", 65, 0.20, True, 6),
    ]
    return PricingModel(sales_history=history)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPricingModelBasics:
    def test_returns_pricing_result(self, model):
        result = model.calculate_discount("strawberry", 72, 2.50)
        assert isinstance(result, PricingResult)

    def test_discount_pct_within_bounds(self, model):
        for score in [35, 50, 65, 80]:
            result = model.calculate_discount("strawberry", score, 3.00)
            assert MIN_DISCOUNT_PCT <= result.discount_pct <= MAX_DISCOUNT_PCT, (
                f"score={score}: discount {result.discount_pct} out of bounds"
            )

    def test_discounted_price_less_than_original(self, model):
        result = model.calculate_discount("banana", 55, 1.20)
        assert result.discounted_price < result.original_price

    def test_discounted_price_is_positive(self, model):
        result = model.calculate_discount("root_vegetable", 45, 0.80)
        assert result.discounted_price > 0

    def test_original_price_preserved_in_result(self, model):
        result = model.calculate_discount("apple", 58, 1.80)
        assert result.original_price == pytest.approx(1.80)

    def test_label_is_non_empty_string(self, model):
        result = model.calculate_discount("lettuce", 62, 1.50)
        assert isinstance(result.label, str)
        assert len(result.label) > 0

    def test_hours_to_expiry_propagated(self, model):
        result = model.calculate_discount("strawberry", 65, 2.00, hours_to_expiry=3)
        assert result.hours_to_expiry == 3

    def test_unknown_category_uses_default(self, model):
        result = model.calculate_discount("purple_mango", 60, 2.00)
        assert MIN_DISCOUNT_PCT <= result.discount_pct <= MAX_DISCOUNT_PCT

    def test_label_urgent_for_short_window(self, model):
        result = model.calculate_discount("strawberry", 65, 2.00, hours_to_expiry=1)
        assert "today" in result.label.lower() or "use" in result.label.lower()

    def test_label_reduced_for_long_window(self, model):
        result = model.calculate_discount("root_vegetable", 45, 0.80, hours_to_expiry=24)
        assert "reduced" in result.label.lower() or "clear" in result.label.lower()


class TestPricingModelWithMinimalData:
    def test_heuristic_fallback_still_returns_valid_result(self, model_minimal_data):
        result = model_minimal_data.calculate_discount("strawberry", 65, 2.00)
        assert isinstance(result, PricingResult)
        assert MIN_DISCOUNT_PCT <= result.discount_pct <= MAX_DISCOUNT_PCT


class TestDiscountClamping:
    def test_very_low_score_does_not_exceed_max_discount(self):
        model = PricingModel()
        result = model.calculate_discount("strawberry", 1, 10.00, hours_to_expiry=1)
        assert result.discount_pct <= MAX_DISCOUNT_PCT

    def test_borderline_score_applies_at_least_min_discount(self):
        model = PricingModel()
        result = model.calculate_discount("root_vegetable", 49, 1.00, hours_to_expiry=48)
        assert result.discount_pct >= MIN_DISCOUNT_PCT


class TestMakeLabel:
    def test_urgent_label_for_1_hour(self):
        label = PricingModel._make_label(0.30, 1)
        assert "use today" in label

    def test_sell_today_label_for_6_hours(self):
        label = PricingModel._make_label(0.20, 6)
        assert "sell today" in label

    def test_reduced_to_clear_label_for_24_hours(self):
        label = PricingModel._make_label(0.15, 24)
        assert "reduced to clear" in label

    def test_label_contains_percentage(self):
        label = PricingModel._make_label(0.25, 5)
        assert "25%" in label
