"""
Tests for freshsight.core.action_engine
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from freshsight.core.freshness_engine import ItemAssessment
from freshsight.core.action_engine import ActionEngine, ActionResult
from freshsight.model.pricing_model import PricingResult
from freshsight.config import CATEGORY_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_assessment(
    action: int,
    item_id: str = "item-001",
    category: str = "strawberry",
    score: float = 75.0,
    quantity: int = 5,
) -> ItemAssessment:
    return ItemAssessment(
        item_id=item_id,
        category=category,
        freshness_score=score,
        freshness_stage="moderate",
        class_probabilities=[0.1, 0.7, 0.15, 0.05],
        recommended_action=action,
        thresholds=CATEGORY_THRESHOLDS.get(category, CATEGORY_THRESHOLDS["default"]),
        camera_id="cam-test",
        quantity=quantity,
    )


def _make_engine(
    label_ok: bool = True,
    foodbank_ok: bool = True,
) -> ActionEngine:
    pricing_model = MagicMock()
    pricing_model.calculate_discount.return_value = PricingResult(
        original_price=2.50,
        discount_pct=0.20,
        discounted_price=2.00,
        label="20% off – sell today",
        hours_to_expiry=6,
    )

    shelf_label = MagicMock()
    shelf_label.update.return_value = label_ok

    foodbank = MagicMock()
    foodbank.send_alert.return_value = foodbank_ok

    return ActionEngine(
        pricing_model=pricing_model,
        shelf_label_client=shelf_label,
        foodbank_client=foodbank,
    )


# ---------------------------------------------------------------------------
# Action 1 – full price
# ---------------------------------------------------------------------------

class TestActionFullPrice:
    def test_returns_action_result(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=1))
        assert isinstance(result, ActionResult)

    def test_action_taken_is_1(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=1))
        assert result.action_taken == 1

    def test_no_pricing_result(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=1))
        assert result.pricing_result is None

    def test_label_not_updated(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=1))
        assert result.label_updated is False

    def test_foodbank_not_notified(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=1))
        assert result.foodbank_notified is False

    def test_message_references_item_id(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=1, item_id="ITEM-XYZ"))
        assert "ITEM-XYZ" in result.message


# ---------------------------------------------------------------------------
# Action 2 – discount
# ---------------------------------------------------------------------------

class TestActionDiscount:
    def test_action_taken_is_2(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=2))
        assert result.action_taken == 2

    def test_pricing_result_populated(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=2))
        assert result.pricing_result is not None

    def test_label_updated_when_esl_succeeds(self):
        engine = _make_engine(label_ok=True)
        result = engine.execute(_make_assessment(action=2))
        assert result.label_updated is True

    def test_label_not_updated_when_esl_fails(self):
        engine = _make_engine(label_ok=False)
        result = engine.execute(_make_assessment(action=2))
        assert result.label_updated is False

    def test_foodbank_not_notified_for_discount(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=2))
        assert result.foodbank_notified is False

    def test_message_contains_discount_info(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=2))
        assert "discount" in result.message.lower() or "%" in result.message

    def test_custom_price_used(self):
        engine = _make_engine()
        engine.execute(_make_assessment(action=2), original_price=3.99)
        engine._pricing_model.calculate_discount.assert_called_once()
        _, kwargs = engine._pricing_model.calculate_discount.call_args
        assert kwargs.get("original_price") == 3.99 or (
            engine._pricing_model.calculate_discount.call_args[0][2] == 3.99
        )


# ---------------------------------------------------------------------------
# Action 3 – food bank
# ---------------------------------------------------------------------------

class TestActionFoodBank:
    def test_action_taken_is_3(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=3))
        assert result.action_taken == 3

    def test_foodbank_notified_when_api_succeeds(self):
        engine = _make_engine(foodbank_ok=True)
        result = engine.execute(_make_assessment(action=3))
        assert result.foodbank_notified is True

    def test_foodbank_not_notified_when_api_fails(self):
        engine = _make_engine(foodbank_ok=False)
        result = engine.execute(_make_assessment(action=3))
        assert result.foodbank_notified is False

    def test_no_pricing_result_for_foodbank(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=3))
        assert result.pricing_result is None

    def test_label_not_updated_for_foodbank(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=3))
        assert result.label_updated is False

    def test_message_mentions_food_bank(self):
        engine = _make_engine()
        result = engine.execute(_make_assessment(action=3))
        assert "food bank" in result.message.lower() or "alert" in result.message.lower()

    def test_quantity_included_in_alert(self):
        engine = _make_engine()
        engine.execute(_make_assessment(action=3, quantity=12))
        alert_arg = engine._foodbank.send_alert.call_args[0][0]
        assert alert_arg.quantity == 12


# ---------------------------------------------------------------------------
# Default price lookup
# ---------------------------------------------------------------------------

class TestDefaultPriceLookup:
    def test_custom_price_lookup_used(self):
        custom_lookup = lambda item_id: 5.99
        engine = ActionEngine(
            pricing_model=MagicMock(
                calculate_discount=MagicMock(
                    return_value=PricingResult(5.99, 0.20, 4.79, "20% off", 6)
                )
            ),
            shelf_label_client=MagicMock(update=MagicMock(return_value=True)),
            foodbank_client=MagicMock(send_alert=MagicMock(return_value=True)),
            default_price_lookup=custom_lookup,
        )
        engine.execute(_make_assessment(action=2))
        call_kwargs = engine._pricing_model.calculate_discount.call_args
        # price should be 5.99 as returned by custom_lookup
        args, kwargs = call_kwargs
        price_used = kwargs.get("original_price") or args[2]
        assert price_used == pytest.approx(5.99)
