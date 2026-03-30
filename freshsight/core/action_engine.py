"""
FreshSight AI – Action Engine

Receives an ``ItemAssessment`` from the freshness engine and executes the
appropriate automated action:

  Action 1 – No price change; item continues to sell at full price.
  Action 2 – Update the electronic shelf label (ESL) with a discounted price
             calculated by the pricing model.
  Action 3 – Send an automated food-bank collection alert via the REST API.

The ``ActionEngine`` composes the pricing model, ESL integration, and food-bank
API client so that the freshness pipeline has a single call-site.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from freshsight.core.freshness_engine import ItemAssessment
from freshsight.integrations.shelf_label import ShelfLabelClient, LabelUpdate
from freshsight.integrations.foodbank_client import FoodBankClient, CollectionAlert
from freshsight.model.pricing_model import PricingModel, PricingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Summary of the action taken for an assessed item."""
    assessment: ItemAssessment
    action_taken: int
    pricing_result: Optional[PricingResult] = None
    label_updated: bool = False
    foodbank_notified: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Action engine
# ---------------------------------------------------------------------------

class ActionEngine:
    """
    Executes the automated action dictated by an ``ItemAssessment``.

    Parameters
    ----------
    pricing_model : PricingModel, optional
        Pricing model instance.  Created with defaults if not provided.
    shelf_label_client : ShelfLabelClient, optional
        ESL API client.  Created with defaults if not provided.
    foodbank_client : FoodBankClient, optional
        Food-bank API client.  Created with defaults if not provided.
    default_price_lookup : callable, optional
        Function ``(item_id: str) -> float`` that returns the full shelf price.
        Falls back to a fixed £2.00 placeholder if not provided.
    default_hours_to_expiry : int
        Used when no real expiry data is available (default 8 h).
    """

    def __init__(
        self,
        pricing_model: Optional[PricingModel] = None,
        shelf_label_client: Optional[ShelfLabelClient] = None,
        foodbank_client: Optional[FoodBankClient] = None,
        default_price_lookup=None,
        default_hours_to_expiry: int = 8,
    ) -> None:
        self._pricing_model = pricing_model or PricingModel()
        self._shelf_label = shelf_label_client or ShelfLabelClient()
        self._foodbank = foodbank_client or FoodBankClient()
        self._price_lookup = default_price_lookup or (lambda _item_id: 2.00)
        self._default_hours = default_hours_to_expiry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        assessment: ItemAssessment,
        original_price: Optional[float] = None,
        hours_to_expiry: Optional[int] = None,
    ) -> ActionResult:
        """
        Execute the automated action for a single ``ItemAssessment``.

        Parameters
        ----------
        assessment : ItemAssessment
        original_price : float, optional
            Full shelf price.  If omitted, ``default_price_lookup`` is used.
        hours_to_expiry : int, optional
            Estimated hours until the item must be removed.

        Returns
        -------
        ActionResult
        """
        price = original_price if original_price is not None else self._price_lookup(assessment.item_id)
        hours = hours_to_expiry if hours_to_expiry is not None else self._default_hours

        action = assessment.recommended_action

        if action == 1:
            return self._action_full_price(assessment)
        if action == 2:
            return self._action_discount(assessment, price, hours)
        # action == 3
        return self._action_foodbank(assessment, price, hours)

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _action_full_price(self, assessment: ItemAssessment) -> ActionResult:
        """Action 1 – item is fresh; no change needed."""
        logger.info(
            "Action 1 (full price): item=%s score=%.1f",
            assessment.item_id,
            assessment.freshness_score,
        )
        return ActionResult(
            assessment=assessment,
            action_taken=1,
            message=f"Item {assessment.item_id} is fresh (score={assessment.freshness_score:.1f}). "
                    "No price change required.",
        )

    def _action_discount(
        self,
        assessment: ItemAssessment,
        original_price: float,
        hours_to_expiry: int,
    ) -> ActionResult:
        """Action 2 – apply AI-optimised discount and update ESL."""
        pricing = self._pricing_model.calculate_discount(
            category=assessment.category,
            freshness_score=assessment.freshness_score,
            original_price=original_price,
            hours_to_expiry=hours_to_expiry,
        )

        update = LabelUpdate(
            item_id=assessment.item_id,
            original_price=pricing.original_price,
            discounted_price=pricing.discounted_price,
            discount_pct=pricing.discount_pct,
            label_text=pricing.label,
            freshness_score=assessment.freshness_score,
        )
        label_ok = self._shelf_label.update(update)

        logger.info(
            "Action 2 (discount): item=%s score=%.1f → %.0f%% off → £%.2f label_ok=%s",
            assessment.item_id,
            assessment.freshness_score,
            pricing.discount_pct * 100,
            pricing.discounted_price,
            label_ok,
        )

        return ActionResult(
            assessment=assessment,
            action_taken=2,
            pricing_result=pricing,
            label_updated=label_ok,
            message=(
                f"Item {assessment.item_id} discounted by {pricing.discount_pct:.0%} "
                f"to £{pricing.discounted_price:.2f} ({pricing.label})."
            ),
        )

    def _action_foodbank(
        self,
        assessment: ItemAssessment,
        original_price: float,
        hours_to_expiry: int,
    ) -> ActionResult:
        """Action 3 – alert registered food-bank partners."""
        alert = CollectionAlert(
            item_id=assessment.item_id,
            category=assessment.category,
            quantity=assessment.quantity,
            freshness_score=assessment.freshness_score,
            freshness_stage=assessment.freshness_stage,
            collection_window_hours=hours_to_expiry,
            assessed_at=assessment.assessed_at,
        )
        notified = self._foodbank.send_alert(alert)

        logger.info(
            "Action 3 (food bank): item=%s score=%.1f notified=%s",
            assessment.item_id,
            assessment.freshness_score,
            notified,
        )

        return ActionResult(
            assessment=assessment,
            action_taken=3,
            foodbank_notified=notified,
            message=(
                f"Food bank alerted for item {assessment.item_id} "
                f"(score={assessment.freshness_score:.1f}, qty={assessment.quantity}). "
                f"Collection window: {hours_to_expiry} h."
            ),
        )
