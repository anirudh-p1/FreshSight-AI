"""
FreshSight AI – Dynamic Pricing Model

When a freshness score falls into the discount band, this module calculates
the minimum price reduction needed to ensure the item sells before it expires.

The pricing model uses a simple linear regression baseline trained on
historical sales-velocity data.  In production, a richer secondary predictive
model (e.g. gradient-boosted trees or a neural net) can be swapped in by
implementing the same ``PricingModel`` interface.

Design goals
------------
• Prevent *over-discounting* (sacrificing revenue unnecessarily).
• Prevent *under-discounting* (leaving unsold stock that becomes waste).
• Integrate with real historical data via the ``SalesRecord`` dataclass.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from freshsight.config import MAX_DISCOUNT_PCT, MIN_DISCOUNT_PCT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SalesRecord:
    """
    Historical sales observation used to calibrate the pricing model.

    Attributes
    ----------
    category : str
        Produce category (must match a key in config.CATEGORY_THRESHOLDS).
    freshness_score : float
        Freshness score at the time the item was discounted.
    discount_pct : float
        Fraction discounted (0–1).  E.g. 0.20 means 20 % off.
    sold_before_expiry : bool
        Whether the item sold within the acceptable window.
    hours_to_expiry : int
        Estimated hours left on the item at the time of discount.
    """
    category: str
    freshness_score: float
    discount_pct: float
    sold_before_expiry: bool
    hours_to_expiry: int


@dataclass
class PricingResult:
    """Result of a pricing calculation."""
    original_price: float
    discount_pct: float
    discounted_price: float
    label: str                  # human-readable e.g. "20% off – sell today"
    hours_to_expiry: int


# ---------------------------------------------------------------------------
# Default historical sales dataset (illustrative; replace with real DB query)
# ---------------------------------------------------------------------------

DEFAULT_SALES_HISTORY: list[SalesRecord] = [
    SalesRecord("strawberry",   68, 0.25, True,  6),
    SalesRecord("strawberry",   55, 0.40, True,  3),
    SalesRecord("strawberry",   45, 0.50, False, 2),
    SalesRecord("banana",       58, 0.20, True,  8),
    SalesRecord("banana",       48, 0.35, True,  4),
    SalesRecord("apple",        57, 0.15, True, 12),
    SalesRecord("lettuce",      60, 0.20, True,  8),
    SalesRecord("lettuce",      48, 0.35, True,  4),
    SalesRecord("tomato",       63, 0.15, True, 10),
    SalesRecord("root_vegetable", 48, 0.10, True, 24),
    SalesRecord("root_vegetable", 35, 0.25, True, 12),
    SalesRecord("default",      58, 0.20, True,  8),
]


# ---------------------------------------------------------------------------
# Pricing model
# ---------------------------------------------------------------------------

class PricingModel:
    """
    Estimates the minimum discount percentage needed for an item to sell
    before it expires, given its freshness score and category.

    The baseline algorithm uses a linear function:

        discount_pct = base_rate
                     + score_slope  * (threshold - score)
                     + time_slope   * (1 / max(1, hours_to_expiry))

    Coefficients are estimated per category from historical ``SalesRecord``
    data via ordinary-least-squares (manual, dependency-free implementation).
    If fewer than three records exist for a category the model falls back to a
    simple heuristic.
    """

    def __init__(
        self,
        sales_history: Optional[list[SalesRecord]] = None,
    ) -> None:
        self._history: list[SalesRecord] = (
            sales_history if sales_history is not None else DEFAULT_SALES_HISTORY
        )
        self._coeffs: dict[str, tuple[float, float, float]] = {}
        self._fit()

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Fit per-category linear models from sales history."""
        from collections import defaultdict

        by_category: dict[str, list[SalesRecord]] = defaultdict(list)
        for record in self._history:
            by_category[record.category].append(record)

        for category, records in by_category.items():
            if len(records) < 3:
                self._coeffs[category] = self._heuristic_coeffs()
                continue
            self._coeffs[category] = self._ols(records)

    @staticmethod
    def _heuristic_coeffs() -> tuple[float, float, float]:
        """Fallback coefficients when not enough data is available."""
        return (0.10, 0.005, 0.02)

    @staticmethod
    def _ols(records: list[SalesRecord]) -> tuple[float, float, float]:
        """
        Ordinary-least-squares fit of the linear model:
            discount = b0 + b1 * score_gap + b2 * inv_time

        Returns (b0, b1, b2).
        """
        n = len(records)
        X = []
        y = []
        for r in records:
            inv_time = 1.0 / max(1, r.hours_to_expiry)
            X.append([1.0, r.freshness_score, inv_time])
            y.append(r.discount_pct)

        import numpy as np  # already in requirements
        X_arr = np.array(X)
        y_arr = np.array(y)
        try:
            coeffs, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
            return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        except np.linalg.LinAlgError:
            logger.warning("OLS failed for category – using heuristic.")
            return PricingModel._heuristic_coeffs()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def calculate_discount(
        self,
        category: str,
        freshness_score: float,
        original_price: float,
        hours_to_expiry: int = 8,
    ) -> PricingResult:
        """
        Calculate the optimised discount for an item.

        Parameters
        ----------
        category : str
            Produce category string.
        freshness_score : float
            Current CNN freshness score (0–100).
        original_price : float
            Full shelf price in store currency.
        hours_to_expiry : int
            Estimated hours until the item must be removed from sale.

        Returns
        -------
        PricingResult
        """
        b0, b1, b2 = self._coeffs.get(
            category, self._coeffs.get("default", self._heuristic_coeffs())
        )
        inv_time = 1.0 / max(1, hours_to_expiry)
        raw_discount = b0 + b1 * freshness_score + b2 * inv_time

        # Clamp to acceptable range
        discount_pct = max(MIN_DISCOUNT_PCT, min(MAX_DISCOUNT_PCT, raw_discount))
        discounted_price = round(original_price * (1.0 - discount_pct), 2)

        label = self._make_label(discount_pct, hours_to_expiry)

        logger.debug(
            "Pricing: category=%s score=%.1f hrs=%d → %.0f%% off → £%.2f",
            category, freshness_score, hours_to_expiry,
            discount_pct * 100, discounted_price,
        )
        return PricingResult(
            original_price=original_price,
            discount_pct=discount_pct,
            discounted_price=discounted_price,
            label=label,
            hours_to_expiry=hours_to_expiry,
        )

    @staticmethod
    def _make_label(discount_pct: float, hours_to_expiry: int) -> str:
        """Generate a human-readable shelf-label string."""
        pct_str = f"{math.floor(discount_pct * 100)}% off"
        if hours_to_expiry <= 2:
            urgency = "use today"
        elif hours_to_expiry <= 8:
            urgency = "sell today"
        else:
            urgency = "reduced to clear"
        return f"{pct_str} – {urgency}"
