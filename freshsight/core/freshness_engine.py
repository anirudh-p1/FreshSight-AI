"""
FreshSight AI – Freshness Engine

Orchestrates the end-to-end pipeline for a single camera frame:
  1. Receive a raw image from the shelf camera.
  2. Run the CNN model to produce a freshness score (0–100).
  3. Look up the category-specific thresholds.
  4. Return an ``ItemAssessment`` ready for the action engine.

The engine is stateless and can be called in a tight loop by the camera
capture thread running on the edge computing unit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from PIL import Image

from freshsight.config import CATEGORY_THRESHOLDS
from freshsight.model.cnn_model import FreshnessPredictor, ModelOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ItemAssessment:
    """
    The result of assessing a single shelf item.

    Attributes
    ----------
    item_id : str
        Unique identifier for the product (e.g. barcode / EAN).
    category : str
        Produce category string (must match config.CATEGORY_THRESHOLDS key).
    freshness_score : float
        CNN freshness score in [0, 100].
    freshness_stage : str
        Qualitative stage: 'fresh', 'moderate', 'near_expiry', or 'spoiled'.
    class_probabilities : list[float]
        CNN output probability for each freshness stage.
    recommended_action : int
        1 = full price, 2 = discount, 3 = food-bank alert.
    thresholds : dict[str, int]
        The thresholds that were applied (full_price, discount, foodbank).
    assessed_at : datetime
        UTC timestamp of the assessment.
    camera_id : str
        Identifier of the camera that captured the image.
    quantity : int
        Number of units of this item visible on the shelf.
    """
    item_id: str
    category: str
    freshness_score: float
    freshness_stage: str
    class_probabilities: list[float]
    recommended_action: int
    thresholds: dict[str, int]
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    camera_id: str = "unknown"
    quantity: int = 1


# ---------------------------------------------------------------------------
# Freshness engine
# ---------------------------------------------------------------------------

class FreshnessEngine:
    """
    Main orchestrator for the FreshSight AI freshness assessment pipeline.

    Parameters
    ----------
    predictor : FreshnessPredictor, optional
        CNN predictor instance.  Created automatically if not provided.
    weights_path : str, optional
        Path to pre-trained model weights (passed to FreshnessPredictor).
    """

    def __init__(
        self,
        predictor: Optional[FreshnessPredictor] = None,
        weights_path: Optional[str] = None,
    ) -> None:
        self._predictor = predictor or FreshnessPredictor(weights_path=weights_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_image(
        self,
        image: Image.Image,
        item_id: str,
        category: str,
        camera_id: str = "unknown",
        quantity: int = 1,
    ) -> ItemAssessment:
        """
        Assess a produce item from a raw camera image.

        Parameters
        ----------
        image : PIL.Image.Image
            Raw camera frame containing the produce.
        item_id : str
            Barcode / EAN of the product.
        category : str
            Produce category (e.g. 'strawberry', 'root_vegetable').
        camera_id : str
            Identifier of the shelf camera.
        quantity : int
            Number of units visible on the shelf.

        Returns
        -------
        ItemAssessment
        """
        output: ModelOutput = self._predictor.predict_from_image(image)
        return self._build_assessment(output, item_id, category, camera_id, quantity)

    def assess_from_path(
        self,
        image_path: str,
        item_id: str,
        category: str,
        camera_id: str = "unknown",
        quantity: int = 1,
    ) -> ItemAssessment:
        """
        Assess a produce item from an image file path.

        Parameters
        ----------
        image_path : str
            File-system path to the image.
        item_id : str
            Barcode / EAN of the product.
        category : str
            Produce category.
        camera_id : str
        quantity : int

        Returns
        -------
        ItemAssessment
        """
        output: ModelOutput = self._predictor.predict_from_path(image_path)
        return self._build_assessment(output, item_id, category, camera_id, quantity)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_assessment(
        self,
        output: ModelOutput,
        item_id: str,
        category: str,
        camera_id: str,
        quantity: int,
    ) -> ItemAssessment:
        thresholds = self._get_thresholds(category)
        action = self._decide_action(output.freshness_score, thresholds)

        logger.info(
            "Assessment: item=%s category=%s score=%.1f stage=%s action=%d",
            item_id,
            category,
            output.freshness_score,
            output.freshness_stage,
            action,
        )

        return ItemAssessment(
            item_id=item_id,
            category=category,
            freshness_score=output.freshness_score,
            freshness_stage=output.freshness_stage,
            class_probabilities=output.class_probabilities.tolist(),
            recommended_action=action,
            thresholds=thresholds,
            camera_id=camera_id,
            quantity=quantity,
        )

    @staticmethod
    def _get_thresholds(category: str) -> dict[str, int]:
        """Return thresholds for *category*, falling back to 'default'."""
        return CATEGORY_THRESHOLDS.get(category, CATEGORY_THRESHOLDS["default"])

    @staticmethod
    def _decide_action(score: float, thresholds: dict[str, int]) -> int:
        """
        Map a freshness score to one of the three automated actions.

        Action 1 – sell at full price  (score >= full_price threshold)
        Action 2 – apply AI discount   (discount threshold <= score < full_price)
        Action 3 – alert food bank     (score < foodbank threshold)

        Parameters
        ----------
        score : float
        thresholds : dict with keys 'full_price', 'discount', 'foodbank'

        Returns
        -------
        int
            1, 2, or 3.
        """
        if score >= thresholds["full_price"]:
            return 1
        if score >= thresholds["foodbank"]:
            return 2
        return 3
