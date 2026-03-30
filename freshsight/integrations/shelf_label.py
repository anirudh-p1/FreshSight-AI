"""
FreshSight AI – Electronic Shelf Label (ESL) Integration

Sends price-update commands to the ESL controller API so that the electronic
barcode / shelf label for a discounted item is updated in real time.

The ``ShelfLabelClient`` communicates with the ESL controller over its REST
API.  In a real store deployment the ESL controller is a small server running
on the store LAN that manages all the Bluetooth / RF-connected shelf labels.

Usage
-----
>>> client = ShelfLabelClient()
>>> update = LabelUpdate(item_id="5012345678900", ...)
>>> success = client.update(update)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests
from requests.exceptions import RequestException

from freshsight.config import ESL_API_BASE_URL, ESL_API_VERSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LabelUpdate:
    """
    Payload sent to the ESL controller to update a shelf label.

    Attributes
    ----------
    item_id : str
        Barcode / EAN that identifies the product.
    original_price : float
        Full (pre-discount) shelf price.
    discounted_price : float
        New price to display on the label.
    discount_pct : float
        Fraction discount applied (0–1).
    label_text : str
        Human-readable label string (e.g. "20% off – sell today").
    freshness_score : float
        CNN freshness score that triggered the update (informational).
    """
    item_id: str
    original_price: float
    discounted_price: float
    discount_pct: float
    label_text: str
    freshness_score: float


# ---------------------------------------------------------------------------
# ESL client
# ---------------------------------------------------------------------------

class ShelfLabelClient:
    """
    REST client for the Electronic Shelf Label (ESL) controller API.

    Parameters
    ----------
    base_url : str, optional
        ESL controller URL.  Defaults to config.ESL_API_BASE_URL.
    api_version : str, optional
        API version string.  Defaults to config.ESL_API_VERSION.
    timeout : int
        HTTP request timeout in seconds (default 5).
    session : requests.Session, optional
        Injectable session for testing.
    """

    def __init__(
        self,
        base_url: str = ESL_API_BASE_URL,
        api_version: str = ESL_API_VERSION,
        timeout: int = 5,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._timeout = timeout
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, update: LabelUpdate) -> bool:
        """
        Send a price-update command to the ESL controller.

        Parameters
        ----------
        update : LabelUpdate

        Returns
        -------
        bool
            ``True`` if the controller acknowledged the update successfully.
        """
        endpoint = (
            f"{self._base_url}/api/{self._api_version}"
            f"/labels/{update.item_id}/price"
        )
        payload = {
            "item_id":          update.item_id,
            "original_price":   update.original_price,
            "discounted_price": update.discounted_price,
            "discount_pct":     round(update.discount_pct, 4),
            "label_text":       update.label_text,
            "freshness_score":  update.freshness_score,
        }
        try:
            response = self._session.put(
                endpoint, json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            logger.info(
                "ESL updated for item %s → £%.2f (%s)",
                update.item_id,
                update.discounted_price,
                update.label_text,
            )
            return True
        except RequestException as exc:
            logger.error("ESL update failed for item %s: %s", update.item_id, exc)
            return False

    def reset_to_full_price(self, item_id: str, original_price: float) -> bool:
        """
        Restore an item's ESL to its full original price.

        Parameters
        ----------
        item_id : str
        original_price : float

        Returns
        -------
        bool
        """
        endpoint = (
            f"{self._base_url}/api/{self._api_version}"
            f"/labels/{item_id}/price"
        )
        payload = {
            "item_id":          item_id,
            "original_price":   original_price,
            "discounted_price": original_price,
            "discount_pct":     0.0,
            "label_text":       "Full price",
            "freshness_score":  None,
        }
        try:
            response = self._session.put(
                endpoint, json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            logger.info("ESL reset to full price for item %s", item_id)
            return True
        except RequestException as exc:
            logger.error("ESL reset failed for item %s: %s", item_id, exc)
            return False
