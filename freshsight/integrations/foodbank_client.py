"""
FreshSight AI – Food Bank REST API Client

When an item's freshness score falls below the food-bank threshold, this
module sends an automated collection alert to all registered food-bank
partners via a simple REST API.

The food bank receives:
  • item type and category
  • quantity available for collection
  • collection window (time from alert to latest pick-up)
  • store location and contact details (configured in the server)

Partners view the alert in their partner app and arrange collection within the
collection window.

Usage
-----
>>> client = FoodBankClient()
>>> alert = CollectionAlert(item_id="5012345678900", ...)
>>> success = client.send_alert(alert)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from requests.exceptions import RequestException

from freshsight.config import (
    FOODBANK_API_BASE_URL,
    FOODBANK_API_VERSION,
    FOODBANK_COLLECTION_WINDOW_HOURS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CollectionAlert:
    """
    Alert payload sent to food-bank partner organisations.

    Attributes
    ----------
    item_id : str
        Barcode / EAN of the product.
    category : str
        Produce category (e.g. 'strawberry').
    quantity : int
        Number of units available for collection.
    freshness_score : float
        CNN freshness score at the time of the alert.
    freshness_stage : str
        Qualitative stage (e.g. 'near_expiry').
    collection_window_hours : int
        Hours from *assessed_at* within which collection must occur.
    assessed_at : datetime
        UTC timestamp when the assessment was made.
    collection_deadline : datetime
        Derived: assessed_at + collection_window_hours (set automatically).
    store_id : str
        Identifier of the store generating the alert.
    """
    item_id: str
    category: str
    quantity: int
    freshness_score: float
    freshness_stage: str
    collection_window_hours: int = FOODBANK_COLLECTION_WINDOW_HOURS
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    store_id: str = "store-001"

    @property
    def collection_deadline(self) -> datetime:
        """Latest time the food bank can collect the item."""
        return self.assessed_at + timedelta(hours=self.collection_window_hours)

    def to_api_payload(self) -> dict:
        """Serialise to the format expected by the food-bank REST API."""
        return {
            "item_id":                self.item_id,
            "category":               self.category,
            "quantity":               self.quantity,
            "freshness_score":        round(self.freshness_score, 2),
            "freshness_stage":        self.freshness_stage,
            "collection_window_hours": self.collection_window_hours,
            "assessed_at":            self.assessed_at.isoformat(),
            "collection_deadline":    self.collection_deadline.isoformat(),
            "store_id":               self.store_id,
        }


@dataclass
class AlertResponse:
    """Response from the food-bank API after sending an alert."""
    success: bool
    alert_id: Optional[str] = None
    message: str = ""
    partners_notified: int = 0


# ---------------------------------------------------------------------------
# Food-bank API client
# ---------------------------------------------------------------------------

class FoodBankClient:
    """
    REST client for the food-bank partner notification API.

    Parameters
    ----------
    base_url : str, optional
        API base URL.  Defaults to config.FOODBANK_API_BASE_URL.
    api_version : str, optional
        API version string.  Defaults to config.FOODBANK_API_VERSION.
    api_key : str, optional
        Bearer token / API key for authentication.
    timeout : int
        HTTP request timeout in seconds (default 10).
    session : requests.Session, optional
        Injectable session for testing.
    """

    def __init__(
        self,
        base_url: str = FOODBANK_API_BASE_URL,
        api_version: str = FOODBANK_API_VERSION,
        api_key: Optional[str] = None,
        timeout: int = 10,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._api_key = api_key
        self._timeout = timeout
        self._session = session or requests.Session()
        if self._api_key:
            self._session.headers.update(
                {"Authorization": f"Bearer {self._api_key}"}
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, alert: CollectionAlert) -> bool:
        """
        POST a collection alert to the food-bank partner API.

        Parameters
        ----------
        alert : CollectionAlert

        Returns
        -------
        bool
            ``True`` if the API accepted the alert.
        """
        result = self._post_alert(alert)
        return result.success

    def send_alert_detailed(self, alert: CollectionAlert) -> AlertResponse:
        """
        POST a collection alert and return the full ``AlertResponse``.

        Parameters
        ----------
        alert : CollectionAlert

        Returns
        -------
        AlertResponse
        """
        return self._post_alert(alert)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_alert(self, alert: CollectionAlert) -> AlertResponse:
        endpoint = (
            f"{self._base_url}/api/{self._api_version}/alerts/collection"
        )
        payload = alert.to_api_payload()
        try:
            response = self._session.post(
                endpoint, json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            data = response.json()
            result = AlertResponse(
                success=True,
                alert_id=data.get("alert_id"),
                message=data.get("message", "Alert accepted"),
                partners_notified=data.get("partners_notified", 0),
            )
            logger.info(
                "Food bank alerted: item=%s qty=%d alert_id=%s partners=%d",
                alert.item_id,
                alert.quantity,
                result.alert_id,
                result.partners_notified,
            )
            return result
        except RequestException as exc:
            logger.error(
                "Food bank notification failed for item %s: %s",
                alert.item_id,
                exc,
            )
            return AlertResponse(success=False, message=str(exc))

    def list_partners(self) -> list[dict]:
        """
        Retrieve the list of registered food-bank partner organisations.

        Returns
        -------
        list[dict]
            Each entry contains at minimum 'partner_id', 'name', and
            'contact_email'.
        """
        endpoint = f"{self._base_url}/api/{self._api_version}/partners"
        try:
            response = self._session.get(endpoint, timeout=self._timeout)
            response.raise_for_status()
            return response.json().get("partners", [])
        except RequestException as exc:
            logger.error("Failed to list food bank partners: %s", exc)
            return []
