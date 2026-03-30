"""
Tests for freshsight.integrations.foodbank_client
"""

import pytest
import requests
from datetime import datetime, timezone
from unittest.mock import MagicMock

from freshsight.integrations.foodbank_client import (
    FoodBankClient,
    CollectionAlert,
    AlertResponse,
)
from freshsight.config import FOODBANK_COLLECTION_WINDOW_HOURS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(**kwargs) -> CollectionAlert:
    defaults = {
        "item_id":         "5012345678900",
        "category":        "strawberry",
        "quantity":        8,
        "freshness_score": 35.0,
        "freshness_stage": "near_expiry",
    }
    defaults.update(kwargs)
    return CollectionAlert(**defaults)


def _mock_session(status_code: int = 200, json_data: dict = None) -> MagicMock:
    if json_data is None:
        json_data = {"alert_id": "alert-abc123", "message": "OK", "partners_notified": 2}
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    else:
        response.raise_for_status = MagicMock()
    session = MagicMock()
    session.post.return_value = response
    session.get.return_value = response
    return session


# ---------------------------------------------------------------------------
# CollectionAlert
# ---------------------------------------------------------------------------

class TestCollectionAlert:
    def test_collection_deadline_is_after_assessed_at(self):
        alert = _make_alert()
        assert alert.collection_deadline > alert.assessed_at

    def test_collection_deadline_offset(self):
        from datetime import timedelta
        alert = _make_alert(collection_window_hours=4)
        delta = alert.collection_deadline - alert.assessed_at
        assert delta == timedelta(hours=4)

    def test_to_api_payload_contains_required_fields(self):
        alert = _make_alert()
        payload = alert.to_api_payload()
        required = {
            "item_id", "category", "quantity", "freshness_score",
            "freshness_stage", "collection_window_hours",
            "assessed_at", "collection_deadline", "store_id",
        }
        assert required.issubset(set(payload.keys()))

    def test_payload_quantity_matches(self):
        alert = _make_alert(quantity=15)
        payload = alert.to_api_payload()
        assert payload["quantity"] == 15

    def test_payload_score_rounded(self):
        alert = _make_alert(freshness_score=35.12345)
        payload = alert.to_api_payload()
        assert payload["freshness_score"] == pytest.approx(35.12, abs=0.01)

    def test_default_window_from_config(self):
        alert = _make_alert()
        assert alert.collection_window_hours == FOODBANK_COLLECTION_WINDOW_HOURS


# ---------------------------------------------------------------------------
# FoodBankClient.send_alert
# ---------------------------------------------------------------------------

class TestFoodBankClientSendAlert:
    def test_returns_true_on_success(self):
        client = FoodBankClient(session=_mock_session(200))
        assert client.send_alert(_make_alert()) is True

    def test_returns_false_on_http_error(self):
        client = FoodBankClient(session=_mock_session(500))
        assert client.send_alert(_make_alert()) is False

    def test_returns_false_on_connection_error(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("no route")
        client = FoodBankClient(session=session)
        assert client.send_alert(_make_alert()) is False

    def test_post_called_with_correct_endpoint(self):
        session = _mock_session(200)
        client = FoodBankClient(
            base_url="https://fb-api.test",
            api_version="v1",
            session=session,
        )
        client.send_alert(_make_alert())
        call_url = session.post.call_args[0][0]
        assert "alerts/collection" in call_url

    def test_payload_sent_as_json(self):
        session = _mock_session(200)
        client = FoodBankClient(session=session)
        alert = _make_alert(item_id="TEST-ITEM", quantity=3)
        client.send_alert(alert)
        payload = session.post.call_args[1]["json"]
        assert payload["item_id"] == "TEST-ITEM"
        assert payload["quantity"] == 3


# ---------------------------------------------------------------------------
# FoodBankClient.send_alert_detailed
# ---------------------------------------------------------------------------

class TestFoodBankClientSendAlertDetailed:
    def test_returns_alert_response(self):
        client = FoodBankClient(session=_mock_session(200))
        result = client.send_alert_detailed(_make_alert())
        assert isinstance(result, AlertResponse)

    def test_success_response_has_alert_id(self):
        client = FoodBankClient(session=_mock_session(200))
        result = client.send_alert_detailed(_make_alert())
        assert result.success is True
        assert result.alert_id == "alert-abc123"
        assert result.partners_notified == 2

    def test_failure_response_has_success_false(self):
        client = FoodBankClient(session=_mock_session(500))
        result = client.send_alert_detailed(_make_alert())
        assert result.success is False


# ---------------------------------------------------------------------------
# FoodBankClient.list_partners
# ---------------------------------------------------------------------------

class TestFoodBankClientListPartners:
    def test_returns_list_on_success(self):
        partners_data = {"partners": [{"partner_id": "p1", "name": "City Food Bank"}]}
        session = _mock_session(200, json_data=partners_data)
        client = FoodBankClient(session=session)
        result = client.list_partners()
        assert isinstance(result, list)
        assert result[0]["partner_id"] == "p1"

    def test_returns_empty_list_on_error(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError()
        client = FoodBankClient(session=session)
        result = client.list_partners()
        assert result == []


# ---------------------------------------------------------------------------
# API key authentication header
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def test_auth_header_set_when_api_key_provided(self):
        session = _mock_session(200)
        client = FoodBankClient(api_key="secret-key-123", session=session)
        session.headers.update.assert_called_once_with(
            {"Authorization": "Bearer secret-key-123"}
        )
