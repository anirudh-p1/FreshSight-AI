"""
Tests for freshsight.integrations.shelf_label
"""

import pytest
import requests
from unittest.mock import MagicMock, patch

from freshsight.integrations.shelf_label import ShelfLabelClient, LabelUpdate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_update(**kwargs) -> LabelUpdate:
    defaults = {
        "item_id":          "5012345678900",
        "original_price":   2.50,
        "discounted_price": 2.00,
        "discount_pct":     0.20,
        "label_text":       "20% off – sell today",
        "freshness_score":  72.4,
    }
    defaults.update(kwargs)
    return LabelUpdate(**defaults)


def _mock_session(status_code=200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock(
        side_effect=None if status_code < 400
        else requests.exceptions.HTTPError(response=response)
    )
    session = MagicMock()
    session.put.return_value = response
    session.get.return_value = response
    return session


# ---------------------------------------------------------------------------
# LabelUpdate dataclass
# ---------------------------------------------------------------------------

class TestLabelUpdate:
    def test_all_fields_set(self):
        upd = _make_update()
        assert upd.item_id == "5012345678900"
        assert upd.original_price == 2.50
        assert upd.discounted_price == 2.00
        assert upd.discount_pct == 0.20


# ---------------------------------------------------------------------------
# ShelfLabelClient.update
# ---------------------------------------------------------------------------

class TestShelfLabelClientUpdate:
    def test_returns_true_on_success(self):
        client = ShelfLabelClient(session=_mock_session(200))
        assert client.update(_make_update()) is True

    def test_returns_false_on_http_error(self):
        session = _mock_session(500)
        session.put.return_value.raise_for_status.side_effect = (
            requests.exceptions.HTTPError()
        )
        client = ShelfLabelClient(session=session)
        assert client.update(_make_update()) is False

    def test_returns_false_on_connection_error(self):
        session = MagicMock()
        session.put.side_effect = requests.exceptions.ConnectionError("unreachable")
        client = ShelfLabelClient(session=session)
        assert client.update(_make_update()) is False

    def test_put_called_with_correct_item_id_in_url(self):
        session = _mock_session(200)
        client = ShelfLabelClient(
            base_url="http://esl.local", api_version="v1", session=session
        )
        client.update(_make_update(item_id="BARCODE-123"))
        call_url = session.put.call_args[0][0]
        assert "BARCODE-123" in call_url

    def test_payload_contains_discount_pct(self):
        session = _mock_session(200)
        client = ShelfLabelClient(session=session)
        client.update(_make_update(discount_pct=0.30))
        payload = session.put.call_args[1]["json"]
        assert payload["discount_pct"] == pytest.approx(0.30, abs=0.0001)

    def test_payload_contains_label_text(self):
        session = _mock_session(200)
        client = ShelfLabelClient(session=session)
        client.update(_make_update(label_text="50% off – use today"))
        payload = session.put.call_args[1]["json"]
        assert payload["label_text"] == "50% off – use today"


# ---------------------------------------------------------------------------
# ShelfLabelClient.reset_to_full_price
# ---------------------------------------------------------------------------

class TestShelfLabelClientReset:
    def test_returns_true_on_success(self):
        client = ShelfLabelClient(session=_mock_session(200))
        assert client.reset_to_full_price("ITEM-001", 2.50) is True

    def test_returns_false_on_error(self):
        session = MagicMock()
        session.put.side_effect = requests.exceptions.ConnectionError()
        client = ShelfLabelClient(session=session)
        assert client.reset_to_full_price("ITEM-001", 2.50) is False

    def test_payload_has_zero_discount(self):
        session = _mock_session(200)
        client = ShelfLabelClient(session=session)
        client.reset_to_full_price("ITEM-001", 3.99)
        payload = session.put.call_args[1]["json"]
        assert payload["discount_pct"] == 0.0
        assert payload["discounted_price"] == pytest.approx(3.99)
