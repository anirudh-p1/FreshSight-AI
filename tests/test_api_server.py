"""
Tests for the Flask API server (ingest, dashboard, foodbank routes)
"""

import pytest
import json

from freshsight.api.server import create_app
from freshsight.api import store as data_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    with app.app_context():
        data_store.clear_records()
        yield app
        data_store.clear_records()


@pytest.fixture
def client(app):
    return app.test_client()


def _ingest_payload(**overrides) -> dict:
    base = {
        "item_id":            "5012345678900",
        "category":           "strawberry",
        "freshness_score":    72.4,
        "freshness_stage":    "moderate",
        "recommended_action": 2,
        "action_taken":       2,
        "camera_id":          "cam-shelf-3",
        "quantity":           8,
        "assessed_at":        "2024-01-15T10:30:00+00:00",
        "discount_pct":       0.20,
        "discounted_price":   2.00,
        "original_price":     2.50,
        "label_text":         "20% off – sell today",
        "foodbank_notified":  False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    def test_post_valid_assessment_returns_201(self, client):
        response = client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload()),
            content_type="application/json",
        )
        assert response.status_code == 201

    def test_response_contains_record_id(self, client):
        response = client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload()),
            content_type="application/json",
        )
        data = response.get_json()
        assert "record_id" in data
        assert data["status"] == "ok"

    def test_missing_required_field_returns_400(self, client):
        payload = _ingest_payload()
        del payload["freshness_score"]
        response = client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_empty_body_returns_400(self, client):
        response = client.post(
            "/api/v1/ingest/assessment",
            data="",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_multiple_ingests_stored(self, client):
        for i in range(3):
            client.post(
                "/api/v1/ingest/assessment",
                data=json.dumps(_ingest_payload(item_id=f"ITEM-{i}")),
                content_type="application/json",
            )
        assert len(data_store.get_all_records()) == 3


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------

class TestDashboardSummary:
    def test_summary_returns_200(self, client):
        response = client.get("/api/v1/dashboard/summary")
        assert response.status_code == 200

    def test_summary_has_expected_keys(self, client):
        response = client.get("/api/v1/dashboard/summary")
        data = response.get_json()
        for key in ["total_assessments", "action_1_count", "action_2_count",
                    "action_3_count", "foodbank_alerts"]:
            assert key in data

    def test_summary_counts_correctly(self, client):
        for action in [1, 2, 3]:
            client.post(
                "/api/v1/ingest/assessment",
                data=json.dumps(_ingest_payload(action_taken=action, recommended_action=action)),
                content_type="application/json",
            )
        response = client.get("/api/v1/dashboard/summary")
        data = response.get_json()
        assert data["total_assessments"] == 3
        assert data["action_1_count"] == 1
        assert data["action_2_count"] == 1
        assert data["action_3_count"] == 1


class TestDashboardAssessments:
    def test_list_all_assessments_returns_200(self, client):
        response = client.get("/api/v1/dashboard/assessments")
        assert response.status_code == 200

    def test_filter_by_action(self, client):
        client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload(action_taken=1, recommended_action=1)),
            content_type="application/json",
        )
        client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload(action_taken=2, recommended_action=2)),
            content_type="application/json",
        )
        response = client.get("/api/v1/dashboard/assessments?action=1")
        data = response.get_json()
        assert all(r["action_taken"] == 1 for r in data["assessments"])

    def test_filter_by_category(self, client):
        client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload(category="banana")),
            content_type="application/json",
        )
        response = client.get("/api/v1/dashboard/assessments?category=banana")
        data = response.get_json()
        assert all(r["category"] == "banana" for r in data["assessments"])

    def test_get_single_assessment(self, client):
        resp = client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload()),
            content_type="application/json",
        )
        record_id = resp.get_json()["record_id"]
        response = client.get(f"/api/v1/dashboard/assessments/{record_id}")
        assert response.status_code == 200
        data = response.get_json()
        assert data["record_id"] == record_id

    def test_unknown_record_id_returns_404(self, client):
        response = client.get("/api/v1/dashboard/assessments/nonexistent-id")
        assert response.status_code == 404

    def test_discounted_items_endpoint(self, client):
        client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload(action_taken=2, recommended_action=2)),
            content_type="application/json",
        )
        response = client.get("/api/v1/dashboard/discounted")
        data = response.get_json()
        assert "discounted_items" in data
        assert data["count"] >= 1


# ---------------------------------------------------------------------------
# Food bank endpoints
# ---------------------------------------------------------------------------

class TestFoodbankEndpoints:
    def test_list_alerts_returns_200(self, client):
        response = client.get("/api/v1/foodbank/alerts")
        assert response.status_code == 200

    def test_foodbank_alert_appears_in_list(self, client):
        client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload(
                action_taken=3,
                recommended_action=3,
                foodbank_notified=True,
            )),
            content_type="application/json",
        )
        response = client.get("/api/v1/foodbank/alerts")
        data = response.get_json()
        assert data["count"] == 1

    def test_get_single_alert(self, client):
        resp = client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload(
                action_taken=3,
                recommended_action=3,
                foodbank_notified=True,
            )),
            content_type="application/json",
        )
        record_id = resp.get_json()["record_id"]
        response = client.get(f"/api/v1/foodbank/alerts/{record_id}")
        assert response.status_code == 200

    def test_non_foodbank_item_not_in_alerts(self, client):
        client.post(
            "/api/v1/ingest/assessment",
            data=json.dumps(_ingest_payload(action_taken=1, foodbank_notified=False)),
            content_type="application/json",
        )
        response = client.get("/api/v1/foodbank/alerts")
        data = response.get_json()
        assert data["count"] == 0

    def test_unknown_alert_id_returns_404(self, client):
        response = client.get("/api/v1/foodbank/alerts/does-not-exist")
        assert response.status_code == 404
