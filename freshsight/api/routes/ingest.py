"""
FreshSight AI – Ingest Route

Edge computing units POST assessment results to this endpoint.  The server
then stores the result and (if needed) triggers further actions.
"""

from __future__ import annotations

from flask import Blueprint, request, jsonify

from freshsight.api import store as data_store
from freshsight.api.store import AssessmentRecord

ingest_bp = Blueprint("ingest", __name__, url_prefix="/api/v1/ingest")


@ingest_bp.route("/assessment", methods=["POST"])
def ingest_assessment():
    """
    Accept a freshness assessment result from an edge unit.

    Expected JSON body
    ------------------
    {
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
        "discounted_price":   1.60,
        "original_price":     2.00,
        "label_text":         "20% off – sell today",
        "foodbank_notified":  false
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    required = {"item_id", "category", "freshness_score", "recommended_action"}
    missing = required - set(data.keys())
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    record = AssessmentRecord(
        item_id=data["item_id"],
        category=data["category"],
        freshness_score=float(data["freshness_score"]),
        freshness_stage=data.get("freshness_stage", "unknown"),
        recommended_action=int(data["recommended_action"]),
        action_taken=int(data.get("action_taken", data["recommended_action"])),
        camera_id=data.get("camera_id", "unknown"),
        quantity=int(data.get("quantity", 1)),
        assessed_at=data.get("assessed_at", ""),
        discount_pct=data.get("discount_pct"),
        discounted_price=data.get("discounted_price"),
        original_price=data.get("original_price"),
        label_text=data.get("label_text"),
        foodbank_notified=bool(data.get("foodbank_notified", False)),
        alert_id=data.get("alert_id"),
    )
    data_store.add_record(record)

    return jsonify({"status": "ok", "record_id": record.record_id}), 201
