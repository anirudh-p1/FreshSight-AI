"""
FreshSight AI – Food Bank Partner Routes

Provides endpoints for registered food-bank partners to view collection
alerts and update collection status.
"""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, request, jsonify

from freshsight.api import store as data_store

foodbank_bp = Blueprint("foodbank", __name__, url_prefix="/api/v1/foodbank")


@foodbank_bp.route("/alerts", methods=["GET"])
def list_alerts():
    """
    Return all active food-bank collection alerts.

    Query parameters
    ----------------
    store_id : str, optional (reserved for future multi-store filtering)
    """
    records = data_store.get_foodbank_alerts()
    return jsonify({"alerts": [asdict(r) for r in records], "count": len(records)})


@foodbank_bp.route("/alerts/<record_id>", methods=["GET"])
def get_alert(record_id: str):
    """Return a single food-bank alert by record ID."""
    alerts = data_store.get_foodbank_alerts()
    match = next((r for r in alerts if r.record_id == record_id), None)
    if match is None:
        return jsonify({"error": "Alert not found"}), 404
    return jsonify(asdict(match))
