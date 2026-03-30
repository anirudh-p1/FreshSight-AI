"""
FreshSight AI – Store Manager Dashboard Routes

Provides endpoints for store managers to monitor freshness assessments,
review discounted items, and check food-bank alert history.
"""

from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, request, jsonify

from freshsight.api import store as data_store

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/v1/dashboard")


@dashboard_bp.route("/assessments", methods=["GET"])
def list_assessments():
    """
    Return all recorded assessments.

    Query parameters
    ----------------
    action : int, optional
        Filter by action taken (1, 2, or 3).
    category : str, optional
        Filter by produce category.
    """
    action = request.args.get("action", type=int)
    category = request.args.get("category", type=str)

    if action is not None:
        records = data_store.get_records_by_action(action)
    elif category is not None:
        records = data_store.get_records_by_category(category)
    else:
        records = data_store.get_all_records()

    return jsonify({"assessments": [asdict(r) for r in records], "count": len(records)})


@dashboard_bp.route("/assessments/<record_id>", methods=["GET"])
def get_assessment(record_id: str):
    """Return a single assessment by record ID."""
    records = data_store.get_all_records()
    match = next((r for r in records if r.record_id == record_id), None)
    if match is None:
        return jsonify({"error": "Assessment not found"}), 404
    return jsonify(asdict(match))


@dashboard_bp.route("/summary", methods=["GET"])
def summary():
    """
    Return a high-level summary of current shelf status.

    Response
    --------
    {
        "total_assessments": 120,
        "action_1_count": 80,
        "action_2_count": 30,
        "action_3_count": 10,
        "foodbank_alerts": 10
    }
    """
    all_records = data_store.get_all_records()
    return jsonify({
        "total_assessments": len(all_records),
        "action_1_count":    sum(1 for r in all_records if r.action_taken == 1),
        "action_2_count":    sum(1 for r in all_records if r.action_taken == 2),
        "action_3_count":    sum(1 for r in all_records if r.action_taken == 3),
        "foodbank_alerts":   sum(1 for r in all_records if r.foodbank_notified),
    })


@dashboard_bp.route("/discounted", methods=["GET"])
def list_discounted():
    """Return all currently discounted items (action 2)."""
    records = data_store.get_records_by_action(2)
    return jsonify({"discounted_items": [asdict(r) for r in records], "count": len(records)})
