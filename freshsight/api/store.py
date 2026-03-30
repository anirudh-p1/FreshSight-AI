"""
FreshSight AI – In-memory data store

Shared state for API routes (production deployments replace this with a
real database such as PostgreSQL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AssessmentRecord:
    """Persisted assessment result."""
    item_id: str
    category: str
    freshness_score: float
    freshness_stage: str
    recommended_action: int
    action_taken: int
    camera_id: str
    quantity: int
    assessed_at: str               # ISO-8601 string
    discount_pct: Optional[float] = None
    discounted_price: Optional[float] = None
    original_price: Optional[float] = None
    label_text: Optional[str] = None
    foodbank_notified: bool = False
    alert_id: Optional[str] = None
    record_id: str = field(default_factory=lambda: _next_id())


_counter = 0


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"rec-{_counter:06d}"


# Module-level in-memory store
_records: list[AssessmentRecord] = []


def add_record(record: AssessmentRecord) -> None:
    _records.append(record)


def get_all_records() -> list[AssessmentRecord]:
    return list(_records)


def get_records_by_action(action: int) -> list[AssessmentRecord]:
    return [r for r in _records if r.action_taken == action]


def get_records_by_category(category: str) -> list[AssessmentRecord]:
    return [r for r in _records if r.category == category]


def get_foodbank_alerts() -> list[AssessmentRecord]:
    return [r for r in _records if r.foodbank_notified]


def clear_records() -> None:
    """Clear all records (used in tests)."""
    global _counter
    _records.clear()
    _counter = 0
