"""
FreshSight AI – Main Entry Point

Demonstrates the end-to-end pipeline:
  1. Generate a synthetic camera image (replace with real camera capture).
  2. Assess freshness via the CNN model.
  3. Execute the appropriate automated action.

Run
---
  python main.py
  python main.py --server   # start the central API server
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from PIL import Image

from freshsight.core.freshness_engine import FreshnessEngine
from freshsight.core.action_engine import ActionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _make_synthetic_image(seed: int = 42) -> Image.Image:
    """
    Create a synthetic RGB image to stand in for a camera frame.

    In production, replace this with an image captured by the shelf camera.
    """
    rng = np.random.default_rng(seed)
    pixels = (rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    return Image.fromarray(pixels, mode="RGB")


def run_demo() -> None:
    """Run a demonstration of the FreshSight AI pipeline."""
    engine = FreshnessEngine()
    action_engine = ActionEngine()

    demo_items = [
        {"item_id": "5012345678900", "category": "strawberry", "seed": 1,  "price": 2.50, "qty": 12},
        {"item_id": "5012345678901", "category": "banana",     "seed": 42, "price": 1.20, "qty": 8},
        {"item_id": "5012345678902", "category": "root_vegetable", "seed": 7, "price": 0.80, "qty": 20},
        {"item_id": "5012345678903", "category": "lettuce",    "seed": 99, "price": 1.50, "qty": 6},
    ]

    print("\n" + "=" * 60)
    print("  FreshSight AI – Freshness Assessment Demo")
    print("=" * 60)

    for item in demo_items:
        image = _make_synthetic_image(seed=item["seed"])
        assessment = engine.assess_image(
            image=image,
            item_id=item["item_id"],
            category=item["category"],
            camera_id="cam-demo",
            quantity=item["qty"],
        )
        result = action_engine.execute(
            assessment=assessment,
            original_price=item["price"],
            hours_to_expiry=6,
        )

        print(f"\nItem:     {item['item_id']} ({item['category']})")
        print(f"Score:    {assessment.freshness_score:.1f}/100  [{assessment.freshness_stage}]")
        print(f"Action:   {result.action_taken}")
        print(f"Outcome:  {result.message}")

    print("\n" + "=" * 60)


def run_server() -> None:
    """Start the central store API server."""
    from freshsight.api.server import run_server as _run
    logger.info("Starting FreshSight AI API server…")
    _run()


def main() -> None:
    parser = argparse.ArgumentParser(description="FreshSight AI")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the central store API server instead of running the demo.",
    )
    args = parser.parse_args()

    if args.server:
        run_server()
    else:
        run_demo()


if __name__ == "__main__":
    main()
