"""
FreshSight AI – Central Store API Server

Provides a REST API accessible to:
  • Store managers  (dashboard endpoints – all assessments, action history)
  • Food bank partners  (partner portal – their own alerts, collection status)

The server receives assessment results from the edge computing units, stores
them in memory (or a real database in production), and serves query endpoints.

Run
---
  python -m freshsight.api.server
  # or via main.py
"""

from __future__ import annotations

import logging
import os

from flask import Flask

from freshsight.api.routes.dashboard import dashboard_bp
from freshsight.api.routes.foodbank import foodbank_bp
from freshsight.api.routes.ingest import ingest_bp
from freshsight.config import STORE_SERVER_HOST, STORE_SERVER_PORT, STORE_SERVER_DEBUG

logger = logging.getLogger(__name__)


def create_app(test_config: dict | None = None) -> Flask:
    """
    Application factory.

    Parameters
    ----------
    test_config : dict, optional
        Override Flask configuration (used in tests).

    Returns
    -------
    Flask
    """
    app = Flask(__name__, instance_relative_config=True)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-change-me-in-production"),
        DEBUG=STORE_SERVER_DEBUG,
    )

    if test_config is not None:
        app.config.update(test_config)

    # Register blueprints
    app.register_blueprint(ingest_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(foodbank_bp)

    logger.info("FreshSight AI API server created.")
    return app


def run_server() -> None:
    """Start the development server."""
    app = create_app()
    app.run(
        host=STORE_SERVER_HOST,
        port=STORE_SERVER_PORT,
        debug=STORE_SERVER_DEBUG,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
