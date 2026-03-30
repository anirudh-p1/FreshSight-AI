"""
FreshSight AI – Configuration
Defines per-category freshness thresholds, action bands, and system settings.
"""

# ---------------------------------------------------------------------------
# Per-category freshness thresholds (score out of 100)
# ---------------------------------------------------------------------------
# FULL_PRICE_THRESHOLD  : score >= this → action 1 (sell at full price)
# DISCOUNT_THRESHOLD    : score >= this but < full_price → action 2 (discount)
# FOODBANK_THRESHOLD    : score < this → action 3 (food-bank alert)
#
# Items whose score falls between DISCOUNT_THRESHOLD and FULL_PRICE_THRESHOLD
# trigger Action 2.  Items whose score falls below FOODBANK_THRESHOLD trigger
# Action 3.  For simplicity both thresholds are stored per category; the
# discount band sits between them.

CATEGORY_THRESHOLDS: dict[str, dict[str, int]] = {
    "strawberry": {
        "full_price": 85,
        "discount": 70,
        "foodbank": 40,
    },
    "banana": {
        "full_price": 80,
        "discount": 60,
        "foodbank": 35,
    },
    "apple": {
        "full_price": 80,
        "discount": 60,
        "foodbank": 35,
    },
    "lettuce": {
        "full_price": 80,
        "discount": 65,
        "foodbank": 35,
    },
    "tomato": {
        "full_price": 80,
        "discount": 65,
        "foodbank": 35,
    },
    "root_vegetable": {
        "full_price": 75,
        "discount": 50,
        "foodbank": 25,
    },
    "default": {
        "full_price": 80,
        "discount": 60,
        "foodbank": 30,
    },
}

# ---------------------------------------------------------------------------
# Freshness stage labels (used during CNN training & for reporting)
# ---------------------------------------------------------------------------
FRESHNESS_STAGES: dict[str, tuple[int, int]] = {
    "fresh":      (85, 100),
    "moderate":   (60, 84),
    "near_expiry": (35, 59),
    "spoiled":    (0,  34),
}

# ---------------------------------------------------------------------------
# CNN model settings
# ---------------------------------------------------------------------------
CNN_INPUT_SIZE: tuple[int, int] = (224, 224)   # (width, height) in pixels
CNN_NUM_CLASSES: int = 4                        # fresh / moderate / near_expiry / spoiled
MODEL_TARGET_ACCURACY: float = 0.92            # design target ≥ 92 %

# ---------------------------------------------------------------------------
# Dynamic pricing settings
# ---------------------------------------------------------------------------
MAX_DISCOUNT_PCT: float = 0.60      # never discount more than 60 %
MIN_DISCOUNT_PCT: float = 0.05      # always discount at least 5 %

# ---------------------------------------------------------------------------
# Food-bank API settings (can be overridden via environment variables)
# ---------------------------------------------------------------------------
FOODBANK_API_BASE_URL: str = "https://foodbank-api.example.com"
FOODBANK_API_VERSION: str = "v1"
FOODBANK_COLLECTION_WINDOW_HOURS: int = 4   # hours from alert to latest pickup

# ---------------------------------------------------------------------------
# Electronic shelf label (ESL) API settings
# ---------------------------------------------------------------------------
ESL_API_BASE_URL: str = "http://esl-controller.local"
ESL_API_VERSION: str = "v1"

# ---------------------------------------------------------------------------
# Central store server settings
# ---------------------------------------------------------------------------
STORE_SERVER_HOST: str = "0.0.0.0"
STORE_SERVER_PORT: int = 5000
STORE_SERVER_DEBUG: bool = False
