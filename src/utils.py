"""
utils.py — Shared utility functions for the Hotel Analytics project.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROFILING_DIR = PROJECT_ROOT / "profiling"
REPORTS_DIR = PROJECT_ROOT / "reports"
REVIEW_JSON_PATH = PROJECT_ROOT / "review.json"

DB_PATH = DATA_DIR / "reviews.db"
SAMPLE_DB_PATH = DATA_DIR / "reviews_sample.db"

RATING_COLUMNS = [
    "rating_service",
    "rating_cleanliness",
    "rating_overall",
    "rating_value",
    "rating_location",
    "rating_sleep_quality",
    "rating_rooms",
]

RATING_LABELS = {
    "rating_service": "Service",
    "rating_cleanliness": "Cleanliness",
    "rating_overall": "Overall",
    "rating_value": "Value",
    "rating_location": "Location",
    "rating_sleep_quality": "Sleep Quality",
    "rating_rooms": "Rooms",
}

# Latest 5 years available in the dataset
MIN_YEAR = 2008
MAX_YEAR = 2012


# Helper Functions

def get_db_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and foreign keys enabled."""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def parse_date(date_str: str) -> datetime | None:
    """Parse date strings like 'December 17, 2012' → datetime object."""
    try:
        return datetime.strptime(date_str.strip(), "%B %d, %Y")
    except (ValueError, AttributeError):
        return None


def get_year(date_str: str) -> int | None:
    """Extract year as integer from a date string like 'December 17, 2012'."""
    try:
        parts = date_str.strip().split(", ")
        return int(parts[-1])
    except (ValueError, AttributeError, IndexError):
        return None


def get_month_year(date_str: str) -> str | None:
    """Extract 'YYYY-MM' from a date string like 'December 17, 2012'."""
    dt = parse_date(date_str)
    if dt:
        return dt.strftime("%Y-%m")
    return None
