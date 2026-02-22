"""
data_processing.py — Load review.json, filter, and build SQLite databases.

Usage:
    python src/data_processing.py            # builds full DB + sample DB
    python src/data_processing.py --sample   # builds only the 5 000-row sample DB
"""

import json
import random
import sqlite3
import sys
import time
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from textblob import TextBlob
from typing import Generator
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("[WARN] langdetect not installed. Please install it using `pip install langdetect`.")
    detect = None
    LangDetectException = Exception

# Allow running both as module and as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.utils as utils
from src.utils import (
    DATA_DIR,
    DB_PATH,
    MAX_YEAR,
    MIN_YEAR,
    REVIEW_JSON_PATH,
    SAMPLE_DB_PATH,
    get_year,
    parse_date,
)

SCHEMA_PATH = DATA_DIR / "data_schema.sql"
SAMPLE_SIZE = 10000  # slightly more than 5 000 required
BATCH_SIZE = 10_000


# Schema helpers

def _create_schema(conn: sqlite3.Connection) -> None:
    """Execute the DDL from data_schema.sql."""
    schema_sql = SCHEMA_PATH.read_text()
    conn.executescript(schema_sql)


def _drop_indexes(conn: sqlite3.Connection) -> None:
    """Drop indexes before bulk insert for speed."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    ).fetchall()
    for (name,) in rows:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.commit()


def _create_indexes(conn: sqlite3.Connection) -> None:
    """Re-create indexes after bulk insert."""
    schema_sql = SCHEMA_PATH.read_text()
    for line in schema_sql.splitlines():
        if line.strip().upper().startswith("CREATE INDEX"):
            conn.execute(line)
    conn.commit()


# Parsing one JSON line

def _parse_review(raw: dict) -> dict | None:
    """
    Parse a single raw JSON record into (hotel_row, author_row, review_row).
    Returns None if the record should be skipped.
    """
    year = get_year(raw.get("date", ""))
    if year is None or year < MIN_YEAR or year > MAX_YEAR:
        return None

    ratings = raw.get("ratings", {})
    author = raw.get("author", {})

    # Filter: Must have overall rating
    if ratings.get("overall") is None:
        return None

    # Filter: Must have substantial text content (> 20 chars)
    text = raw.get("text", "").strip()
    if not text or len(text) < 20:
        return None

    # Fast Language Filter
    if detect is not None:
        try:
            if detect(text) != 'en':
                return None # Skip non-English reviews
        except LangDetectException:
            return None # Skip reviews where language detection fails

    dt = parse_date(raw.get("date", ""))
    date_iso = dt.strftime("%Y-%m-%d") if dt else None
    month = dt.month if dt else None

    # Sentiment Analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return {
        "hotel_id": raw.get("offering_id"),
        "author": {
            "author_id": author.get("id"),
            "username": author.get("username"),
            "location": author.get("location"),
            "num_cities": author.get("num_cities"),
            "num_helpful_votes": author.get("num_helpful_votes"),
            "num_reviews": author.get("num_reviews"),
            "num_type_reviews": author.get("num_type_reviews"),
        },
        "review": {
            "review_id": raw.get("id"),
            "hotel_id": raw.get("offering_id"),
            "author_id": author.get("id"),
            "title": raw.get("title"),
            "text": text,
            "date": raw.get("date"),
            "date_parsed": date_iso,
            "year": year,
            "month": month,
            "date_stayed": raw.get("date_stayed"),
            "rating_service": ratings.get("service"),
            "rating_cleanliness": ratings.get("cleanliness"),
            "rating_overall": ratings.get("overall"),
            "rating_value": ratings.get("value"),
            "rating_location": ratings.get("location"),
            "rating_sleep_quality": ratings.get("sleep_quality"),
            "rating_rooms": ratings.get("rooms"),
            "num_helpful_votes": raw.get("num_helpful_votes", 0),
            "via_mobile": 1 if raw.get("via_mobile") else 0,
            "sentiment_polarity": polarity,
            "sentiment_subjectivity": subjectivity,
        },
    }


# Bulk insert

def _insert_batch(
    conn: sqlite3.Connection,
    hotels: dict,
    authors: dict,
    reviews: list[dict],
) -> None:
    """Insert a batch of parsed records into the database."""
    # Hotels (upsert)
    hotel_rows = [(hid,) for hid in hotels if hid is not None]
    conn.executemany(
        "INSERT OR IGNORE INTO hotels (hotel_id) VALUES (?)", hotel_rows
    )

    # Authors (upsert)
    author_rows = [
        (
            a["author_id"], a["username"], a["location"],
            a["num_cities"], a["num_helpful_votes"],
            a["num_reviews"], a["num_type_reviews"],
        )
        for a in authors.values()
        if a["author_id"] is not None
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO authors
           (author_id, username, location, num_cities,
            num_helpful_votes, num_reviews, num_type_reviews)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        author_rows,
    )

    # Reviews
    review_rows = [
        (
            r["review_id"], r["hotel_id"], r["author_id"],
            r["title"], r["text"], r["date"], r["date_parsed"],
            r["year"], r["month"], r["date_stayed"],
            r["rating_service"], r["rating_cleanliness"],
            r["rating_overall"], r["rating_value"],
            r["rating_location"], r["rating_sleep_quality"],
            r["rating_rooms"], r["num_helpful_votes"], r["via_mobile"],
            r["sentiment_polarity"], r["sentiment_subjectivity"],
        )
        for r in reviews
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO reviews
           (review_id, hotel_id, author_id, title, text, date, date_parsed,
            year, month, date_stayed, rating_service, rating_cleanliness,
            rating_overall, rating_value, rating_location,
            rating_sleep_quality, rating_rooms, num_helpful_votes, via_mobile,
            sentiment_polarity, sentiment_subjectivity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        review_rows,
    )
    conn.commit()


# Main pipeline

def build_database(db_path: Path = DB_PATH, limit: int | None = 100_000) -> int:
    """
    Read review.json, filter to MIN_YEAR–MAX_YEAR, and write to SQLite.

    Args:
        db_path: Output database path.
        limit:   Max number of reviews to insert (None = all).

    Returns:
        Number of reviews inserted.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Remove existing DB so we start fresh
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    _create_schema(conn)
    _drop_indexes(conn)  # faster bulk inserts without indexes

    hotels_batch: dict[int, bool] = {}
    authors_batch: dict[str, dict] = {}
    reviews_batch: list[dict] = []
    total = 0

    t0 = time.time()
    print(f"[data_processing] Reading {REVIEW_JSON_PATH} …")
    with open(REVIEW_JSON_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = json.loads(line)
            parsed = _parse_review(raw)
            if parsed is None:
                continue

            hotels_batch[parsed["hotel_id"]] = True
            aid = parsed["author"]["author_id"]
            if aid and aid not in authors_batch:
                authors_batch[aid] = parsed["author"]
            reviews_batch.append(parsed["review"])
            total += 1

            if len(reviews_batch) >= BATCH_SIZE:
                _insert_batch(conn, hotels_batch, authors_batch, reviews_batch)
                hotels_batch.clear()
                authors_batch.clear()
                reviews_batch.clear()
                print(f"  … {total:,} reviews inserted ({line_no:,} lines scanned)")

            if limit and total >= limit:
                break

    # flush remaining
    if reviews_batch:
        _insert_batch(conn, hotels_batch, authors_batch, reviews_batch)

    print(f"[data_processing] Creating indexes …")
    _create_indexes(conn)

    # Update hotel review counts
    conn.execute("""
        UPDATE hotels SET num_reviews = (
            SELECT COUNT(*) FROM reviews WHERE reviews.hotel_id = hotels.hotel_id
        )
    """)
    conn.commit()

    print(f"[data_processing] Filtering hotels with < 50 reviews and cleaning up …")
    conn.execute("DELETE FROM hotels WHERE num_reviews < 50")
    conn.execute("DELETE FROM reviews WHERE hotel_id NOT IN (SELECT hotel_id FROM hotels)")
    conn.execute("DELETE FROM authors WHERE author_id NOT IN (SELECT author_id FROM reviews)")
    conn.commit()

    # Get final count
    total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]

    elapsed = time.time() - t0
    print(f"[data_processing] Done — {total:,} reviews in {elapsed:.1f}s → {db_path}")

    # Compute credibility weights for every review
    compute_review_weights(conn)

    conn.close()
    return total


def ingest_uploaded_reviews(
    records: list[dict],
    db_path: Path = DB_PATH,
    progress_callback=None,
) -> dict:
    """
    Ingest a list of raw JSON review dicts (already parsed from an uploaded file)
    into an existing SQLite database, appending new records without wiping existing data.

    Args:
        records:           List of raw dicts matching the source JSON schema.
        db_path:           Target SQLite database path (must already have schema).
        progress_callback: Optional callable(done: int, total: int) for UI progress updates.

    Returns:
        dict with keys: inserted, skipped, total_reviews_in_db
    """
    if not db_path.exists():
        # Bootstrap the schema if this is a fresh DB
        conn = sqlite3.connect(str(db_path))
        _create_schema(conn)
        conn.close()

    conn = sqlite3.connect(str(db_path))

    hotels_batch: dict = {}
    authors_batch: dict = {}
    reviews_batch: list = []
    inserted = 0
    skipped = 0
    total = len(records)

    for i, raw in enumerate(records):
        parsed = _parse_review(raw)
        if parsed is None:
            skipped += 1
        else:
            hotels_batch[parsed["hotel_id"]] = True
            aid = parsed["author"]["author_id"]
            if aid and aid not in authors_batch:
                authors_batch[aid] = parsed["author"]
            reviews_batch.append(parsed["review"])
            inserted += 1

        # Flush in batches for responsiveness
        if len(reviews_batch) >= BATCH_SIZE:
            _insert_batch(conn, hotels_batch, authors_batch, reviews_batch)
            hotels_batch.clear()
            authors_batch.clear()
            reviews_batch.clear()

        if progress_callback:
            progress_callback(i + 1, total)

    # Flush remaining
    if reviews_batch:
        _insert_batch(conn, hotels_batch, authors_batch, reviews_batch)

    # Update hotel review counts
    conn.execute("""
        UPDATE hotels SET num_reviews = (
            SELECT COUNT(*) FROM reviews WHERE reviews.hotel_id = hotels.hotel_id
        )
    """)
    conn.commit()

    # Recompute credibility weights to include the new records
    compute_review_weights(conn)

    total_in_db = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    conn.close()

    return {"inserted": inserted, "skipped": skipped, "total_reviews_in_db": total_in_db}


def compute_review_weights(conn: sqlite3.Connection) -> None:
    """
    Compute a credibility weight for each review based on 4 signals:
      1. Helpful votes (community validation)
      2. Author experience (num_reviews, num_cities)
      3. Text length (detail level)
      4. Objectivity (1 - subjectivity)
    
    Weight formula: 1.0 (baseline) + sum of 4 normalized boosts.
    Results stored in reviews.review_weight column.
    """
    import math

    print("[data_processing] Computing review credibility weights …")
    
    # Fetch review data with author stats via JOIN
    df = pd.read_sql_query("""
        SELECT r.review_id,
               COALESCE(r.num_helpful_votes, 0) AS helpful,
               COALESCE(a.num_reviews, 0)        AS author_revs,
               COALESCE(LENGTH(r.text), 0)       AS text_len,
               COALESCE(r.sentiment_subjectivity, 0.5) AS subj
        FROM reviews r
        LEFT JOIN authors a ON r.author_id = a.author_id
    """, conn)

    if df.empty:
        print("[data_processing] No reviews to weight.")
        return

    # Log-scaled normalization denominators (avoid division by zero)
    log_max_helpful = max(np.log1p(df["helpful"].max()), 1e-9)
    log_max_author  = max(np.log1p(df["author_revs"].max()), 1e-9)

    # Vectorized weight computation
    w = (1.0
         + np.log1p(df["helpful"])     / log_max_helpful      # helpful boost
         + np.log1p(df["author_revs"]) / log_max_author       # experience boost
         + np.minimum(df["text_len"] / 500.0, 1.0)            # length boost (capped)
         + (1.0 - df["subj"])                                 # objectivity boost
    )

    # Write weights back to DB
    conn.executemany(
        "UPDATE reviews SET review_weight = ? WHERE review_id = ?",
        list(zip(w.tolist(), df["review_id"].tolist()))
    )
    conn.commit()

    print(f"[data_processing] Weights computed — min={w.min():.2f}, avg={w.mean():.2f}, max={w.max():.2f}")


def build_sample_database(
    source_db: Path = DB_PATH,
    sample_db: Path = SAMPLE_DB_PATH,
    sample_size: int = SAMPLE_SIZE,
) -> int:
    """
    Create a small sample DB from the full DB for TA testing.
    Samples `sample_size` reviews randomly.
    """
    if sample_db.exists():
        sample_db.unlink()

    src = sqlite3.connect(str(source_db))
    dst = sqlite3.connect(str(sample_db))

    # Create schema in destination
    _create_schema(dst)

    # Sample hotels instead of reviews
    # Get all valid hotels and shuffle them
    hotels = src.execute("SELECT hotel_id, num_reviews FROM hotels WHERE num_reviews >= 50").fetchall()
    random.shuffle(hotels)

    sampled_hotel_ids = []
    current_size = 0
    for hid, count in hotels:
        sampled_hotel_ids.append(hid)
        current_size += count
        if current_size >= sample_size:
            break

    if not sampled_hotel_ids:
        print("[data_processing] No valid hotels found for sample.")
        src.close()
        dst.close()
        return 0

    # Process in chunks of 900 to avoid SQLite variable limits
    chunk_size = 900
    
    # 1. Copy related hotels
    for i in range(0, len(sampled_hotel_ids), chunk_size):
        chunk = sampled_hotel_ids[i:i + chunk_size]
        ph = ",".join("?" * len(chunk))
        hotel_rows = src.execute(f"SELECT * FROM hotels WHERE hotel_id IN ({ph})", chunk).fetchall()
        dst.executemany("INSERT OR IGNORE INTO hotels (hotel_id, num_reviews) VALUES (?, ?)", hotel_rows)

    # 2. Copy related reviews
    cols = [d[0] for d in src.execute("SELECT * FROM reviews LIMIT 1").description]
    insert_sql = f"INSERT OR IGNORE INTO reviews ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    
    for i in range(0, len(sampled_hotel_ids), chunk_size):
        chunk = sampled_hotel_ids[i:i + chunk_size]
        ph = ",".join("?" * len(chunk))
        rows = src.execute(f"SELECT * FROM reviews WHERE hotel_id IN ({ph})", chunk).fetchall()
        dst.executemany(insert_sql, rows)

    # 3. Copy related authors
    # Extract distinct author_ids from the destination database's reviews
    author_ids = [row[0] for row in dst.execute("SELECT DISTINCT author_id FROM reviews WHERE author_id IS NOT NULL").fetchall()]
    author_cols = [d[0] for d in src.execute("SELECT * FROM authors LIMIT 1").description]
    author_insert_sql = f"INSERT OR IGNORE INTO authors ({','.join(author_cols)}) VALUES ({','.join('?' * len(author_cols))})"
    
    for i in range(0, len(author_ids), chunk_size):
        chunk = author_ids[i:i + chunk_size]
        ph = ",".join("?" * len(chunk))
        author_rows = src.execute(f"SELECT * FROM authors WHERE author_id IN ({ph})", chunk).fetchall()
        dst.executemany(author_insert_sql, author_rows)

    dst.commit()

    # Update hotel counts in sample just to be safe
    dst.execute("""
        UPDATE hotels SET num_reviews = (
            SELECT COUNT(*) FROM reviews WHERE reviews.hotel_id = hotels.hotel_id
        )
    """)
    dst.commit()

    count = dst.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    print(f"[data_processing] Sample DB → {count:,} reviews → {sample_db}")

    # Compute credibility weights for the sample DB
    compute_review_weights(dst)

    src.close()
    dst.close()
    return count


# CLI entry point

if __name__ == "__main__":
    sample_only = "--sample" in sys.argv

    if not sample_only:
        build_database()

    build_sample_database()
    print("[data_processing] All done ✓")
