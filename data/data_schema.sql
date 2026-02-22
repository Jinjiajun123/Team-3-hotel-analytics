-- ============================================================================
-- Hotel Reviews Analytics — Database Schema
-- ============================================================================

-- Hotels table: one row per unique property (offering_id)
CREATE TABLE IF NOT EXISTS hotels (
    hotel_id   INTEGER PRIMARY KEY, 
    num_reviews INTEGER DEFAULT 0   
);

-- Authors table: one row per unique reviewer
CREATE TABLE IF NOT EXISTS authors (
    author_id          TEXT PRIMARY KEY,  
    username           TEXT,
    location           TEXT,
    num_cities         INTEGER,
    num_helpful_votes  INTEGER,
    num_reviews        INTEGER,
    num_type_reviews   INTEGER
);

-- Reviews table: one row per review
CREATE TABLE IF NOT EXISTS reviews (
    review_id           INTEGER PRIMARY KEY,
    hotel_id            INTEGER NOT NULL,
    author_id           TEXT,
    title               TEXT,
    text                TEXT,
    date                TEXT,              
    date_parsed         TEXT,          
    year                INTEGER,
    month               INTEGER,
    date_stayed         TEXT,
    rating_service      REAL,
    rating_cleanliness  REAL,
    rating_overall      REAL,
    rating_value        REAL,
    rating_location     REAL,
    rating_sleep_quality REAL,
    rating_rooms        REAL,
    num_helpful_votes INTEGER,
    via_mobile BOOLEAN,
    sentiment_polarity REAL,
    sentiment_subjectivity REAL,
    review_weight REAL DEFAULT 1.0,
    ml_is_luxury BOOLEAN DEFAULT 0,
    ml_is_budget BOOLEAN DEFAULT 0,
    ml_is_business BOOLEAN DEFAULT 0,
    FOREIGN KEY(hotel_id) REFERENCES hotels(hotel_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);

-- ── Indexes for query performance ────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_reviews_hotel_id ON reviews(hotel_id);
CREATE INDEX IF NOT EXISTS idx_reviews_author_id ON reviews(author_id);
CREATE INDEX IF NOT EXISTS idx_reviews_year ON reviews(year);
CREATE INDEX IF NOT EXISTS idx_reviews_date_parsed ON reviews(date_parsed);
CREATE INDEX IF NOT EXISTS idx_reviews_rating_overall ON reviews(rating_overall);
CREATE INDEX IF NOT EXISTS idx_reviews_hotel_year ON reviews(hotel_id, year);
