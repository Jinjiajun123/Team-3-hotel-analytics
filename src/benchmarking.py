"""
benchmarking.py — Competitive benchmarking via K-Means clustering.

Provides functions to:
1. Aggregate per-hotel rating features
2. Cluster hotels into comparable groups
3. Analyse within-group performance
4. Generate actionable recommendations for under-performers
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import DB_PATH, RATING_COLUMNS, RATING_LABELS, get_db_connection


# 1. Feature extraction

def compute_hotel_features(db_path: Path = DB_PATH, min_reviews: int = 10) -> pd.DataFrame:
    """
    Compute per-hotel aggregated features for clustering.

    Returns a DataFrame indexed by hotel_id with columns:
        avg_<rating>, review_count, avg_text_length, mobile_ratio
    """
    conn = get_db_connection(db_path)

    # Check if the ML columns and review_weight exist in this database
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(reviews)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    has_ml_cols = "ml_is_luxury" in existing_cols
    has_weights = "review_weight" in existing_cols

    if has_ml_cols:
        if has_weights:
            semantic_sql = """
            SUM(ml_is_luxury   * review_weight) / SUM(review_weight) AS pct_luxury_mentions,
            SUM(ml_is_budget   * review_weight) / SUM(review_weight) AS pct_budget_mentions,
            SUM(ml_is_business * review_weight) / SUM(review_weight) AS pct_business_mentions"""
        else:
            semantic_sql = """
            AVG(ml_is_luxury)              AS pct_luxury_mentions,
            AVG(ml_is_budget)              AS pct_budget_mentions,
            AVG(ml_is_business)            AS pct_business_mentions"""
    else:
        if has_weights:
            semantic_sql = """
            SUM((CASE WHEN LOWER(text) LIKE '%luxury%' OR LOWER(text) LIKE '%elegant%' OR LOWER(text) LIKE '%premium%' THEN 1.0 ELSE 0.0 END) * review_weight) / SUM(review_weight) AS pct_luxury_mentions,
            SUM((CASE WHEN LOWER(text) LIKE '%cheap%' OR LOWER(text) LIKE '%budget%' OR LOWER(text) LIKE '%affordable%' THEN 1.0 ELSE 0.0 END) * review_weight) / SUM(review_weight) AS pct_budget_mentions,
            SUM((CASE WHEN LOWER(text) LIKE '%business%' OR LOWER(text) LIKE '%conference%' OR LOWER(text) LIKE '%work%' THEN 1.0 ELSE 0.0 END) * review_weight) / SUM(review_weight) AS pct_business_mentions"""
        else:
            semantic_sql = """
            AVG(CASE WHEN LOWER(text) LIKE '%luxury%' OR LOWER(text) LIKE '%elegant%' OR LOWER(text) LIKE '%premium%' THEN 1.0 ELSE 0.0 END) AS pct_luxury_mentions,
            AVG(CASE WHEN LOWER(text) LIKE '%cheap%' OR LOWER(text) LIKE '%budget%' OR LOWER(text) LIKE '%affordable%' THEN 1.0 ELSE 0.0 END) AS pct_budget_mentions,
            AVG(CASE WHEN LOWER(text) LIKE '%business%' OR LOWER(text) LIKE '%conference%' OR LOWER(text) LIKE '%work%' THEN 1.0 ELSE 0.0 END) AS pct_business_mentions"""

    if has_weights:
        # Weighted averages: high-credibility reviews have more influence
        query = f"""
            SELECT
                hotel_id,
                COUNT(*)                       AS review_count,
                SUM(rating_overall       * review_weight) / SUM(review_weight) AS avg_rating_overall,
                SUM(rating_service       * review_weight) / SUM(review_weight) AS avg_rating_service,
                SUM(rating_cleanliness   * review_weight) / SUM(review_weight) AS avg_rating_cleanliness,
                SUM(rating_value         * review_weight) / SUM(review_weight) AS avg_rating_value,
                SUM(rating_location      * review_weight) / SUM(review_weight) AS avg_rating_location,
                SUM(rating_sleep_quality * review_weight) / SUM(review_weight) AS avg_rating_sleep_quality,
                SUM(rating_rooms         * review_weight) / SUM(review_weight) AS avg_rating_rooms,
                SUM(LENGTH(text)         * review_weight) / SUM(review_weight) AS avg_text_length,
                AVG(via_mobile)                AS mobile_ratio,
                SUM(sentiment_polarity   * review_weight) / SUM(review_weight) AS avg_sentiment_polarity,
                {semantic_sql}
            FROM reviews
            GROUP BY hotel_id
            HAVING COUNT(*) >= ?
        """
    else:
        # Fallback: simple averages (no weight column)
        query = f"""
            SELECT
                hotel_id,
                COUNT(*)                       AS review_count,
                AVG(rating_overall)            AS avg_rating_overall,
                AVG(rating_service)            AS avg_rating_service,
                AVG(rating_cleanliness)        AS avg_rating_cleanliness,
                AVG(rating_value)              AS avg_rating_value,
                AVG(rating_location)           AS avg_rating_location,
                AVG(rating_sleep_quality)      AS avg_rating_sleep_quality,
                AVG(rating_rooms)              AS avg_rating_rooms,
                AVG(LENGTH(text))              AS avg_text_length,
                AVG(via_mobile)                AS mobile_ratio,
                AVG(sentiment_polarity)        AS avg_sentiment_polarity,
                {semantic_sql}
            FROM reviews
            GROUP BY hotel_id
            HAVING COUNT(*) >= ?
        """

    df = pd.read_sql_query(query, conn, params=(min_reviews,))
    conn.close()
    df.set_index("hotel_id", inplace=True)
    return df


# 2. Clustering

def cluster_hotels(
    features_df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float, KMeans, StandardScaler]:
    """
    Cluster hotels by their rating profile and objective traits using K-Means.

    Returns:
        (df_with_cluster_col, silhouette, fitted_kmeans, fitted_scaler)
    """
    rating_cols = [c for c in features_df.columns if c.startswith("avg_rating_")]
    
    # Feature Engineering: Include objective indicators & semantic markers of hotel size & class
    hotel_traits = [
        "avg_text_length", "mobile_ratio", "avg_sentiment_polarity",
        "pct_luxury_mentions", "pct_budget_mentions", "pct_business_mentions"
    ]
    
    feat_df = features_df.copy()
    if "review_count" in feat_df.columns:
        feat_df["log_review_count"] = np.log1p(feat_df["review_count"])
        hotel_traits.append("log_review_count")
    
    # Keep only traits that actually existing in dataframe (safeguard)
    hotel_traits = [t for t in hotel_traits if t in feat_df.columns]
    features_to_use = rating_cols + hotel_traits
    X = feat_df[features_to_use].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)

    result = features_df.loc[X.index].copy()
    result["cluster"] = labels
    return result, sil, km, scaler


def find_optimal_k(
    features_df: pd.DataFrame,
    k_range: range = range(3, 11),
    random_state: int = 42,
) -> dict[int, float]:
    """Compute silhouette score for each k → helps choose n_clusters."""
    rating_cols = [c for c in features_df.columns if c.startswith("avg_rating_")]
    
    hotel_traits = [
        "avg_text_length", "mobile_ratio", "avg_sentiment_polarity",
        "pct_luxury_mentions", "pct_budget_mentions", "pct_business_mentions"
    ]
    
    feat_df = features_df.copy()
    if "review_count" in feat_df.columns:
        feat_df["log_review_count"] = np.log1p(feat_df["review_count"])
        hotel_traits.append("log_review_count")
        
    # Keep only traits that actually existing in dataframe (safeguard)
    hotel_traits = [t for t in hotel_traits if t in feat_df.columns]
    features_to_use = rating_cols + hotel_traits
    X = feat_df[features_to_use].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def assign_cluster_labels(
    df_cl: pd.DataFrame,
    labels: list[str] | None = None,
) -> dict[int, str]:
    """
    Rank clusters by average overall rating (descending) and assign labels.

    Default labels are a 4-tier segmentation:
        Luxury, High-end, Mid-range, Economy

    If there are more clusters than labels, the last label is reused.
    Returns:
        dict mapping cluster_id -> label
    """
    if labels is None:
        labels = ["Luxury", "High-end", "Mid-range", "Economy"]

    if "cluster" not in df_cl.columns or "avg_rating_overall" not in df_cl.columns:
        raise ValueError("df_cl must contain 'cluster' and 'avg_rating_overall' columns.")

    cluster_means = df_cl.groupby("cluster")["avg_rating_overall"].mean()
    ranked_clusters = cluster_means.sort_values(ascending=False).index.tolist()

    cluster_labels: dict[int, str] = {}
    for i, cid in enumerate(ranked_clusters):
        cluster_labels[int(cid)] = labels[min(i, len(labels) - 1)]
    return cluster_labels


# 3. Group analysis

def analyze_group_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each cluster group's average ratings and size.

    Returns a DataFrame with one row per cluster.
    """
    rating_cols = [c for c in df.columns if c.startswith("avg_rating_")]
    summary = df.groupby("cluster").agg(
        num_hotels=("review_count", "size"),
        avg_reviews=("review_count", "mean"),
        **{col: (col, "mean") for col in rating_cols},
    ).round(2)
    return summary


# 4. Recommendations

def generate_recommendations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    For each cluster, identify under-performing hotels and suggest
    which rating dimensions to improve.

    Returns a DataFrame of hotel-level recommendations.
    """
    rating_cols = [c for c in df.columns if c.startswith("avg_rating_")]
    recs = []

    for cluster_id, group in df.groupby("cluster"):
        cluster_means = group[rating_cols].mean()

        for hotel_id, row in group.iterrows():
            gaps = cluster_means - row[rating_cols]
            # Positive gap = hotel is below cluster average
            weak_dims = gaps[gaps > 0.3].sort_values(ascending=False)

            if len(weak_dims) > 0:
                biggest_gap_col = weak_dims.index[0]
                label = RATING_LABELS.get(
                    biggest_gap_col.replace("avg_", ""), biggest_gap_col
                )
                recs.append({
                    "hotel_id": hotel_id,
                    "cluster": cluster_id,
                    "overall_rating": row.get("avg_rating_overall", None),
                    "biggest_weakness": label,
                    "gap_vs_peers": round(weak_dims.iloc[0], 2),
                    "num_weak_dimensions": len(weak_dims),
                    "recommendation": f"Focus on improving {label} "
                                      f"(gap of {weak_dims.iloc[0]:.2f} vs peer avg).",
                })

    rec_df = pd.DataFrame(recs)
    if len(rec_df) > 0:
        rec_df.sort_values("gap_vs_peers", ascending=False, inplace=True)
    return rec_df
