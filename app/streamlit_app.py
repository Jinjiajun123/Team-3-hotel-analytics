"""
streamlit_app.py — Hotel Analytics Dashboard

Run:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DB_PATH, SAMPLE_DB_PATH, RATING_COLUMNS, RATING_LABELS, get_db_connection

st.set_page_config(
    page_title="Hotel Performance Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
    /* Global Font & Background */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container { padding-top: 2rem; }
    
    /* Headings */
    h1, h2, h3 {
        color: #2D3748;
        font-weight: 600;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E2E8F0;
    }
    

</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_data(db_path: str):
    """Load all reviews into a DataFrame (cached)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    # Enforce 100k limit also on read if DB is huge, though build process handles it.
    df = pd.read_sql_query("SELECT * FROM reviews LIMIT 100000", conn)
    df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    conn.close()
    return df


@st.cache_data(ttl=600)
def load_hotels(db_path: str):
    import sqlite3
    conn = sqlite3.connect(db_path)
    hotels = pd.read_sql_query("SELECT * FROM hotels ORDER BY num_reviews DESC", conn)
    conn.close()
    return hotels


@st.cache_data(ttl=3600)
def get_hotel_clusters(db_path_str: str, n_clusters: int = 4, min_reviews: int = 20):
    """Cached computation of hotel market positioning."""
    from src.benchmarking import compute_hotel_features, cluster_hotels
    features = compute_hotel_features(db_path=Path(db_path_str), min_reviews=min_reviews)
    if features.empty:
        return pd.DataFrame()
    df_cl, sil, _, _ = cluster_hotels(features, n_clusters=n_clusters)
    return df_cl


def detect_db():
    """Use full DB if available, else sample."""
    if DB_PATH.exists():
        return str(DB_PATH)
    return str(SAMPLE_DB_PATH)


HOTEL_TIER_LABELS = [
    "👑 Luxury",
    "🏨 Upscale",
    "🏠 Midscale",
    "🏷️ Economy",
]


def assign_cluster_labels(df_cl: pd.DataFrame) -> dict:
    """
    Rank clusters by their average overall rating (descending)
    and assign the fixed 4-tier labels accordingly.
    Returns a dict mapping cluster_id -> label string.
    """
    cluster_means = df_cl.groupby("cluster")["avg_rating_overall"].mean()
    ranked = cluster_means.sort_values(ascending=False).index.tolist()
    labels = {}
    for i, cid in enumerate(ranked):
        labels[cid] = HOTEL_TIER_LABELS[min(i, len(HOTEL_TIER_LABELS) - 1)]
    return labels


def get_cluster_label(cluster_mean_series: pd.Series) -> str:
    """Fallback: generate a label for a single cluster based on overall rating."""
    overall = cluster_mean_series.get("avg_rating_overall", 3.0)
    if overall >= 4.2:
        return HOTEL_TIER_LABELS[0]
    elif overall >= 3.8:
        return HOTEL_TIER_LABELS[1]
    elif overall >= 3.3:
        return HOTEL_TIER_LABELS[2]
    else:
        return HOTEL_TIER_LABELS[3]

# Sidebar
st.sidebar.markdown(
    """
    <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
        <svg viewBox="0 0 200 200" width="120" height="120" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="primary" x1="0%" y1="100%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#4f46e5"/>
                    <stop offset="100%" stop-color="#3b82f6"/>
                </linearGradient>
            </defs>
            <circle cx="100" cy="100" r="95" fill="#EEF2FF" />
            <rect x="50" y="90" width="25" height="60" rx="4" fill="url(#primary)" opacity="0.8"/>
            <rect x="85" y="50" width="25" height="100" rx="4" fill="url(#primary)"/>
            <rect x="120" y="110" width="25" height="40" rx="4" fill="url(#primary)" opacity="0.6"/>
            <path d="M 40 120 L 85 70 L 120 90 L 155 45" fill="none" stroke="#f43f5e" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="155" cy="45" r="8" fill="#f43f5e" />
        </svg>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Hotel Analytics")

# ── Cross-page Navigation ────────────────────────────────────────────────────
if "nav_hotel_id" not in st.session_state:
    st.session_state.nav_hotel_id = None

main_page = st.sidebar.radio(
    "Main Menu",
    [
        "📊 Executive Overview", 
        "🔍 Hotel Explorer",
        "⭐ Customer Priorities",
        "📥 Upload Data",
    ],
    key="main_menu",
)

active_view = main_page

db_path = detect_db()
df = load_data(db_path)
hotels = load_hotels(db_path)

hotel_ids = sorted(df["hotel_id"].unique())

if main_page == "🔍 Hotel Explorer":
    pass 
else:
    global_selected_hotel = None


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if active_view == "📊 Executive Overview":
    st.title("Executive Overview")
    st.caption("High-level performance snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews Analyzed", f"{len(df):,}")
    c2.metric("Hotels Monitored", f"{df['hotel_id'].nunique():,}")
    c3.metric("Average Rating", f"{df['rating_overall'].mean():.2f} / 5.0")
    c4.metric("Data Period",
              f"{df['date_parsed'].dt.year.min()} - {df['date_parsed'].dt.year.max()}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # MARKET POSITIONING (Moved from standalone page)
    # ═══════════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Market Positioning & Segmentation")
    

    n_clusters = 4
    min_rev = 20

    with st.spinner("Analyzing market segments..."):
        df_cl = get_hotel_clusters(db_path, n_clusters=n_clusters, min_reviews=min_rev)

    if not df_cl.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Segment Distribution**")
            sizes = df_cl["cluster"].value_counts().sort_index().reset_index()
            sizes.columns = ["cluster", "Hotels"]
            
            cluster_means = df_cl.groupby("cluster").mean()
            semantic_names = assign_cluster_labels(df_cl)
            sizes["Segment"] = sizes["cluster"].map(semantic_names)
            
            fig = px.bar(sizes, x="Segment", y="Hotels", color="Segment",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Segment Characteristics**")
            rating_cols = [c for c in df_cl.columns if c.startswith("avg_rating_")]
            labels_short = [c.replace("avg_rating_", "").replace("_", " ").title() for c in rating_cols]

            fig = go.Figure()
            for cid, row in cluster_means.iterrows():
                r_values = row[rating_cols].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=r_values + [r_values[0]],
                    theta=labels_short + [labels_short[0]],
                    fill="toself", name=semantic_names[cid],
                ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[2, 5], visible=True)), height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**🏨 Hotels by Tier**")
        tier_order = {label: i for i, label in enumerate(HOTEL_TIER_LABELS)}
        sorted_tiers = sorted(semantic_names.items(), key=lambda x: tier_order.get(x[1], 99))

        for cid, tier_name in sorted_tiers:
            tier_hotels = df_cl[df_cl["cluster"] == cid].copy()
            with st.expander(f"{tier_name}  —  {len(tier_hotels)} hotels", expanded=False):
                sort_by = st.radio(
                    "Sort by", ["Overall Rating", "Review Count"],
                    horizontal=True, key=f"sort_{cid}",
                )
                if sort_by == "Overall Rating":
                    tier_hotels = tier_hotels.sort_values("avg_rating_overall", ascending=False)
                else:
                    tier_hotels = tier_hotels.sort_values("review_count", ascending=False)

                display_df = tier_hotels[["review_count", "avg_rating_overall"]].copy()
                display_df.columns = ["Reviews", "Overall Rating"]
                display_df["Overall Rating"] = display_df["Overall Rating"].round(2)
                display_df.index.name = "Hotel ID"
                st.dataframe(display_df, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Rating Distribution")
        rating_counts = df["rating_overall"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        rating_counts["Rating Name"] = rating_counts["Rating"].astype(str) + " Stars"
        
        fig1 = px.pie(rating_counts, values="Count", names="Rating Name", hole=0.55,
                      color_discrete_sequence=px.colors.sequential.Teal,
                      title="Distribution of Overall Ratings")
        fig1.update_traces(textposition='inside', textinfo='percent+label', 
                           marker=dict(line=dict(color='#0E1117', width=2)))
        fig1.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Review Volume Trend")
        year_counts = df.groupby("year").size().reset_index(name="count")
        
        fig2 = px.area(year_counts, x="year", y="count",
                       title="Reviews Collected per Year",
                       color_discrete_sequence=["#667EEA"])
        fig2.update_traces(line_shape='spline', fill='tozeroy', 
                           line=dict(width=3), fillcolor='rgba(102, 126, 234, 0.4)')
        fig2.update_layout(xaxis_title="Year", yaxis_title="Review Count",
                           margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top Performing Hotels")
    st.markdown("_Hotels with the highest average rating (minimum 50 reviews)_")
    top = (df.groupby("hotel_id")
           .agg(avg_rating=("rating_overall", "mean"), reviews=("rating_overall", "count"))
           .query("reviews >= 50")
           .nlargest(10, "avg_rating")
           .reset_index())
    top["avg_rating"] = top["avg_rating"].round(2)
    top.columns = ["Hotel ID", "Average Rating", "Review Count"]
    
    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hotel ID": st.column_config.TextColumn(
                "🏨 Hotel ID",
                help="Unique Identifier for the Hotel"
            ),
            "Average Rating": st.column_config.NumberColumn(
                "⭐ Average Rating",
                help="Overall score out of 5 stars",
                format="%.2f",
            ),
            "Review Count": st.column_config.ProgressColumn(
                "📝 Review Volume",
                help="Total number of reviews collected",
                format="%d",
                min_value=0,
                max_value=int(top["Review Count"].max())
            )
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: HOTEL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif main_page == "🔍 Hotel Explorer":
    st.markdown(
        """
        <style>
        /* Target the selectbox dropdown container and scale it uniformly */
        .stSelectbox {
            transform: scale(1.15);
            transform-origin: top center;
            margin-bottom: 20px;
            width: 80% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 8px;
        }
        /* Style the header elements */
        .chat-header-container {
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        .chat-header-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 0px;
            padding-bottom: 0px;
            background: -webkit-linear-gradient(45deg, #4f46e5, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .chat-header-subtitle {
            font-size: 1.25rem;
            color: #64748b;
            margin-top: 5px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="chat-header-container">
            <h1 class="chat-header-title">Hotel Explorer</h1>
            <p class="chat-header-subtitle">Which hotel would you like to analyze today?</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    default_idx = None
    if st.session_state.nav_hotel_id is not None and st.session_state.nav_hotel_id in hotel_ids:
        default_idx = hotel_ids.index(st.session_state.nav_hotel_id)
        st.session_state.nav_hotel_id = None  

    global_selected_hotel = st.selectbox(
        "Search Hotel ID", 
        hotel_ids, 
        index=default_idx,
        placeholder="Type or select a Hotel ID...",
        label_visibility="collapsed"
    )

    if global_selected_hotel:

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        try:
            sub_page = st.pills(
                "View Context",
                ["🔍 Overview", "📈 Historical Trends", "🏅 Comparable Hotels"],
                default="🔍 Overview",
                label_visibility="collapsed",
                key="hotel_submenu_main"
            )
            if not sub_page: 
                sub_page = "🔍 Overview"
        except AttributeError:
            sub_page = st.radio(
                "View Context",
                ["🔍 Overview", "📈 Historical Trends", "🏅 Comparable Hotels"],
                horizontal=True,
                label_visibility="collapsed",
                key="hotel_submenu_main"
            )
    
        st.divider()

        selected = global_selected_hotel
        hdf = df[df["hotel_id"] == selected]

        if sub_page == "🔍 Overview":
            st.markdown(f"_Deep dive into specific hotel performance for **Hotel {selected}**_")

            min_rev_explorer = 5 
            df_cl = get_hotel_clusters(db_path, n_clusters=4, min_reviews=min_rev_explorer)
            in_cluster_analysis = not df_cl.empty and selected in df_cl.index
    
            if in_cluster_analysis:
                tier_labels = assign_cluster_labels(df_cl)
                cluster_id = df_cl.loc[selected, "cluster"]
                cluster_peers = df_cl[df_cl["cluster"] == cluster_id]
                cluster_mean = cluster_peers.mean()
                segment_name = tier_labels.get(cluster_id, get_cluster_label(cluster_mean))
            else:
                st.warning(f"**Market Segment:** This hotel does not have enough reviews (minimum {min_rev_explorer}) to be reliably placed into a competitive market segment.")

            st.subheader("⭐ Rating Breakdown")
            means = hdf[RATING_COLUMNS].mean()

            def _delta(col_name):
                if not in_cluster_analysis:
                    return None
                seg_val = cluster_mean.get(f"avg_{col_name}", cluster_mean.get(col_name, None))
                if seg_val is None:
                    return None
                diff = means.get(col_name, 0) - seg_val
                return f"{diff:+.2f} vs segment"

            r0c1, r1c2 = st.columns(2)
            r0c1.metric("Total Reviews", f"{len(hdf):,}")
            r1c2.metric("Hotel Tier", f"{segment_name}")

            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Overall", f"{means.get('rating_overall', 0):.2f} / 5", delta=_delta("rating_overall"))
            r1c2.metric("Service", f"{means.get('rating_service', 0):.2f} / 5", delta=_delta("rating_service"))
            r1c3.metric("Cleanliness", f"{means.get('rating_cleanliness', 0):.2f} / 5", delta=_delta("rating_cleanliness"))
            r1c4.metric("Value", f"{means.get('rating_value', 0):.2f} / 5", delta=_delta("rating_value"))

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("Location", f"{means.get('rating_location', 0):.2f} / 5", delta=_delta("rating_location"))
            r2c2.metric("Sleep Quality", f"{means.get('rating_sleep_quality', 0):.2f} / 5", delta=_delta("rating_sleep_quality"))
            r2c3.metric("Rooms", f"{means.get('rating_rooms', 0):.2f} / 5", delta=_delta("rating_rooms"))
            if "sentiment_polarity" in hdf.columns:
                avg_sent = hdf["sentiment_polarity"].mean()
                sent_emoji = "😊" if avg_sent > 0.2 else ("😐" if avg_sent > 0 else "😟")
                r2c4.metric("Sentiment", f"{avg_sent:.2f} {sent_emoji}",
                             help="Sentiment Polarity measures how positive or negative the review language is, computed by TextBlob NLP.\n\n"
                                  "• **+0.2 to +1.0**: Positive 😊 (e.g. 'amazing', 'excellent')\n"
                                  "• **0 to +0.2**: Neutral 😐 (e.g. 'okay', 'average')\n"
                                  "• **-1.0 to 0**: Negative 😟 (e.g. 'terrible', 'horrible')")

            st.divider()

            # Customer Feedback (Tabs: Latest + Most Helpful + Search)
            st.subheader("💬 Customer Feedback")
            tab_latest, tab_helpful, tab_search = st.tabs(["🕒 Latest Reviews", "👍 Most Helpful Reviews", "🔎 Search Reviews"])

            with tab_latest:
                recent = hdf.nlargest(5, "date_parsed")[["date", "title", "rating_overall", "text"]]
                recent.columns = ["Date", "Title", "Rating", "Review"]
                for _, row in recent.iterrows():
                    with st.container():
                        st.markdown(f"**{'⭐' * int(row['Rating'])}** — *{row['Date']}*")
                        st.markdown(f"**{row['Title']}**")
                        st.markdown(f"> {row['Review'][:500]}{'…' if len(str(row['Review'])) > 500 else ''}")
                        st.divider()

            with tab_helpful:
                if "num_helpful_votes" in hdf.columns:
                    helpful = hdf.nlargest(5, "num_helpful_votes")[["date", "title", "rating_overall", "text", "num_helpful_votes"]]
                    helpful.columns = ["Date", "Title", "Rating", "Review", "Helpful Votes"]
                    for _, row in helpful.iterrows():
                        with st.container():
                            st.markdown(f"**{'⭐' * int(row['Rating'])}** — *{row['Date']}* — 👍 **{int(row['Helpful Votes'])} helpful votes**")
                            st.markdown(f"**{row['Title']}**")
                            st.markdown(f"> {row['Review'][:500]}{'…' if len(str(row['Review'])) > 500 else ''}")
                            st.divider()
                else:
                    st.caption("Helpful votes data not available.")

            with tab_search:
                query = st.text_input("Search this hotel's reviews", placeholder="e.g. breakfast, noisy, front desk", key="hotel_search")
                if query:
                    mask = hdf["text"].str.contains(query, case=False, na=False) | \
                           hdf["title"].str.contains(query, case=False, na=False)
                    results = hdf[mask]
                    st.info(f"Found **{len(results):,}** reviews matching '**{query}**'")
                    if not results.empty:
                        for _, row in results.sort_values("date_parsed", ascending=False).head(20).iterrows():
                            with st.container():
                                st.markdown(f"**{'⭐' * int(row['rating_overall'])}** — *{row['date']}*")
                                st.markdown(f"**{row['title']}**")
                                st.markdown(f"> {str(row['text'])[:500]}{'…' if len(str(row['text'])) > 500 else ''}")
                                st.divider()
                else:
                    st.caption("Enter keywords above to search within this hotel's reviews.")


        elif sub_page == "📈 Historical Trends":
            st.markdown(f"_Track performance metrics over time for **Hotel {selected}**_")

            st.subheader("Monthly Performance Trends")
            rating_choice = st.selectbox("Select Metric", list(RATING_LABELS.values()), key="hotel_trend_metric")
            col_name = [k for k, v in RATING_LABELS.items() if v == rating_choice][0]

            monthly = (hdf.set_index("date_parsed")
                       .resample("ME")[col_name].agg(["mean", "count"]).reset_index())
        
            monthly = monthly.dropna(subset=["mean"])
        
            if not monthly.empty:
                fig = px.line(monthly, x="date_parsed", y="mean",
                              labels={"date_parsed": "Date", "mean": f"Avg {rating_choice}"},
                              color_discrete_sequence=["#4299E1"])
            
                fig.add_bar(x=monthly["date_parsed"], y=monthly["count"], name="Review Volume",
                            opacity=0.15, marker_color="#ED8936", yaxis="y2")
            
                fig.update_layout(
                    title=f"{rating_choice} Trend — Hotel {selected}",
                    yaxis2=dict(title="Review Volume", overlaying="y", side="right", showgrid=False),
                    legend=dict(orientation="h", y=1.1),
                    hovermode="x unified",
                    yaxis=dict(range=[1, 5.2])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Not enough data to plot monthly trends for this hotel.")

            st.subheader("Year-over-Year Performance")
            yearly = hdf.groupby("year")[RATING_COLUMNS].mean().reset_index()
            if not yearly.empty:
                yearly_melted = yearly.melt(id_vars="year", var_name="dimension", value_name="avg_rating")
                yearly_melted["dimension"] = yearly_melted["dimension"].map(RATING_LABELS)
            
                fig = px.bar(yearly_melted, x="dimension", y="avg_rating", color="year",
                             barmode="group", color_continuous_scale="Purples",
                             labels={"dimension": "", "avg_rating": "Avg Rating"})
                fig.update_layout(yaxis=dict(range=[1, 5.2]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Not enough data to plot yearly trends for this hotel.")




        elif sub_page == "🏅 Comparable Hotels":
            st.markdown(f"_Who are the direct competitors for **Hotel {selected}**?_")
    
            min_rev_explorer = 5 
            with st.spinner("Finding market peers..."):
                df_cl = get_hotel_clusters(db_path, n_clusters=4, min_reviews=min_rev_explorer)
            
            in_cluster_analysis = not df_cl.empty and selected in df_cl.index
        
            if in_cluster_analysis:
                tier_labels = assign_cluster_labels(df_cl)
                cluster_id = df_cl.loc[selected, "cluster"]
                cluster_peers = df_cl[df_cl["cluster"] == cluster_id]
                cluster_mean = cluster_peers.mean()
                segment_name = tier_labels.get(cluster_id, get_cluster_label(cluster_mean))
            
                st.success(f"**Hotel Tier:** Hotel {selected} belongs to **{segment_name}** along with {len(cluster_peers) - 1} other comparable hotels.")
            
                @st.dialog("Hotel Details & Semantic Profile")
                def show_competitor_details(peer_id, peer_overall, peer_count, cluster_mean_series):
                    st.markdown(f"**Hotel {peer_id}**  —  ⭐ {peer_overall:.2f}  •  {peer_count} reviews")
                
                    peer_reviews = df[df["hotel_id"] == peer_id]
                    peer_means = peer_reviews[RATING_COLUMNS].mean()
                
                    def _delta(col):
                        val = peer_means.get(col, 0)
                        avg = cluster_mean_series.get(f"avg_{col}", 3.0)
                        diff = val - avg
                        return f"{diff:+.2f} vs segment" if not pd.isna(diff) else None

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Overall", f"{peer_means.get('rating_overall', 0):.2f}", delta=_delta("rating_overall"))
                    c2.metric("Service", f"{peer_means.get('rating_service', 0):.2f}", delta=_delta("rating_service"))
                    c3.metric("Cleanliness", f"{peer_means.get('rating_cleanliness', 0):.2f}", delta=_delta("rating_cleanliness"))
                    c4.metric("Value", f"{peer_means.get('rating_value', 0):.2f}", delta=_delta("rating_value"))
                
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Location", f"{peer_means.get('rating_location', 0):.2f}", delta=_delta("rating_location"))
                    c6.metric("Sleep", f"{peer_means.get('rating_sleep_quality', 0):.2f}", delta=_delta("rating_sleep_quality"))
                    c7.metric("Rooms", f"{peer_means.get('rating_rooms', 0):.2f}", delta=_delta("rating_rooms"))
                
                    if "sentiment_polarity" in peer_reviews.columns:
                        ps = peer_reviews["sentiment_polarity"].mean()
                        sent_icon = "😊" if ps > 0.2 else ("😐" if ps > 0 else "😟")
                        ps_avg = cluster_mean_series.get("avg_sentiment_polarity", 0)
                        c8.metric("Sentiment", f"{ps:.2f} {sent_icon}", delta=f"{ps - ps_avg:+.2f} vs segment" if not pd.isna(ps_avg) else None)

                    st.divider()

                    from sklearn.feature_extraction.text import TfidfVectorizer
                    texts = peer_reviews["text"].dropna().tolist()
                    if len(texts) >= 5:
                        with st.spinner("Extracting representative phrases..."):
                            hotel_stops = [
                                "hotel", "room", "rooms", "bed", "bathroom", "good", "great", 
                                "nice", "stay", "stayed", "time", "place", "really", "just", 
                                "like", "didn", "don", "got", "did", "one", "night", "nights",
                                "staff", "clean", "location", "would", "nyc", "york", "new york",
                                "manhattan", "times square", "city", "friendly", "helpful",
                                "small", "yes", "no", "very", "too", "much", "also", "well",
                                "even", "could", "get", "make", "us", "we"
                            ]
                            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                            custom_stops = list(ENGLISH_STOP_WORDS) + hotel_stops

                            tfidf = TfidfVectorizer(
                                max_features=500, 
                                stop_words=custom_stops,
                                ngram_range=(2, 3), 
                                min_df=2, max_df=0.60,
                            )
                            try:
                                X = tfidf.fit_transform(texts)
                                scores = X.mean(axis=0).A1
                                top_idx = scores.argsort()[-40:][::-1] 
                                raw_keywords = [tfidf.get_feature_names_out()[i] for i in top_idx]
                            
                                final_keywords = []
                                for kw in raw_keywords:
                                    kw_words = set(kw.split())
                                    if any(kw_words.issubset(set(fkw.split())) for fkw in final_keywords):
                                        continue
                                    final_keywords = [fkw for fkw in final_keywords if not set(fkw.split()).issubset(kw_words)]
                                    final_keywords.append(kw)
                                    if len(final_keywords) == 8:
                                        break
                            
                                st.markdown("**🔑 Representative Phrases:**")
                                bubble_css = "background-color: rgba(128, 128, 128, 0.2); color: inherit; border-radius: 12px; padding: 4px 12px; margin: 4px; display: inline-block; font-size: 0.9em;"
                                st.markdown(" ".join([f"<span style='{bubble_css}'>{kw}</span>" for kw in final_keywords]), unsafe_allow_html=True)
                            except Exception:
                                st.caption("Could not extract enough meaningful phrases.")
                    else:
                        st.caption("Not enough reviews to extract reliable phrases.")

                if len(cluster_peers) > 1:
                    st.divider()
                
                    peers_display = cluster_peers.drop(index=selected).copy()
                    sort_peers_by = st.radio(
                        "Sort by", ["Overall Rating", "Review Count"],
                        horizontal=True, key="sort_peers_page", label_visibility="collapsed"
                    )
                    if sort_peers_by == "Overall Rating":
                        peers_display = peers_display.sort_values("avg_rating_overall", ascending=False)
                    else:
                        peers_display = peers_display.sort_values("review_count", ascending=False)

                    top_peers = peers_display.head(10)

                    st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
                
                    h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
                    h1.caption("**HOTEL**")
                    h2.caption("**RATING**")
                    h3.caption("**REVIEWS**")
                    h4.caption("")
                    st.markdown("<hr style='margin: 0px 0 8px 0; border: none; border-bottom: 2px solid #e0e0e0;'/>", unsafe_allow_html=True)

                    for peer_id in top_peers.index:
                        peer_overall = top_peers.loc[peer_id, "avg_rating_overall"]
                        peer_count = int(top_peers.loc[peer_id, "review_count"])
                    
                        c1, c2, c3, c4 = st.columns([3, 2, 2, 2], vertical_alignment="center")
                        c1.markdown(f"**Hotel {peer_id}**")
                    
                        stars = "⭐" * int(round(peer_overall))
                        c2.markdown(f"{stars} **{peer_overall:.2f}**")
                    
                        c3.markdown(f"{peer_count:,}")
                    
                        with c4:
                            if st.button("View Details", key=f"btn_page_{peer_id}", use_container_width=True):
                                show_competitor_details(peer_id, peer_overall, peer_count, cluster_mean)
                    
                        st.markdown("<hr style='margin: 4px 0; border: none; border-bottom: 1px solid #f0f2f6;'/>", unsafe_allow_html=True)
                else:
                    st.info("There are no other hotels currently classified in this exact segment to display.")

            else:
                st.warning(f"**Market Segment:** This hotel does not have enough reviews (minimum {min_rev_explorer}) to be reliably placed into a competitive market segment, so we cannot identify direct comparable hotels for it.")



    # ═══════════════════════════════════════════════════════════════════════════════
    # PAGE 4: CUSTOMER PRIORITIES
    # ═══════════════════════════════════════════════════════════════════════════════
elif active_view == "⭐ Customer Priorities":
    st.title("Customer Priorities")
    st.markdown("_What drives guest satisfaction?_")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Key Satisfaction Drivers")
        st.markdown(
            "This chart shows which factors correlate most strongly with overall satisfaction. "
            "**Focus on improving the top bars to boost overall ratings.**"
        )
        
        corr = df[RATING_COLUMNS].corr()["rating_overall"].drop("rating_overall").sort_values(ascending=True)
        corr.index = [RATING_LABELS.get(c, c) for c in corr.index]
        
        fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                    labels={"x": "Impact on Overall Satisfaction", "y": ""},
                    color=corr.values, color_continuous_scale="Teal")
        fig.update_layout(height=500, xaxis_title="Correlation Strength")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Performance Gap")
        st.markdown("Current average performance across dimensions:")
        
        avg_ratings = df[RATING_COLUMNS].mean().sort_values(ascending=False)
        avg_ratings.index = [RATING_LABELS[c] for c in avg_ratings.index]
        
        st.dataframe(avg_ratings.to_frame(name="Avg Rating").round(2), use_container_width=True)

    st.divider()
    
    st.subheader("Digital Experience: Mobile vs Desktop")
    st.markdown("Comparing satisfaction levels by booking/review platform.")
    
    mobile_comp = df.groupby("via_mobile")[RATING_COLUMNS].mean().T
    mobile_comp.columns = ["Desktop", "Mobile"]
    mobile_comp.index = [RATING_LABELS[c] for c in RATING_COLUMNS]
    
    mobile_melted = mobile_comp.reset_index().melt(id_vars="index", var_name="Platform", value_name="Rating")
    fig = px.bar(mobile_melted, x="index", y="Rating", color="Platform", barmode="group",
                color_discrete_map={"Desktop": "#CBD5E0", "Mobile": "#667EEA"},
                labels={"index": ""})
    fig.update_layout(yaxis_range=[3, 5])
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: UPLOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif active_view == "📥 Upload Data":
    st.markdown("""
    <style>
    .upload-hero { text-align: center; padding: 3rem 1rem 1.5rem 1rem; }
    .upload-hero h1 { font-size: 2.4rem; font-weight: 700; color: #1e293b; margin-bottom: 0.4rem; }
    .upload-hero p { font-size: 1.05rem; color: #64748b; max-width: 560px; margin: 0 auto; }
    .schema-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.2rem 1.5rem; font-family: monospace; font-size: 0.85rem;
        color: #334155; line-height: 1.7;
    }
    .step-badge {
        display: inline-block; background: #4f46e5; color: white; border-radius: 50%;
        width: 28px; height: 28px; text-align: center; line-height: 28px;
        font-weight: 700; font-size: 0.85rem; margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-hero">
        <h1>📥 Upload Review Data</h1>
        <p>Append new hotel reviews from a JSON file directly into the analytics database.
           Uploaded records are processed, filtered, and reflected across all dashboard views instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    left_col, right_col = st.columns([1, 1.1], gap="large")

    with left_col:
        st.subheader("📋 How It Works")
        st.markdown("""
        <div style="line-height: 2.2; color: #475569;">
        <span class="step-badge">1</span> Prepare a <strong>.json</strong> file with <strong>one review per line</strong>.<br>
        <span class="step-badge">2</span> Drop the file into the uploader panel on the right.<br>
        <span class="step-badge">3</span> Click <strong>Start Ingestion</strong> and watch the live progress.<br>
        <span class="step-badge">4</span> Dashboard <strong>auto-refreshes</strong> — no manual reload needed.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Required JSON Fields")
        st.markdown("""
        <div class="schema-card">
        {<br>
        &nbsp;&nbsp;<span style="color:#4f46e5;">"id"</span>: 123456,<br>
        &nbsp;&nbsp;<span style="color:#4f46e5;">"offering_id"</span>: 78901,<br>
        &nbsp;&nbsp;<span style="color:#4f46e5;">"date"</span>: <span style="color:#16a34a;">"December 5, 2011"</span>,<br>
        &nbsp;&nbsp;<span style="color:#4f46e5;">"text"</span>: <span style="color:#16a34a;">"Great hotel..."</span>,<br>
        &nbsp;&nbsp;<span style="color:#4f46e5;">"ratings"</span>: { <span style="color:#4f46e5;">"overall"</span>: 5 },<br>
        &nbsp;&nbsp;<span style="color:#4f46e5;">"author"</span>: { <span style="color:#4f46e5;">"id"</span>: <span style="color:#16a34a;">"abc123"</span> }<br>
        }
        </div>
        """, unsafe_allow_html=True)
        st.caption("Filters: English only · ≥20 chars · Year 2008–2012 · Must have overall rating.")

    with right_col:
        st.subheader("📂 Select File")

        target_db = Path(detect_db())
        st.info(f"**Target:** Production DB (`{target_db.name}`)", icon="🗄️")

        uploaded_file = st.file_uploader(
            "Drop your JSON file here",
            type=["json"],
            key="json_uploader_page",
            help="One JSON object per line. Max 200 MB.",
        )

        if uploaded_file is not None:
            file_size_kb = len(uploaded_file.getvalue()) / 1024
            st.success(f"**{uploaded_file.name}** — {file_size_kb:,.1f} KB ready to process.", icon="📄")

            if st.button("⚡ Start Ingestion", use_container_width=True, type="primary", key="ingest_btn"):
                import json as _json
                from src.data_processing import ingest_uploaded_reviews

                raw_bytes = uploaded_file.read()
                text = raw_bytes.decode("utf-8").strip()
                lines = text.splitlines()

                if not lines:
                    st.error("The file appears to be empty.")
                else:
                    records = []
                    parse_errors = 0

                    for ln in lines:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            records.append(_json.loads(ln))
                        except _json.JSONDecodeError:
                            parse_errors += 1

                    if not records or (parse_errors > 0 and parse_errors >= len(records)):
                        records = []
                        parse_errors = 0
                        depth = 0
                        buf = ""
                        for ch in text:
                            buf += ch
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0 and buf.strip():
                                    try:
                                        records.append(_json.loads(buf.strip()))
                                    except _json.JSONDecodeError:
                                        parse_errors += 1
                                    buf = ""


                    if not records:
                        st.error(f"No valid JSON lines found. ({parse_errors} malformed lines skipped.)")
                    else:
                        if parse_errors > 0:
                            st.warning(f"⚠️ {parse_errors} malformed line(s) will be skipped.")

                        st.markdown("---")
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        stats_placeholder = st.empty()

                        def _update_progress(done: int, total: int):
                            pct = done / total
                            progress_bar.progress(
                                pct,
                                text=f"Processing **{done:,}** / **{total:,}** records — {pct*100:.1f}%"
                            )

                        status_text.info(f"🔄 Ingesting **{len(records):,}** records into `{target_db.name}`…")

                        try:
                            result = ingest_uploaded_reviews(
                                records,
                                db_path=target_db,
                                progress_callback=_update_progress,
                            )
                            progress_bar.progress(1.0, text="✅ Processing complete!")
                            status_text.empty()
                            stats_placeholder.success(
                                f"**Ingestion complete!**\n\n"
                                f"- ✅ **{result['inserted']:,}** new reviews inserted\n"
                                f"- 🚫 **{result['skipped']:,}** records filtered/skipped\n"
                                f"- 🗄️ **{result['total_reviews_in_db']:,}** total reviews in database"
                            )
                            st.balloons()
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as exc:
                            progress_bar.empty()
                            status_text.error(f"❌ Ingestion failed: {exc}")
        else:
            st.markdown("""
            <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 2.5rem 1rem;
                        text-align: center; color: #94a3b8; margin-top: 1rem; background: #f8fafc;">
                <div style="font-size: 2.8rem;">☁️</div>
                <div style="font-size: 1rem; margin-top: 0.5rem; line-height: 1.8;">
                    Drag & drop your JSON file above<br>or click <strong>Browse files</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("Powered by SQLite & Streamlit")
