## Dashboard User Documentation

Welcome to the **Hotel Analytics Dashboard**. This application is designed to help hotel managers visualize reviews, understand competitive positioning, and identify actionable areas for improvement.

### How to Start the Dashboard
1. Ensure your virtual environment is active and dependencies are installed (`pip install -r requirements.txt`).
2. Run the Streamlit application from the root directory:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. The dashboard will automatically open in your default web browser at `http://localhost:8501`.

### Navigation & Features
Use the **Main Menu** in the left sidebar to navigate between different analytical views:

#### 1. 📊 Executive Overview
* **Purpose:** A high-level, macro perspective of the entire datasets.
* **Key Visuals:**
  * **Global Rating Distribution:** A donut chart showing the proportion of reviews by star rating.
  * **Review Volume Trend:** A smooth area chart plotting the growth of review volume over the 5-year period.
  * **Top Performing Hotels Leaderboard:** A dynamic table highlighting the absolute best hotels (minimum 50 reviews), featuring interactive progress bars for review volume.

#### 2. 🔍 Hotel Explorer
* **Purpose:** A micro-level deep dive into a specific hotel's performance compared to its algorithmic peers.
* **How to use:** 
  * Select or type a `Hotel ID` in the large, centralized search box.
  * **Overview Tab:** Displays the hotel's exact market segment (e.g., Luxury, Economic) calculated via K-Means clustering, and shows delta comparisons (+/-) against its direct competitors for Service, Cleanliness, Value, etc.
  * **Historical Trends Tab:** Visualizes how this specific hotel's ratings have evolved over time.
  * **Comparable Hotels Tab:** Lists other properties within the same market segment for direct competitive analysis.

#### 3. ⭐ Customer Priorities
* **Purpose:** Analyzes the structured NLP data to determine what customers actually care about.
* **Key Visuals:** Explores correlations between text length, sentiment polarity, and the ultimate `rating_overall` score.