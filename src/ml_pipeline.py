import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import DB_PATH, SAMPLE_DB_PATH

# Prevent bus error on MacOS due to tokenizer multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def enrich_reviews_with_ml(db_path: Path, batch_size: int = 16):
    """
    Run Zero-Shot Classification on existing reviews to populate ml_is_luxury, ml_is_budget, ml_is_business.
    """
    if not db_path.exists():
        print(f"[ML] Error: DB not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    # Add columns if they do not exist
    try:
        conn.execute("ALTER TABLE reviews ADD COLUMN ml_is_luxury BOOLEAN DEFAULT 0")
        conn.execute("ALTER TABLE reviews ADD COLUMN ml_is_budget BOOLEAN DEFAULT 0")
        conn.execute("ALTER TABLE reviews ADD COLUMN ml_is_business BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Columns already exist
        
    cursor = conn.cursor()
    
    # Process all reviews
    cursor.execute("SELECT COUNT(*) FROM reviews")
    total_reviews = cursor.fetchone()[0]
    
    print(f"[ML] Initializing Zero-Shot pipeline (this may take a moment)...")
    # Use distilbert for faster CPU inference
    classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)
    
    labels = ["luxury", "budget", "business"]
    
    # Fetch all IDs and text
    cursor.execute("SELECT review_id, text FROM reviews")
    rows = cursor.fetchall()
    
    print(f"[ML] Processing {len(rows)} reviews...")
    
    update_batch = []
    
    for row in tqdm(rows, total=len(rows), desc="Classifying Reviews"):
        review_id, text = row
        
        # Truncate text to avoid model length errors
        short_text = text[:500] 
        
        try:
            result = classifier(short_text, labels, multi_label=True)
            
            # Extract scores
            scores = {label: score for label, score in zip(result['labels'], result['scores'])}
            
            is_luxury = 1 if scores.get("luxury", 0) > 0.6 else 0
            is_budget = 1 if scores.get("budget", 0) > 0.6 else 0
            is_business = 1 if scores.get("business", 0) > 0.6 else 0
            
            update_batch.append((is_luxury, is_budget, is_business, review_id))
            
            if len(update_batch) >= batch_size:
                cursor.executemany("UPDATE reviews SET ml_is_luxury=?, ml_is_budget=?, ml_is_business=? WHERE review_id=?", update_batch)
                conn.commit()
                update_batch = []
                
        except Exception as e:
            print(f"Error processing {review_id}: {e}")
            update_batch.append((0, 0, 0, review_id)) # default fallback if error occurs
            if len(update_batch) >= batch_size:
                cursor.executemany("UPDATE reviews SET ml_is_luxury=?, ml_is_budget=?, ml_is_business=? WHERE review_id=?", update_batch)
                conn.commit()
                update_batch = []
            continue

    if update_batch:
        cursor.executemany("UPDATE reviews SET ml_is_luxury=?, ml_is_budget=?, ml_is_business=? WHERE review_id=?", update_batch)
        conn.commit()
        
    print(f"[ML] Finished enriching database: {db_path}")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich database with ML inference.")
    parser.add_argument("--sample", action="store_true", help="Run on the sample database instead of full.")
    args = parser.parse_args()

    target_db = SAMPLE_DB_PATH if args.sample else DB_PATH
    enrich_reviews_with_ml(target_db)
