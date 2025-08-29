import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV  = os.path.join(BASE_DIR, "csv_output", "reviews_clean_en.csv")
OUT_DIR  = os.path.join(BASE_DIR, "csv_output", "labeling")
OUT_CSV  = os.path.join(OUT_DIR, "labelset.csv")

# How many rows to label (tune as you wish)
N_RANDOM = 120
N_SHORT  = 60   # very short reviews (often low-signal or irrelevant)
N_URL    = 40   # contains url/email/phone (ad candidates)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(SRC_CSV)

    # candidate buckets to get a diverse starter set
    random_chunk = df.sample(n=min(N_RANDOM, len(df)), random_state=42)
    short_chunk  = df.sort_values("review_len_words").head(min(N_SHORT, len(df)))
    url_mask = (df.get("has_url", 0) == 1) | (df.get("has_email", 0) == 1) | (df.get("has_phone", 0) == 1)
    url_chunk  = df[url_mask].sample(n=min(N_URL, url_mask.sum()), random_state=42) if url_mask.any() else df.head(0)

    pool = pd.concat([random_chunk, short_chunk, url_chunk], axis=0)
    pool = pool.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

    # keep only columns helpful to annotators
    label_cols = [
        "review_id", "business_name", "rating", "text",
        # empty label placeholders
        "label_advertisement", "label_irrelevant", "label_rant_without_visit", "notes"
    ]
    for c in ["label_advertisement", "label_irrelevant", "label_rant_without_visit", "notes"]:
        pool[c] = ""  # annotators fill 0/1 (and free-text for notes)

    pool[label_cols].to_csv(OUT_CSV, index=False)
    print(f"Saved labeling CSV → {OUT_CSV}")
    print("Open it in Google Sheets or Excel, fill 0/1 for label_* columns, save as CSV.")
    print("When done, place the annotated file back at the same path and run merge_labels.py")

if __name__ == "__main__":
    main()
