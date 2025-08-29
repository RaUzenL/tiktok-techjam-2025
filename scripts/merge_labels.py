import os
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV  = os.path.join(BASE_DIR, "csv_output", "reviews_clean_en.csv")
LABEL_DIR  = os.path.join(BASE_DIR, "csv_output", "labeling")
LABEL_CSV  = os.path.join(LABEL_DIR, "labelset.csv")              # annotated by humans
OUT_CSV    = os.path.join(BASE_DIR, "csv_output", "reviews_labeled.csv")

REQUIRED = ["review_id", "label_advertisement", "label_irrelevant", "label_rant_without_visit"]

def to_binary(x):
    # convert '1'/'0'/1/0/'' to int 0/1
    if pd.isna(x) or x == "":
        return 0
    try:
        return 1 if int(x) == 1 else 0
    except Exception:
        s = str(x).strip().lower()
        return 1 if s in {"1", "yes", "y", "true", "t"} else 0

def main():
    if not os.path.exists(CLEAN_CSV):
        print(f"Missing: {CLEAN_CSV}")
        return
    if not os.path.exists(LABEL_CSV):
        print(f"Missing: {LABEL_CSV}")
        return

    base = pd.read_csv(CLEAN_CSV)
    lab  = pd.read_csv(LABEL_CSV)

    for c in REQUIRED:
        if c not in lab.columns:
            raise ValueError(f"Label file missing column: {c}")

    # normalize label columns to 0/1 ints
    for c in ["label_advertisement", "label_irrelevant", "label_rant_without_visit"]:
        lab[c] = lab[c].apply(to_binary)

    merged = base.merge(lab[["review_id", "label_advertisement", "label_irrelevant",
                             "label_rant_without_visit"]],
                        on="review_id", how="left")

    # fill missing with 0 (not labeled/assumed negative)
    for c in ["label_advertisement", "label_irrelevant", "label_rant_without_visit"]:
        merged[c] = merged[c].fillna(0).astype(int)

    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved merged dataset with human labels → {OUT_CSV}")

    # quick counts
    counts = {
        c: int(merged[c].sum())
        for c in ["label_advertisement", "label_irrelevant", "label_rant_without_visit"]
    }
    print("Positive counts:", counts)

if __name__ == "__main__":
    main()
