
import argparse
import os
import re
import sys
from typing import Optional, Tuple, List

import pandas as pd

try:
    import gdown
except Exception as e:
    print("Error: gdown is required. Install with `pip install gdown`.", file=sys.stderr)
    raise


def extract_drive_id(url: str) -> Optional[str]:
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def download_from_drive(url: str, output: str) -> str:
    
    file_id = extract_drive_id(url)
    if not file_id:
        raise ValueError(f"Cannot parse Google Drive file id from URL: {url}")
   
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    gdown.download(id=file_id, output=output, quiet=False)
    if not os.path.exists(output):
        raise RuntimeError(f"Download failed for {url}")
    return output

def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "business name": ["business name", "business_name", "title", "name", "place", "restaurant", "poi", "shop"],
        "rating": ["rating", "stars", "score", "star", "rate"],
        "review text": ["review text", "review", "text", "comment", "content", "body"]
    }

    def find_col(possibles: List[str]) -> Optional[str]:
        cols_lower = {c.lower(): c for c in df.columns}
        for p in possibles:
            if p in cols_lower:
                return cols_lower[p]
        return None

    col_business = find_col(candidates["business name"])
    col_rating   = find_col(candidates["rating"])
    col_review   = find_col(candidates["review text"])

    missing = [t for t, c in zip(["business name","rating","review text"], [col_business, col_rating, col_review]) if c is None]
    if missing:
        raise KeyError(f"Missing required columns (or aliases) in input CSV: {missing}. "
                       f"Available columns: {list(df.columns)}")

    out = df[[col_business, col_rating, col_review]].rename(columns={
        col_business: "business name",
        col_rating: "rating",
        col_review: "review text"
    })

    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out["business name"] = out["business name"].astype(str).str.strip()
    out["review text"] = out["review text"].astype(str).str.strip()

    return out

def clean_merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    
    merged = pd.concat([df1, df2], ignore_index=True)
    merged = merged.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    merged = merged.dropna(subset=["business name", "rating", "review text"])

  
    merged["business name"] = merged["business name"].astype(str).str.strip()
    merged["review text"] = merged["review text"].astype(str).str.strip()
    merged = merged[(merged["business name"] != "") & (merged["review text"] != "")]

    merged = merged.drop_duplicates(subset=["business name", "rating", "review text"])

    return merged


def main():
    parser = argparse.ArgumentParser(description="Clean and merge Google reviews CSVs from Google Drive.")
    parser.add_argument("--url1", required=True, help="Google Drive share URL for first CSV")
    parser.add_argument("--url2", required=True, help="Google Drive share URL for second CSV")
    parser.add_argument("--out", default="cleaned_merged_reviews_full.csv", help="Output CSV filename")
    parser.add_argument("--tempdir", default="downloads", help="Directory to store downloaded raw CSVs")
    args = parser.parse_args()

   
    tmp1 = os.path.join(args.tempdir, "file1.csv")
    tmp2 = os.path.join(args.tempdir, "file2.csv")

    
    print("[1/4] Downloading CSV #1...")
    download_from_drive(args.url1, tmp1)
    print("[2/4] Downloading CSV #2...")
    download_from_drive(args.url2, tmp2)

  
    print("[3/4] Reading and standardizing columns...")
    df1_raw = pd.read_csv(tmp1)
    df2_raw = pd.read_csv(tmp2)

    df1_std = select_and_rename(df1_raw)
    df2_std = select_and_rename(df2_raw)

   
    print("[4/4] Merging and cleaning...")
    cleaned = clean_merge(df1_std, df2_std)

   
    cleaned.to_csv(args.out, index=False)
    print(f"Done. Saved cleaned file to: {args.out}")
    print(f"Total rows after cleaning: {len(cleaned)}")

if __name__ == "__main__":
    main()
