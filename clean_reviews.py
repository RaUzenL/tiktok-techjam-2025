#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean TWO review CSVs from Google Drive and merge them.

Fixes:
- Ensures *both* URLs are used (no dataset is ignored)
- Properly removes missing values (including blank strings/whitespace)
- Drops duplicates across the 3 standardized columns

Output columns:
- business name
- rating
- review text
"""
import argparse
import os
import re
import sys
from typing import Optional, List

import pandas as pd

try:
    import gdown
except Exception:
    print("Error: gdown is required. Install with `pip install gdown`.", file=sys.stderr)
    raise

# ---------- helpers ----------
def extract_drive_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from typical share URL formats."""
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
    """Download a file from Google Drive share URL to `output` path using gdown."""
    file_id = extract_drive_id(url)
    if not file_id:
        raise ValueError(f"Cannot parse Google Drive file id from URL: {url}")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    gdown.download(id=file_id, output=output, quiet=False)
    if not os.path.exists(output):
        raise RuntimeError(f"Download failed for {url}")
    return output

def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to ['business name','rating','review text'] using robust alias matching."""
    alias = {
        "business name": ["business name","business_name","title","name","place","restaurant","poi","shop"],
        "rating": ["rating","stars","score","star","rate"],
        "review text": ["review text","review","text","comment","content","body"]
    }
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(possibles: List[str]) -> Optional[str]:
        for p in possibles:
            if p in cols_lower:
                return cols_lower[p]
        return None

    c_biz = pick(alias["business name"])
    c_rat = pick(alias["rating"])
    c_rev = pick(alias["review text"])

    missing = [n for n,c in zip(["business name","rating","review text"], [c_biz,c_rat,c_rev]) if c is None]
    if missing:
        raise KeyError(f"Missing required columns (or aliases): {missing}. Available: {list(df.columns)}")

    out = df[[c_biz, c_rat, c_rev]].rename(columns={
        c_biz: "business name",
        c_rat: "rating",
        c_rev: "review text"
    })

    # normalize
    out["business name"] = out["business name"].astype(str).str.strip()
    out["review text"] = out["review text"].astype(str).str.strip()
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    return out

def clean_merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Concatenate → drop missing/empties → drop duplicates."""
    merged = pd.concat([df1, df2], ignore_index=True)

    # Treat blanks as NA
    merged.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}, inplace=True)

    # Re-strip again to catch whitespace-only
    merged["business name"] = merged["business name"].astype(str).str.strip()
    merged["review text"] = merged["review text"].astype(str).str.strip()

    # Empty strings to NA (after strip)
    merged.loc[merged["business name"] == "", "business name"] = pd.NA
    merged.loc[merged["review text"] == "", "review text"] = pd.NA

    # Drop rows missing any required field
    merged = merged.dropna(subset=["business name","rating","review text"])

    # Drop exact duplicates across the three cols
    merged = merged.drop_duplicates(subset=["business name","rating","review text"])

    return merged

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Download, clean, and merge two reviews CSVs from Google Drive.")
    parser.add_argument("--url1", required=True, help="Google Drive share URL for first CSV")
    parser.add_argument("--url2", required=True, help="Google Drive share URL for second CSV")
    parser.add_argument("--out", default="cleaned_reviews.csv", help="Output CSV path")
    parser.add_argument("--tempdir", default="downloads", help="Directory to store downloaded raw CSVs")
    args = parser.parse_args()

    tmp1 = os.path.join(args.tempdir, "file1.csv")
    tmp2 = os.path.join(args.tempdir, "file2.csv")

    print("[1/5] Downloading CSV #1...")
    download_from_drive(args.url1, tmp1)
    print("[2/5] Downloading CSV #2...")
    download_from_drive(args.url2, tmp2)

    print("[3/5] Reading CSVs...")
    df1_raw = pd.read_csv(tmp1)
    df2_raw = pd.read_csv(tmp2)
    print(f"   - df1 rows: {len(df1_raw)}, cols: {list(df1_raw.columns)}")
    print(f"   - df2 rows: {len(df2_raw)}, cols: {list(df2_raw.columns)}")

    print("[4/5] Standardizing columns...")
    df1_std = select_and_rename(df1_raw)
    df2_std = select_and_rename(df2_raw)

    print(f"   - df1_std rows: {len(df1_std)}")
    print(f"   - df2_std rows: {len(df2_std)}")

    print("[5/5] Merging & cleaning (drop missing + duplicates)...")
    cleaned = clean_merge(df1_std, df2_std)
    print(f"   - Final cleaned rows: {len(cleaned)}")

    cleaned.to_csv(args.out, index=False)
    print(f"Done. Saved to: {args.out}")

if __name__ == "__main__":
    main()
