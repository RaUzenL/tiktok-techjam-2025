#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean TWO review CSVs from Google Drive and merge them.

Key fixes:
- Prefer 'stars' over 'rating' when both exist (some 'rating' cols are strings like '4/5')
- Parse rating strings like '4/5', '10/10' into numeric
- Strictly drop missing values (incl. blanks/whitespace) and duplicates
- Print per-source stats so it's obvious BOTH datasets were used

Final columns:
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

def parse_rating_series(s: pd.Series) -> pd.Series:
    """Coerce ratings into numeric; supports numbers and 'x/y' strings."""
    if s.dtype != "O":
        return pd.to_numeric(s, errors="coerce")

    def parse_val(x):
        if pd.isna(x):
            return pd.NA
        t = str(x).strip()
        if t == "" or t.lower() in ("nan", "none"):
            return pd.NA
        # fraction like 4/5, 10/10
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", t)
        if m:
            # Keep numerator as the rating (no normalization)
            return float(m.group(1))
        # plain number string
        try:
            return float(t)
        except Exception:
            return pd.NA

    return s.apply(parse_val)

def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to ['business name','rating','review text'] using robust alias matching.
       Prefer 'stars' over 'rating' for the rating column.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(possibles: List[str]) -> Optional[str]:
        for p in possibles:
            if p in cols_lower:
                return cols_lower[p]
        return None

    # Prefer 'stars' first, then 'rating'
    c_biz = pick(["business name","business_name","title","name","place","restaurant","poi","shop"])
    c_rat = pick(["stars","rating","score","star","rate"])
    c_rev = pick(["review text","review","text","comment","content","body"])

    missing = [n for n,c in zip(["business name","rating","review text"], [c_biz,c_rat,c_rev]) if c is None]
    if missing:
        raise KeyError(f"Missing required columns (or aliases): {missing}. Available: {list(df.columns)}")

    out = df[[c_biz, c_rat, c_rev]].rename(columns={
        c_biz: "business name",
        c_rat: "rating",
        c_rev: "review text"
    })

    # Normalize fields
    out["business name"] = out["business name"].astype(str).str.strip()
    out["review text"]   = out["review text"].astype(str).str.strip()
    out["rating"]        = parse_rating_series(out["rating"])

    return out

def clean_and_merge(df1_std: pd.DataFrame, df2_std: pd.DataFrame) -> pd.DataFrame:
    """Concatenate, drop missing/empties, drop duplicates."""
    merged = pd.concat([df1_std.assign(__src="google_places"),
                        df2_std.assign(__src="london")],
                       ignore_index=True)

    # Treat blanks as NA
    merged.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}, inplace=True)

    # Re-strip again to catch whitespace-only
    merged["business name"] = merged["business name"].astype(str).str.strip()
    merged["review text"]   = merged["review text"].astype(str).str.strip()

    # Empty strings to NA (after strip)
    merged.loc[merged["business name"] == "", "business name"] = pd.NA
    merged.loc[merged["review text"]   == "", "review text"]   = pd.NA

    # Keep only fully-populated rows
    merged = merged.dropna(subset=["business name","rating","review text"])

    # Deduplicate across the 3 content columns (keep first = google_places wins on ties)
    merged = merged.drop_duplicates(subset=["business name","rating","review text"], keep="first")

    return merged

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Download, clean, and merge two reviews CSVs from Google Drive.")
    parser.add_argument("--url1", required=True, help="Google Drive share URL for first CSV (google places crawler)")
    parser.add_argument("--url2", required=True, help="Google Drive share URL for second CSV (london dataset)")
    parser.add_argument("--out", default="cleaned_reviews.csv", help="Output CSV path")
    parser.add_argument("--tempdir", default="downloads", help="Directory to store downloaded raw CSVs")
    args = parser.parse_args()

    tmp1 = os.path.join(args.tempdir, "file1.csv")
    tmp2 = os.path.join(args.tempdir, "file2.csv")

    print("[1/6] Downloading CSV #1 (google places)...")
    download_from_drive(args.url1, tmp1)
    print("[2/6] Downloading CSV #2 (london)...")
    download_from_drive(args.url2, tmp2)

    print("[3/6] Reading CSVs...")
    df1_raw = pd.read_csv(tmp1)
    df2_raw = pd.read_csv(tmp2)
    print(f"   - df1_raw shape: {df1_raw.shape}")
    print(f"   - df2_raw shape: {df2_raw.shape}")

    print("[4/6] Standardizing columns...")
    df1_std = select_and_rename(df1_raw)
    df2_std = select_and_rename(df2_raw)
    print(f"   - df1_std cols: {list(df1_std.columns)} rows: {len(df1_std)}")
    print(f"   - df2_std cols: {list(df2_std.columns)} rows: {len(df2_std)}")

    # Pre-clean completeness per source (non-null across the three columns)
    valid1 = df1_std.dropna(subset=["business name","rating","review text"])
    valid1 = valid1[(valid1["business name"].astype(str).str.strip()!="") & (valid1["review text"].astype(str).str.strip()!="")]
    valid2 = df2_std.dropna(subset=["business name","rating","review text"])
    valid2 = valid2[(valid2["business name"].astype(str).str.strip()!="") & (valid2["review text"].astype(str).str.strip()!="")]
    print(f"   - df1 valid rows before merge: {len(valid1)}")
    print(f"   - df2 valid rows before merge: {len(valid2)}")

    print("[5/6] Merging & cleaning (drop missing + duplicates)...")
    cleaned = clean_and_merge(df1_std, df2_std)
    print(f"   - After cleaning total rows: {len(cleaned)}")

    # Contribution by source after dedup
    src_counts = cleaned["__src"].value_counts().to_dict()
    print(f"   - Contribution after dedup by source: {src_counts}")

    print("[6/6] Saving...")
    cleaned.drop(columns=["__src"]).to_csv(args.out, index=False)
    print(f"Done. Saved to: {args.out}")

if __name__ == "__main__":
    main()
