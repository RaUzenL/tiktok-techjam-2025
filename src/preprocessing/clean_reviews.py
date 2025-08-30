import os
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Make results reproducible for langdetect
DetectorFactory.seed = 42

# --- Robust paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV   = os.path.join(BASE_DIR, "csv_output", "reviews_processed.csv")
OUT_DIR  = os.path.join(BASE_DIR, "csv_output")
OUT_CSV  = os.path.join(OUT_DIR, "reviews_clean_en.csv")
STATS_TXT = os.path.join(OUT_DIR, "cleaning_stats.txt")

os.makedirs(OUT_DIR, exist_ok=True)

def safe_detect_lang(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unk"
    try:
        return detect(text)
    except LangDetectException:
        return "unk"

def main():
    if not os.path.exists(IN_CSV):
        print(f"Input file not found: {IN_CSV}")
        return

    df = pd.read_csv(IN_CSV)

    # 1) Duplicates: define a key that makes sense for this dataset
    before_rows = len(df)
    df["__dupe_key"] = (
        df["business_name"].astype(str).str.strip() + "||" +
        df["author_name"].astype(str).str.strip()  + "||" +
        df["text"].astype(str).str.strip()
    )
    dupes = df.duplicated(subset="__dupe_key", keep="first")
    n_dupes = dupes.sum()
    df = df[~dupes].drop(columns="__dupe_key")
    after_dupe_rows = len(df)

    # 2) Language detection
    df["lang"] = df["text"].apply(safe_detect_lang)

    # Keep English only for the first model pass (you can change later)
    before_lang_rows = len(df)
    df_en = df[df["lang"] == "en"].copy()
    after_lang_rows = len(df_en)

    # Save results
    df_en.to_csv(OUT_CSV, index=False)

    # Write simple stats for your README/docs
    with open(STATS_TXT, "w") as f:
        f.write("Cleaning stats\n")
        f.write("================\n")
        f.write(f"Input rows: {before_rows}\n")
        f.write(f"Removed duplicates: {n_dupes}\n")
        f.write(f"Rows after de-dup: {after_dupe_rows}\n")
        f.write(f"Rows English-only: {after_lang_rows} (from {before_lang_rows})\n")
        f.write("Language counts (all rows after de-dup):\n")
        f.write(str(df["lang"].value_counts()) + "\n")

    print(f"Saved cleaned English-only CSV → {OUT_CSV}")
    print(f"Wrote stats → {STATS_TXT}")

if __name__ == "__main__":
    main()
