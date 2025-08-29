import os
import re
import hashlib
import pandas as pd

# --- Paths (robust to where you run from) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV  = os.path.join(BASE_DIR, "data", "reviews.csv")
OUT_DIR  = os.path.join(BASE_DIR, "csv_output")
OUT_CSV  = os.path.join(OUT_DIR, "reviews_processed.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# --- Helpers ---
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_REGEX = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
PROMO_WORDS = [
    "discount", "promo", "promotion", "deal", "sale", "coupon", "code",
    "visit us at", "follow us", "dm us", "order now", "book now", "call now",
    "open 24/7", "limited time", "free delivery", "free shipping"
]
NO_VISIT_PHRASES = [
    "never been", "haven't been", "have not been", "didn't go",
    "did not go", "i heard", "someone told me", "they say", "people say"
]
OFF_TOPIC_HINTS = [
    # crude off-topic hints you can refine later
    "my phone", "phone camera", "laptop", "headphones", "ios", "android",
    "windows update", "crypto", "stock market", "football match", "politics"
]

def stable_id(row):
    h = hashlib.sha256()
    h.update((str(row.get("business_name","")) + "||" + str(row.get("author_name","")) + "||" + str(row.get("text",""))).encode("utf-8"))
    return h.hexdigest()[:16]

def clean_text(s):
    if not isinstance(s, str):
        return ""
    # normalize whitespace and punctuation spacing
    t = s.replace("\r", " ").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def pct_upper(s):
    if not s: return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters: return 0.0
    uppers = sum(1 for c in letters if c.isupper())
    return uppers / len(letters)

def contains_any(text, phrases):
    t = text.lower()
    return any(p in t for p in phrases)

def has_url(text):
    return URL_REGEX.search(text) is not None

def has_email(text):
    return EMAIL_REGEX.search(text) is not None

def has_phone(text):
    return PHONE_REGEX.search(text) is not None

def exclaim_count(text):
    return text.count("!")

def word_count(text):
    return len(text.split()) if text else 0

def avg_word_len(text):
    words = text.split()
    return sum(len(w) for w in words)/len(words) if words else 0.0

def advertisement_flag(text):
    t = text.lower()
    if has_url(t) or has_email(t) or has_phone(t):
        return 1
    if contains_any(t, PROMO_WORDS):
        return 1
    return 0

def irrelevant_flag(text):
    # crude heuristics: extremely short OR off-topic hints
    wc = word_count(text)
    if wc < 4:
        return 1
    if contains_any(text, OFF_TOPIC_HINTS):
        return 1
    return 0

def rant_without_visit_flag(text):
    # phrases implying the reviewer might not have visited
    return 1 if contains_any(text, NO_VISIT_PHRASES) else 0

def main():
    if not os.path.exists(RAW_CSV):
        print(f"File not found: {RAW_CSV}")
        return

    df = pd.read_csv(RAW_CSV)

    # Basic normalize
    df["text"] = df["text"].apply(clean_text)

    # Stable ids so you can join later
    df["review_id"] = df.apply(stable_id, axis=1)

    # Metadata features
    df["review_len_words"]   = df["text"].apply(word_count)
    df["avg_word_length"]    = df["text"].apply(avg_word_len)
    df["pct_uppercase"]      = df["text"].apply(pct_upper)
    df["num_exclaim"]        = df["text"].apply(exclaim_count)
    df["has_url"]            = df["text"].apply(has_url).astype(int)
    df["has_email"]          = df["text"].apply(has_email).astype(int)
    df["has_phone"]          = df["text"].apply(has_phone).astype(int)

    # Weak policy labels (rule-based)
    df["flag_advertisement"]      = df["text"].apply(advertisement_flag)
    df["flag_irrelevant"]         = df["text"].apply(irrelevant_flag)
    df["flag_rant_without_visit"] = df["text"].apply(rant_without_visit_flag)

    # Optional: binary “short review” signal
    df["is_very_short"] = (df["review_len_words"] < 6).astype(int)

    # Save
    keep_cols = [
        "review_id", "business_name", "author_name", "text", "rating",
        "review_len_words", "avg_word_length", "pct_uppercase", "num_exclaim",
        "has_url", "has_email", "has_phone",
        "flag_advertisement", "flag_irrelevant", "flag_rant_without_visit",
        "is_very_short"
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df[existing].to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(df[existing].head(10))

if __name__ == "__main__":
    main()
