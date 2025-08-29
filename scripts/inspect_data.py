import pandas as pd
import os

# Paths (data folder is inside scripts/)
RAW_DATA_PATH = "data/reviews.csv"  # relative to scripts/

def main():
    # Check if file exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"File not found: {RAW_DATA_PATH}")
        return

    # Load CSV
    df = pd.read_csv(RAW_DATA_PATH)

    # Quick overview
    print("=== Data Overview ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types and missing values:")
    print(df.info())

    print("\nMissing values per column:")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
