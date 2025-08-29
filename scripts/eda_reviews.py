import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV   = os.path.join(BASE_DIR, "csv_output", "reviews_clean_en.csv")
OUT_DIR  = os.path.join(BASE_DIR, "csv_output", "eda")

os.makedirs(OUT_DIR, exist_ok=True)

def plot_distribution(df, col, title, filename, bins=None):
    plt.figure(figsize=(6,4))
    if bins:
        sns.histplot(df[col], bins=bins, kde=False)
    else:
        sns.countplot(x=col, data=df)
    plt.title(title)
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, filename)
    plt.savefig(outpath)
    plt.close()
    print(f"Saved plot → {outpath}")

def main():
    if not os.path.exists(IN_CSV):
        print(f"File not found: {IN_CSV}")
        return

    df = pd.read_csv(IN_CSV)

    print("=== Dataset Overview ===")
    print(df.info())
    print(df.head(5))

    # 1. Rating distribution
    plot_distribution(df, "rating", "Distribution of Ratings", "ratings_dist.png")

    # 2. Review length distribution
    if "review_len_words" in df.columns:
        plot_distribution(df, "review_len_words",
                          "Distribution of Review Length (words)",
                          "review_length.png", bins=30)

    # 3. Flags: advertisement, irrelevant, rant
    for col in ["flag_advertisement", "flag_irrelevant", "flag_rant_without_visit"]:
        if col in df.columns:
            plot_distribution(df, col,
                              f"Distribution of {col}",
                              f"{col}.png")

    # 4. Correlation heatmap (numerical features only)
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, "correlation_heatmap.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved plot → {outpath}")

    print("\nEDA completed. Plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
