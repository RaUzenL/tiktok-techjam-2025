import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV   = os.path.join(BASE_DIR, "csv_output", "reviews_labeled.csv")
OUT_DIR  = os.path.join(BASE_DIR, "csv_output", "reports")
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = ["label_advertisement", "label_irrelevant", "label_rant_without_visit"]
MIN_POS = 5

def safe_split(df, y, test_size=0.2, random_state=42):
    counts = Counter(y)
    stratify = y if min(counts.values()) >= 2 else None
    return train_test_split(df, y, test_size=test_size, random_state=random_state, stratify=stratify)

def plot_cm(cm, title, path):
    plt.figure(figsize=(3.2,3.2))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0","1"])
    plt.yticks(tick_marks, ["0","1"])
    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def train_one(df, target):
    print(f"\n=== Training for {target} ===")
    y = df[target]
    pos = int(y.sum())
    neg = len(y) - pos
    print(f"Class counts → 0: {neg}, 1: {pos}")
    if pos < MIN_POS:
        print(f"Skipping {target}: only {pos} positive(s).")
        return

    X_train, X_test, y_train, y_test = safe_split(df, y)

    text_vec = TfidfVectorizer(max_features=6000, ngram_range=(1,2), stop_words="english")
    num_cols = ["review_len_words", "avg_word_length", "pct_uppercase", "num_exclaim"]

    pre = ColumnTransformer([
        ("text", text_vec, "text"),
        ("num", StandardScaler(), num_cols)
    ], remainder="drop")

    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    clf.fit(X_train, y_train)

    # Default 0.5 threshold
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)
    print(report)

    # Save report + confusion matrix
    task_tag = target.replace("label_", "")
    with open(os.path.join(OUT_DIR, f"report_{task_tag}.txt"), "w") as f:
        f.write(report)
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, f"CM @{task_tag} (thr=0.5)", os.path.join(OUT_DIR, f"cm_{task_tag}_0.5.png"))

    # Optional: tune threshold to favor precision (or recall)
    y_prob = clf.predict_proba(X_test)[:,1]
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.3, 0.7, 21):
        yp = (y_prob >= thr).astype(int)
        # compute simple F1 for positive class
        tp = ((yp==1)&(y_test==1)).sum()
        fp = ((yp==1)&(y_test==0)).sum()
        fn = ((yp==0)&(y_test==1)).sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    yp_best = (y_prob >= best_thr).astype(int)
    tuned_report = classification_report(y_test, yp_best, digits=3, zero_division=0)
    print(f"\n-- Tuned threshold {best_thr:.2f} (max F1 on val) --\n{tuned_report}")
    with open(os.path.join(OUT_DIR, f"report_{task_tag}_tuned.txt"), "w") as f:
        f.write(f"Best threshold: {best_thr:.2f}\n\n{tuned_report}")
    cm2 = confusion_matrix(y_test, yp_best)
    plot_cm(cm2, f"CM @{task_tag} (thr={best_thr:.2f})", os.path.join(OUT_DIR, f"cm_{task_tag}_{best_thr:.2f}.png"))

def main():
    if not os.path.exists(IN_CSV):
        print(f"Missing: {IN_CSV}")
        return
    df = pd.read_csv(IN_CSV)

    need = ["text", "review_len_words", "avg_word_length", "pct_uppercase", "num_exclaim"]
    for c in need + TASKS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    for t in TASKS:
        train_one(df, t)

if __name__ == "__main__":
    main()
