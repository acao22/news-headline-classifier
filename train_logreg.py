import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

from preprocess import prepare_data
import numpy as np

import joblib


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("y_true distribution:", Counter(y_true))
    print("y_pred distribution:", Counter(y_pred))

    # Also sanity-check unique labels
    print("unique y_true:", np.unique(y_true))
    print("unique y_pred:", np.unique(y_pred))
    print(classification_report(y_true, y_pred))
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="URL-only CSV path (must have 'url' column)")
    parser.add_argument("--max_features", type=int, default=2000)
    parser.add_argument("--ngrams", type=int, default=2, help="1=unigram, 2=up to bigram")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_csv", default=None, help="Optional extra URL-only CSV to evaluate on")
    args = parser.parse_args()

    #  1. Load the CSV files & Preprocess the data (using from preprocess.py)
    print("Loading data via prepare_data()...")
    X, y = prepare_data(args.csv_path)
    # print to check prepare_data() output print(X[:5], y[:5])

    print(f"Loaded {len(X)} examples. Class balance: fox={(sum(1 for t in y if t==0))}, nbc={(sum(1 for t in y if t==1))}")



    # 2. 80/20 train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"Train={len(X_train)}  Val={len(X_val)}")

    # 3. Convert the text data to TF-IDF features
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, args.ngrams)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 4. Train a Logistic Regression model
    lr_model = LogisticRegression(max_iter=100, class_weight='balanced')
    lr_model.fit(X_train_vec, y_train)

    # 5. Prediction on the val set & evaluate
    y_val_pred = lr_model.predict(X_val_vec)
    evaluate("LogReg (val)", y_val, y_val_pred)

    # Optional: evaluate on another URL-only dataset (same prepare_data rules)
    if args.test_csv:
        X_test, y_test = prepare_data(args.test_csv)
        X_test_vec = vectorizer.transform(X_test)  
        y_test_pred = lr_model.predict(X_test_vec)
        evaluate("LogReg (external)", y_test, y_test_pred)


    # Save the model and vectorize for model deployment
    joblib.dump(vectorizer, "logreg_vectorizer.joblib")
    joblib.dump(lr_model, "logreg_model.joblib")


if __name__ == "__main__":
    main()
