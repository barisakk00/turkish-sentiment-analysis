import re
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pathlib import Path


TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "models/turkish_sentiment_model.pkl"
REPORT_PATH = "reports/classification_report.txt"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def clean_text(text: str) -> str:
    """
    Basic Turkish text cleaning.
    Keeps Turkish characters and removes URLs, numbers, and unnecessary symbols.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    train_df = train_df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    test_df = test_df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

    train_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].apply(clean_text)
    test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].apply(clean_text)

    X_train = train_df[TEXT_COLUMN]
    y_train = train_df[LABEL_COLUMN]

    X_test = test_df[TEXT_COLUMN]
    y_test = test_df[LABEL_COLUMN]

    print("\nTrain label distribution:")
    print(y_train.value_counts())

    print("\nTest label distribution:")
    print(y_test.value_counts())

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=3
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", round(accuracy, 4))

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(matrix)

    Path("reports").mkdir(exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {round(accuracy, 4)}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(matrix))

    print(f"\nReport saved to: {REPORT_PATH}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()