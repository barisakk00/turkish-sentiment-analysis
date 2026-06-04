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
MODELS_DIR = "models"
REPORTS_DIR = "reports"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Feature counts to iterate through
FEATURE_COUNTS = [1000, 5000, 10000, 20000, 30000]


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

    # Create directories if they don't exist
    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path(REPORTS_DIR).mkdir(exist_ok=True)

    # Store results for comparison
    all_results = []

    # Iterate through different feature counts
    for feature_count in FEATURE_COUNTS:
        print(f"\n{'='*60}")
        print(f"Training model with {feature_count} features...")
        print(f"{'='*60}")

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=feature_count,
                ngram_range=(1, 2),
                min_df=3
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {round(accuracy, 4)}")

        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        print("\nClassification Report:")
        print(report)

        print("\nConfusion Matrix:")
        print(matrix)

        # Save model with feature count in name
        model_name = f"turkish_sentiment_model_{feature_count}_features.pkl"
        model_path = Path(MODELS_DIR) / model_name
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")

        # Save report with feature count in name
        report_name = f"classification_report_{feature_count}_features.txt"
        report_path = Path(REPORTS_DIR) / report_name
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Features: {feature_count}\n")
            f.write(f"Accuracy: {round(accuracy, 4)}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(matrix))
        print(f"Report saved to: {report_path}")

        # Store results for comparison
        all_results.append({
            "features": feature_count,
            "accuracy": round(accuracy, 4),
            "model_path": str(model_path),
            "report_path": str(report_path)
        })

    # Print summary of all models
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL MODELS")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))

    # Save summary to file
    summary_path = Path(REPORTS_DIR) / "models_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SUMMARY OF ALL TRAINED MODELS\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()