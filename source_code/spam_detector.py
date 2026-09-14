"""
Spam Message Detection using NLP
Simple application-based mini project for an NLP course.

Workflow:
1. Load labelled SMS/message data
2. Split data into training and testing sets
3. Convert text into numerical TF-IDF features
4. Train a Logistic Regression classifier
5. Evaluate the model
6. Predict new messages
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = os.path.join("dataset", "spam_messages.csv")
OUTPUT_DIR = "output"


def load_data():
    """Load the labelled message dataset."""
    return pd.read_csv(DATA_PATH)


def train_model(data):
    """Train a TF-IDF + Logistic Regression NLP pipeline."""
    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, y_test, predictions


def evaluate_model(y_test, predictions):
    """Print and save evaluation results."""
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

    print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
    print("\nClassification Report:\n", report)
    print("Confusion Matrix [rows=true, columns=predicted]:")
    print(matrix)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "model_results.txt"), "w", encoding="utf-8") as f:
        f.write("Spam Message Detection Results\n")
        f.write("=" * 35 + "\n")
        f.write("Accuracy: {:.2f}%\n\n".format(accuracy * 100))
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix [ham, spam]:\n")
        f.write(str(matrix))

    return accuracy


def predict_message(model, message):
    """Predict whether a new message is spam or ham."""
    prediction = model.predict([message])[0]
    return prediction


def main():
    data = load_data()

    print("Dataset size:", len(data))
    print("\nClass distribution:")
    print(data["label"].value_counts())

    model, y_test, predictions = train_model(data)
    evaluate_model(y_test, predictions)

    test_messages = [
        "Congratulations! You have won a free gift. Claim now!",
        "Please send me the notes after class."
    ]

    print("\nSample Predictions:")
    for message in test_messages:
        result = predict_message(model, message)
        print("\nMessage:", message)
        print("Prediction:", "SPAM" if result == "spam" else "NOT SPAM")


if __name__ == "__main__":
    main()
