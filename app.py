"""
Simple command-line Spam Detector.
Run from the project root:
    python source_code/app.py
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.path.join("dataset", "spam_messages.csv")

data = pd.read_csv(DATA_PATH)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data["message"])
y = data["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("======================================")
print("       NLP SPAM MESSAGE DETECTOR")
print("======================================")
print("Type a message and press Enter.")
print("Type 'exit' to close the program.\n")

while True:
    message = input("Enter message: ")

    if message.lower() == "exit":
        print("Program closed.")
        break

    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]

    if prediction == "spam":
        print("Result: SPAM\n")
    else:
        print("Result: NOT SPAM\n")
