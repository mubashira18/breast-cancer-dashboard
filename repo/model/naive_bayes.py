"""
Gaussian Naive Bayes classifier.
Run this file directly to train, evaluate, and save the model:
    python model/naive_bayes.py
"""

import os
import joblib
from sklearn.naive_bayes import GaussianNB

from preprocessing import get_train_test_split, evaluate

MODEL_PATH = os.path.join(os.path.dirname(__file__), "naive_bayes.pkl")


def train():
    X_train, X_test, y_train, y_test, _, _, _ = get_train_test_split()

    model = GaussianNB()
    model.fit(X_train, y_train)

    print("=== Gaussian Naive Bayes ===")
    metrics = evaluate(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    return model, metrics


if __name__ == "__main__":
    train()
