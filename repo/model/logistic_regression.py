"""
Logistic Regression classifier.
Run this file directly to train, evaluate, and save the model:
    python model/logistic_regression.py
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression

from preprocessing import get_train_test_split, evaluate, RANDOM_STATE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic_regression.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")


def train():
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, scaler) = get_train_test_split()

    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    print("=== Logistic Regression ===")
    metrics = evaluate(model, X_test_scaled, y_test)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    return model, metrics


if __name__ == "__main__":
    train()
