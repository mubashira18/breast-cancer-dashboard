"""
Random Forest (ensemble) classifier.
Run this file directly to train, evaluate, and save the model:
    python model/random_forest.py
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocessing import get_train_test_split, evaluate, RANDOM_STATE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest.pkl")


def train():
    X_train, X_test, y_train, y_test, _, _, _ = get_train_test_split()

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    print("=== Random Forest ===")
    metrics = evaluate(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    return model, metrics


if __name__ == "__main__":
    train()
