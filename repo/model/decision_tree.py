"""
Decision Tree classifier.
Run this file directly to train, evaluate, and save the model:
    python model/decision_tree.py
"""

import os
import joblib
from sklearn.tree import DecisionTreeClassifier

from preprocessing import get_train_test_split, evaluate, RANDOM_STATE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_tree.pkl")


def train():
    X_train, X_test, y_train, y_test, _, _, _ = get_train_test_split()

    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    print("=== Decision Tree ===")
    metrics = evaluate(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    return model, metrics


if __name__ == "__main__":
    train()
