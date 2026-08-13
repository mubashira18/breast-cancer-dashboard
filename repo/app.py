"""
Breast Cancer Classification Dashboard
Wisconsin Diagnostic Dataset

A real Streamlit application (not a notebook dump) with:
  - Prediction tab: enter tumor measurements, pick a model, get a diagnosis + probability
  - Model Comparison tab: metrics table across 6 models
  - ROC Analysis tab: combined ROC curve + per-metric bar charts
  - Interpretability tab: Random Forest feature importance + Logistic Regression coefficients
  - Test Data Evaluation tab: upload a CSV, pick a model, evaluate metrics + confusion matrix

Run with:  streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    auc,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "UCI Breast Cancer Wisconsin", "data.csv")

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="\U0001FA7A",
    layout="wide",
)

# Models that need scaled input vs. raw input, matching the original notebook
SCALED_MODELS = {"Logistic Regression", "kNN", "SVM"}


# --------------------------------------------------------------------------
# Data loading & model training (cached so this only runs once per session)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"].map({"B": 0, "M": 1})
    return df, X, y


@st.cache_resource
def train_models():
    df, X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    fitted = {}
    predictions = {}
    for name, model in models.items():
        if name in SCALED_MODELS:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        fitted[name] = model
        predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}

    rows = []
    for name, pred in predictions.items():
        y_pred, y_prob = pred["y_pred"], pred["y_prob"]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
        })
    results = pd.DataFrame(rows)

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "models": fitted,
        "predictions": predictions,
        "results": results,
    }


def feature_groups(columns):
    """Split the 30 feature names into mean / se / worst groups."""
    mean_feats = [c for c in columns if c.endswith("_mean")]
    se_feats = [c for c in columns if c.endswith("_se")]
    worst_feats = [c for c in columns if c.endswith("_worst")]
    return mean_feats, se_feats, worst_feats


def render_confusion_matrix(cm, labels=("Benign", "Malignant")):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Load everything up front
# --------------------------------------------------------------------------
try:
    state = train_models()
except FileNotFoundError:
    st.error(
        f"Could not find the dataset at `{DATA_PATH}`. "
        "Make sure `UCI Breast Cancer Wisconsin/data.csv` sits next to app.py."
    )
    st.stop()

MODEL_NAMES = list(state["models"].keys())

st.title("\U0001FA7A Breast Cancer Classification")
st.caption("Wisconsin Diagnostic Dataset")

tab_predict, tab_compare, tab_roc, tab_interpret, tab_eval = st.tabs([
    "\U0001F52C Prediction",
    "\U0001F4CA Model Comparison",
    "\U0001F4C8 ROC Analysis",
    "\U0001F50E Interpretability",
    "\U0001F4C1 Test Data Evaluation",
])

# --------------------------------------------------------------------------
# TAB 1 - Prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict Diagnosis from Tumor Measurements")

    X = state["X"]
    mean_feats, se_feats, worst_feats = feature_groups(X.columns)
    defaults = X.mean()

    def render_inputs(feat_list, group_label, cols_per_row=3):
        st.markdown(f"**{group_label}**")
        values = {}
        cols = st.columns(cols_per_row)
        for i, feat in enumerate(feat_list):
            with cols[i % cols_per_row]:
                values[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    value=float(defaults[feat]),
                    format="%.4f",
                    key=f"input_{feat}",
                )
        return values

    input_values = {}
    input_values.update(render_inputs(mean_feats, "Mean Features"))
    st.divider()
    input_values.update(render_inputs(se_feats, "SE Features"))
    st.divider()
    input_values.update(render_inputs(worst_feats, "Worst Features"))
    st.divider()

    model_choice = st.selectbox("Select Model", MODEL_NAMES, key="predict_model")
    predict_clicked = st.button("\U0001F50D Predict", type="primary")

    if predict_clicked:
        row = pd.DataFrame([input_values])[X.columns]
        model = state["models"][model_choice]

        if model_choice in SCALED_MODELS:
            row_for_model = pd.DataFrame(
                state["scaler"].transform(row), columns=X.columns
            )
        else:
            row_for_model = row

        pred = model.predict(row_for_model)[0]
        prob = model.predict_proba(row_for_model)[0]
        benign_pct, malignant_pct = prob[0] * 100, prob[1] * 100

        st.markdown("### Prediction")
        if pred == 0:
            st.success(f"\U0001F7E2 BENIGN  (using {model_choice})")
        else:
            st.error(f"\U0001F534 MALIGNANT  (using {model_choice})")

        c1, c2 = st.columns(2)
        c1.metric("Benign probability", f"{benign_pct:.1f}%")
        c2.metric("Malignant probability", f"{malignant_pct:.1f}%")
        st.progress(int(round(malignant_pct)))

# --------------------------------------------------------------------------
# TAB 2 - Model Comparison
# --------------------------------------------------------------------------
with tab_compare:
    st.subheader("Model Comparison")

    results = state["results"].set_index("Model")
    styled = results.style.format({
        "Accuracy": "{:.2%}",
        "AUC": "{:.4f}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1": "{:.2%}",
        "MCC": "{:.4f}",
    }).highlight_max(axis=0, color="lightgreen")
    st.dataframe(styled, use_container_width=True)

    best_model = results["Accuracy"].idxmax()
    st.markdown(
        f"\U0001F3C6 **{best_model}** currently has the highest test accuracy "
        f"({results.loc[best_model, 'Accuracy']:.2%})."
    )

# --------------------------------------------------------------------------
# TAB 3 - ROC Analysis
# --------------------------------------------------------------------------
with tab_roc:
    st.subheader("ROC Curve Comparison")

    y_test = state["y_test"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pred in state["predictions"].items():
        fpr, tpr, _ = roc_curve(y_test, pred["y_prob"])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Metric Comparison Across Models")
    metric_choice = st.selectbox(
        "Metric", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    )
    chart_df = state["results"].set_index("Model")[[metric_choice]].sort_values(
        metric_choice, ascending=False
    )
    st.bar_chart(chart_df)

# --------------------------------------------------------------------------
# TAB 4 - Interpretability
# --------------------------------------------------------------------------
with tab_interpret:
    st.subheader("Model Interpretability")

    rf_model = state["models"]["Random Forest"]
    lr_model = state["models"]["Logistic Regression"]
    X = state["X"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Random Forest — Top 10 Feature Importances**")
        rf_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf_model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(10)
        top_rf = rf_importance.sort_values("Importance")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top_rf["Feature"], top_rf["Importance"], color="seagreen")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "How much each feature reduces impurity across the Random Forest's trees."
        )

    with col2:
        st.markdown("**Logistic Regression — Top 10 Coefficients**")
        logreg_coef = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": lr_model.coef_[0],
        })
        logreg_coef["Absolute_Coefficient"] = logreg_coef["Coefficient"].abs()
        logreg_coef = logreg_coef.sort_values(
            "Absolute_Coefficient", ascending=False
        ).head(10)
        top_lr = logreg_coef.sort_values("Coefficient")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top_lr["Feature"], top_lr["Coefficient"], color="steelblue")
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_xlabel("Coefficient")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Effect of a one-unit increase in each standardized feature on the "
            "log-odds of malignancy, holding other features constant."
        )

# --------------------------------------------------------------------------
# TAB 5 - Test Data Evaluation
# --------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluate a Test Dataset")
    st.write(
        "Upload a CSV with the same 30 feature columns as the training data. "
        "Include a `diagnosis` column (B/M) if you want metrics computed; "
        "otherwise you'll just get predictions."
    )

    uploaded_file = st.file_uploader("Upload test data", type=["csv"])
    eval_model_choice = st.selectbox("Select Model", MODEL_NAMES, key="eval_model")
    evaluate_clicked = st.button("Evaluate", type="primary")

    if uploaded_file is not None and evaluate_clicked:
        try:
            test_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            st.stop()

        drop_cols = [c for c in ["id", "Unnamed: 32"] if c in test_df.columns]
        test_df = test_df.drop(columns=drop_cols)

        X_cols = state["X"].columns
        missing = [c for c in X_cols if c not in test_df.columns]
        if missing:
            st.error(
                f"Uploaded file is missing {len(missing)} required feature column(s), "
                f"e.g. {missing[:5]}"
            )
            st.stop()

        has_labels = "diagnosis" in test_df.columns
        X_upload = test_df[X_cols]
        y_upload = None
        if has_labels:
            raw_labels = test_df["diagnosis"]
            if raw_labels.dtype == object:
                # Labels look like 'B'/'M'
                y_upload = raw_labels.map({"B": 0, "M": 1})
            else:
                # Labels are already numeric (0 = benign, 1 = malignant)
                y_upload = raw_labels.astype(int)
            if y_upload.isnull().any():
                st.error(
                    "Could not interpret the `diagnosis` column — expected 'B'/'M' "
                    "or 0/1 values."
                )
                st.stop()

        model = state["models"][eval_model_choice]
        if eval_model_choice in SCALED_MODELS:
            X_upload_model = pd.DataFrame(
                state["scaler"].transform(X_upload), columns=X_cols
            )
        else:
            X_upload_model = X_upload

        y_pred_upload = model.predict(X_upload_model)
        y_prob_upload = model.predict_proba(X_upload_model)[:, 1]

        st.markdown(f"**Uploaded dataset:** {test_df.shape[0]} rows x {test_df.shape[1]} columns")

        if has_labels:
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{accuracy_score(y_upload, y_pred_upload):.2%}")
            m2.metric("AUC", f"{roc_auc_score(y_upload, y_prob_upload):.4f}")
            m3.metric("Precision", f"{precision_score(y_upload, y_pred_upload):.2%}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Recall", f"{recall_score(y_upload, y_pred_upload):.2%}")
            m5.metric("F1", f"{f1_score(y_upload, y_pred_upload):.2%}")
            m6.metric("MCC", f"{matthews_corrcoef(y_upload, y_pred_upload):.4f}")

            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_upload, y_pred_upload)
            render_confusion_matrix(cm)
        else:
            st.info("No `diagnosis` column found — showing predictions only.")
            out = test_df.copy()
            out["Predicted"] = np.where(y_pred_upload == 0, "Benign", "Malignant")
            out["Malignant_Probability"] = y_prob_upload
            st.dataframe(out, use_container_width=True)
    elif evaluate_clicked and uploaded_file is None:
        st.warning("Please upload a CSV file first.")
