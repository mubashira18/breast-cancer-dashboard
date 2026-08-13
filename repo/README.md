# Breast Cancer Classification Dashboard

A Streamlit app for classifying breast tumors as Benign or Malignant using the
UCI Breast Cancer Wisconsin (Diagnostic) dataset, with six trained models
(Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, Random Forest, SVM).

## Structure

```
├── app.py                              # Streamlit dashboard
├── requirements.txt
├── UCI Breast Cancer Wisconsin/
│   └── data.csv                        # training dataset (add this yourself)
├── test_data.csv                       # sample evaluation data
└── model/                              # (optional) saved model artifacts
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **Prediction** — enter tumor measurements, pick a model, get a diagnosis + probability
- **Model Comparison** — accuracy/AUC/precision/recall/F1/MCC across all six models
- **ROC Analysis** — combined ROC curve and per-metric comparison
- **Interpretability** — Random Forest feature importance, Logistic Regression coefficients
- **Test Data Evaluation** — upload a CSV, pick a model, see metrics + confusion matrix
