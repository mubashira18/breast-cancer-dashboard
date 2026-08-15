

## a. Problem Statement

Breast cancer diagnosis relies on correctly classifying tumors as **Benign** or **Malignant** based on measurements extracted from digitized images of a fine needle aspirate (FNA) of a breast mass. This project builds and compares six machine learning classification models to predict tumor diagnosis, and deploys the best-performing models in an interactive Streamlit dashboard that supports live prediction, model comparison, ROC analysis, feature interpretability, and evaluation on new test data.

## b. Dataset Description

The dataset used is the **UCI Breast Cancer Wisconsin (Diagnostic) Dataset**, containing 569 samples with 30 numeric features computed from cell nuclei present in the FNA images. Features are grouped into three categories per measurement (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension):

- **Mean** features
- **Standard Error (SE)** features
- **Worst** (largest) features

The target variable is `diagnosis`, with two classes: **B (Benign)** and **M (Malignant)**.


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

## c. GitHub Repository Link

[1 mark for maintaining the GitHub repo with all required files]

`https://github.com/mubashira18/breast-cancer-dashboard.git`

## d. Models Used

Six classification models were trained and evaluated on a held-out test split: Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, Random Forest (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 96.49% | 0.9960 | 97.50% | 92.86% | 95.12% | 0.9245 |
| Decision Tree | 92.98% | 0.9246 | 90.48% | 90.48% | 90.48% | 0.8492 |
| kNN | 95.61% | 0.9823 | 97.44% | 90.48% | 93.83% | 0.9058 |
| Naive Bayes | 93.86% | 0.9934 | 100.00% | 83.33% | 90.91% | 0.8715 |
| Random Forest (Ensemble) | 97.37% | 0.9929 | 100.00% | 92.86% | 96.30% | 0.9442 |


### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the best AUC (0.9960) of all models, with strong precision (97.50%), indicating it separates the two classes very well and rarely misclassifies benign cases as malignant. |
| Decision Tree | The weakest performer, with the lowest accuracy (92.98%) and AUC (0.9246), suggesting it overfits more to the training data compared to the ensemble and margin-based models. |
| kNN | Performs solidly with 95.61% accuracy and high precision (97.44%), though its recall (90.48%) is a little lower, meaning it misses slightly more malignant cases than the top models. |
| Naive Bayes | Reaches perfect precision (100%) but the lowest recall (83.33%) of all models — it is very conservative and avoids false positives, at the cost of missing some actual malignant cases. |
| Random Forest (Ensemble) | Best overall by Accuracy (97.37%), F1 (96.30%), and MCC (0.9442), combining strong precision and recall consistently. |

| **Overall Winner for your dataset?** | **Random Forest** is the overall winner, leading on Accuracy, F1, and MCC, while Logistic Regression edges it out slightly on AUC alone. All models learned meaningful, non-random patterns, the ROC curves are strong across the board, confusion matrices show relatively few errors, and feature-importance analysis (Random Forest impurity reduction, Logistic Regression standardized coefficients) produced sensible results. A tuned Random Forest (via GridSearchCV) did not improve test performance over the baseline, so the baseline Random Forest was kept as the final model. 