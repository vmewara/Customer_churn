# Telecom Customer Churn Prediction

## Overview

An end-to-end binary classification project to predict whether a telecom customer will churn using the WA Telco dataset. The project covers data loading, exploratory data analysis, feature engineering, model training, evaluation, and model export.

---

## Project Details

| Item | Detail |
|------|--------|
| **Problem Type** | Binary Classification |
| **Target Variable** | Churn (0 = No, 1 = Yes) |
| **Dataset** | WA_Fn-UseC_-Telco-Customer-Churn.csv |
| **Best Model** | Logistic Regression |
| **Saved Model** | log_model.pkl (joblib) |
| **Train / Test Split** | 80% / 20% — random_state=42 |

---

## Model Comparison (Recorded Results)

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression ✅ Best | 0.82 | 0.69 | 0.59 |
| Decision Tree | 0.80 | 0.61 | 0.63 |

> **Best Model: Logistic Regression** — Higher Accuracy and Precision. Saved as `log_model.pkl`

---

## Confusion Matrix — Logistic Regression (Best Model)

| | Predicted No | Predicted Yes |
|--|--------------|---------------|
| **Actual No** | 939 | 97 |
| **Actual Yes** | 151 | 222 |

- True Negatives: 939
- False Positives: 97
- False Negatives: 151
- True Positives: 222

---

## Confusion Matrix — Decision Tree

| | Predicted No | Predicted Yes |
|--|--------------|---------------|
| **Actual No** | 890 | 146 |
| **Actual Yes** | 135 | 238 |

- True Negatives: 890
- False Positives: 146
- False Negatives: 135
- True Positives: 238

---

## Pipeline

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
  └── Drop customerID column
        └── Encode Churn: No → 0, Yes → 1
              └── EDA (describe, isnull, countplot, scatterplot, heatmap)
                    └── Feature split: numerical (StandardScaler) + categorical (OneHotEncoder)
                          └── ColumnTransformer → Pipeline
                                └── Train/Test Split (80/20)
                                      ├── LogisticRegression → Evaluate → Export log_model.pkl
                                      └── DecisionTreeClassifier → Evaluate
```

---

## Key EDA Findings

- Month-to-month contract customers churn significantly more
- Higher monthly charges correlate with increased churn
- Longer tenure customers are less likely to churn
- Churn is a minority class — precision and recall used alongside accuracy

---

## Setup

```bash
pip install -r requirements.txt
```

Open the notebook in Google Colab or Jupyter and run cells top to bottom.

---

## File Structure

```
├── notebook.ipynb                              # Main notebook
├── app.py                                      # FastAPI / Streamlit app
├── WA_Fn-UseC_-Telco-Customer-Churn.csv        # Dataset
├── log_model.pkl                               # Saved best model (joblib)
├── requirements.txt
└── README.md
```

---

## Notes

- customerID was dropped before modelling as it is a non-predictive identifier
- Logistic Regression outperformed Decision Tree on Accuracy and Precision
- Decision Tree achieved higher Recall (0.63 vs 0.59) — catches more churners
- Logistic Regression selected as best model and saved for deployment
