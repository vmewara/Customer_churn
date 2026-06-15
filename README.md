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
| **Best Model** | Decision Tree Classifier |
| **Saved Model** | log_model.pkl (joblib) |
| **Train / Test Split** | 80% / 20% — random_state=42 |

---

## Model Comparison (Recorded Results)

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Decision Tree ✅ Best | 0.80 | 0.61 | 0.63 |

---

## Confusion Matrix

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
                                      └── DecisionTreeClassifier → Evaluate → Export log_model.pkl
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

 app.py
├── log_model.pkl                               # Saved best model (joblib)
├── requirements.txt
└── README.md
```

---

## Notes

- customerID was dropped before modelling as it is a non-predictive identifier
- Decision Tree achieved accuracy of 0.80 with balanced precision and recall
- Model captures 238 true churners out of 373 actual churners (recall 63%)
- Logistic Regression was planned for comparison but Decision Tree was the executed and saved model
