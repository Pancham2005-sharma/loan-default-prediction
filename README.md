# Loan Default Prediction

A Streamlit web application that predicts the probability of loan default
based on applicant and loan details, using a trained XGBoost model.

## Project Background

This is the deployment stage of an end-to-end Data Science capstone project.
The model was trained on the Kaggle Loan Default Dataset (148,670 records).

**Key steps in the pipeline (see the accompanying Jupyter notebook for full detail):**
- Missing value imputation (median for numeric, mode for categorical)
- One-hot encoding of categorical features
- SMOTE applied to the training set to address class imbalance (75% no-default / 25% default)
- **Data leakage detected and corrected**: `rate_of_interest`, `Interest_rate_spread`,
  and `Upfront_charges` were removed after an initial model scored a suspicious 100%
  accuracy — these fields are only populated post-approval and were leaking the outcome.
- Three models compared: Logistic Regression, Random Forest, and XGBoost
- **XGBoost selected** as the final model (89.8% accuracy, 0.895 ROC-AUC, 0.65 recall on defaults)

## Folder Structure
'''
loan-default-app/
│
├── app.py
├── model/
│ ├── loan_model.pkl
│ ├── scaler.pkl
│ ├── loan_default_columns.pkl
│ └── loan_default_categorical_options.pkl
├── utils/
│ └── preprocessing.py
├── data/
│ └── README.txt
├── requirements.txt
└── README.md
'''


## Setup & Running Locally

1. Clone this repository.
2. Install dependencies:
3. Run the app:
4. The app opens in your browser at `http://localhost:8501`.

## App Features

**Inputs:** Application year, loan amount, term, property value, income, credit score,
loan-to-value ratio, debt-to-income ratio, plus all categorical loan/applicant attributes
(loan type, purpose, region, occupancy type, etc.) used during training.

**Outputs:**
- Default Probability (%)
- Risk Category (Low / Medium / High)
- Suggested Decision (Approve / Reject)

## Model Comparison

| Model | Accuracy | ROC-AUC | Default Recall |
|---|---|---|---|
| Logistic Regression | 85.7% | 0.813 | 0.54 |
| Random Forest | 88.8% | 0.882 | 0.61 |
| **XGBoost (selected)** | **89.8%** | **0.895** | **0.65** |

## Limitations

- The model misses roughly 35% of actual defaults (recall = 0.65). It should support,
  not replace, human underwriting decisions.
- Trained on historical data through 2019; predictions may not reflect newer market
  conditions without retraining.
