# Loan Default Prediction App

A Streamlit web application that predicts the probability of loan default
based on applicant and loan details, using a trained XGBoost model.

## Project Background

This app is the deployment stage of an end-to-end Data Science capstone project.
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

```
loan-default-app/
│
├── app.py                     # Main Streamlit application
├── model/
│   ├── loan_model.pkl              # Trained XGBoost model
│   ├── scaler.pkl                  # StandardScaler (fit during training)
│   ├── loan_default_columns.pkl    # Exact column order the model expects
│   └── loan_default_categorical_options.pkl  # Valid values for each dropdown
├── utils/
│   └── preprocessing.py       # Rebuilds a model-ready row from form inputs
├── data/
│   └── sample.csv             # (Optional) sample rows for reference/testing
├── requirements.txt
└── README.md
```

## Setup & Running Locally

1. Make sure the four `.pkl` files generated in the training notebook are placed
   inside the `model/` folder (they are not included in this repo — generate them
   by running the notebook, which saves them via `joblib.dump`).

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. The app will open in your browser, typically at `http://localhost:8501`.

## App Features

**Inputs:** Application year, loan amount, term, property value, income, credit score,
loan-to-value ratio, debt-to-income ratio, plus all categorical loan/applicant attributes
(loan type, purpose, region, occupancy type, etc.) used during training.

**Outputs:**
- Default Probability (%)
- Risk Category (Low / Medium / High)
- Suggested Decision (Approve / Reject)

## Limitations

- The model misses roughly 35% of actual defaults (recall = 0.65). It should support,
  not replace, human underwriting decisions.
- Trained on historical data through 2019; predictions may not reflect newer market
  conditions without retraining.
