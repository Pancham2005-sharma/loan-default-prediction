import streamlit as st
import joblib
import os
import sys

# Allow importing from utils/
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from preprocessing import build_input_dataframe

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💰", layout="centered")

# ---------------------------
# Load model artifacts
# ---------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model_columns = joblib.load(os.path.join(MODEL_DIR, "loan_default_columns.pkl"))
    categorical_options = joblib.load(os.path.join(MODEL_DIR, "loan_default_categorical_options.pkl"))
    return model, scaler, model_columns, categorical_options

model, scaler, model_columns, categorical_options = load_artifacts()

# Numeric fields the model was trained on
NUMERIC_FIELDS = ["year", "loan_amount", "term", "property_value",
                   "income", "Credit_Score", "LTV", "dtir1"]

# ---------------------------
# UI
# ---------------------------
st.title("💰 Loan Default Risk Predictor")
st.markdown(
    "Enter applicant and loan details below to estimate the probability of default. "
    "This tool supports underwriting decisions and does not replace human judgment."
)

st.subheader("Applicant & Loan Details")

numeric_inputs = {}
col1, col2 = st.columns(2)
with col1:
    numeric_inputs["year"] = st.number_input("Application Year", min_value=2000, max_value=2030, value=2019)
    numeric_inputs["loan_amount"] = st.number_input("Loan Amount", min_value=0, value=300000, step=1000)
    numeric_inputs["term"] = st.number_input("Loan Term (months)", min_value=0, value=360, step=12)
    numeric_inputs["property_value"] = st.number_input("Property Value", min_value=0, value=350000, step=1000)
with col2:
    numeric_inputs["income"] = st.number_input("Applicant Income (monthly)", min_value=0, value=6000, step=100)
    numeric_inputs["Credit_Score"] = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    numeric_inputs["LTV"] = st.number_input("Loan-to-Value Ratio (%)", min_value=0.0, max_value=200.0, value=85.0)
    numeric_inputs["dtir1"] = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=35.0)

st.subheader("Applicant & Loan Attributes")

# Dynamically build a dropdown for every categorical column the model was trained on
categorical_inputs = {}
cat_cols = list(categorical_options.keys())
col_a, col_b = st.columns(2)
for i, col in enumerate(cat_cols):
    target_col = col_a if i % 2 == 0 else col_b
    with target_col:
        categorical_inputs[col] = st.selectbox(col, categorical_options[col])

st.markdown("---")

if st.button("Predict Default Risk", type="primary"):
    input_df = build_input_dataframe(numeric_inputs, categorical_inputs,
                                      categorical_options, model_columns)

    # Final model (XGBoost) was trained on unscaled features
    probability = model.predict_proba(input_df)[0][1]
    probability_pct = probability * 100

    # Risk category thresholds
    if probability < 0.30:
        risk_label, risk_color = "Low Risk", "green"
    elif probability < 0.60:
        risk_label, risk_color = "Medium Risk", "orange"
    else:
        risk_label, risk_color = "High Risk", "red"

    decision = "Reject" if probability >= 0.5 else "Approve"

    st.subheader("Prediction Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Default Probability", f"{probability_pct:.1f}%")
    m2.markdown(f"**Risk Category:** :{risk_color}[{risk_label}]")
    m3.metric("Suggested Decision", decision)

    st.progress(float(min(probability, 1.0)))

    st.caption(
        "This model achieved 89.8% accuracy and 0.895 ROC-AUC on held-out test data. "
        "It correctly identifies about 65% of actual defaults, meaning some risky applicants "
        "may still be scored as lower risk. Use this as a decision-support tool alongside "
        "standard underwriting review, not as a sole approval mechanism."
    )
