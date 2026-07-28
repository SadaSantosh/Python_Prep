import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set Page Config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# 1. Title & Header Banner
st.title("🔮 Enterprise Telco Customer Churn Prediction Engine")
st.markdown("""
This interactive application uses an **XGBoost Machine Learning Pipeline** (balanced via **SMOTE**) 
to evaluate customer churn probability in real time. Adjust the customer profile parameters below to compute risk score!
""")
st.divider()

# 2. Load Model & Scaler Artifacts
@st.cache_resource
def load_artifacts():
 with open("Project_03_Telco_Customer_Churn/best_churn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("Project_03_Telco_Customer_Churn/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    # Load sample clean dataset to extract feature column alignment
    df_sample = pd.read_csv("project_03_telco_customer_churn/telco_churn_cleaned.csv")
    feature_cols = [c for c in df_sample.columns if c != 'Churn']
    return model, scaler, feature_cols

try:
    model, scaler, feature_cols = load_artifacts()
    st.sidebar.success("✅ Machine Learning Models Successfully Loaded!")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

# 3. Sidebar Input Form
st.sidebar.header("📋 Customer Demographics & Plan Details")

tenure = st.sidebar.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges))

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

tech_support = st.sidebar.selectbox("Tech Support Included?", ["Yes", "No", "No internet service"])
online_security = st.sidebar.selectbox("Online Security Included?", ["Yes", "No", "No internet service"])
paperless = st.sidebar.selectbox("Paperless Billing?", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

# 4. Feature Construction & Normalization
# Engineered Features
avg_monthly_cost = total_charges / max(tenure, 1)
monthly_price_diff = monthly_charges - avg_monthly_cost

# Scale numeric attributes using serialized Scaler
num_df = pd.DataFrame([[tenure, monthly_charges, total_charges, avg_monthly_cost, monthly_price_diff]], 
                      columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Cost', 'Monthly_Price_Diff'])
scaled_numerics = scaler.transform(num_df)[0]

# Build input dictionary initialized to zeros matching training feature matrix
input_data = {col: 0 for col in feature_cols}

# Populate scaled numerics
input_data['tenure'] = scaled_numerics[0]
input_data['MonthlyCharges'] = scaled_numerics[1]
input_data['TotalCharges'] = scaled_numerics[2]
input_data['Avg_Monthly_Cost'] = scaled_numerics[3]
input_data['Monthly_Price_Diff'] = scaled_numerics[4]

# Populate binary maps
input_data['PaperlessBilling'] = 1 if paperless == "Yes" else 0
input_data['Partner'] = 1 if partner == "Yes" else 0
input_data['Dependents'] = 1 if dependents == "Yes" else 0

# Populate One-Hot Dummies dynamically
if f"Contract_{contract}" in input_data:
    input_data[f"Contract_{contract}"] = 1
if f"InternetService_{internet_service}" in input_data:
    input_data[f"InternetService_{internet_service}"] = 1
if f"PaymentMethod_{payment_method}" in input_data:
    input_data[f"PaymentMethod_{payment_method}"] = 1
if f"TechSupport_{tech_support}" in input_data:
    input_data[f"TechSupport_{tech_support}"] = 1
if f"OnlineSecurity_{online_security}" in input_data:
    input_data[f"OnlineSecurity_{online_security}"] = 1

input_df = pd.DataFrame([input_data])

# 5. Prediction Execution
st.subheader("📊 Real-Time Risk Assessment")

col1, col2 = st.columns(2)

if st.button("🚀 Calculate Churn Risk", use_container_width=True):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    churn_prob = probabilities[1] * 100

    with col1:
        st.metric(label="Churn Probability Score", value=f"{churn_prob:.1f}%")
        
    with col2:
        if churn_prob >= 50.0:
            st.error("⚠️ **HIGH CHURN RISK**")
            st.warning("Customer exhibits high probability of canceling service! Recommended Action: Issue retention discount offer.")
        else:
            st.success("✅ **LOW CHURN RISK**")
            st.info("Customer account is healthy with high retention likelihood.")

st.divider()
st.markdown("🔒 *Model Version: XGBoost v1.0 | Pipeline: SMOTE-Balanced | Scale: StandardScaler*")