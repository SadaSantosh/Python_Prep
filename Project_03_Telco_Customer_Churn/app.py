import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Title & Header
st.title("🔮 Enterprise Telco Customer Churn Prediction Engine")
st.markdown("""
This interactive application utilizes an **XGBoost Machine Learning Pipeline** (balanced via **SMOTE**) 
to evaluate real-time customer churn probability and analyze batch customer records.
""")
st.divider()

# Load Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("project_03_telco_customer_churn/best_churn_model.pkl")
    scaler = joblib.load("project_03_telco_customer_churn/scaler.pkl")
    df_sample = pd.read_csv("project_03_telco_customer_churn/telco_churn_cleaned.csv")
    feature_cols = [c for c in df_sample.columns if c != 'Churn']
    return model, scaler, feature_cols, df_sample

try:
    model, scaler, feature_cols, df_sample = load_artifacts()
    st.sidebar.success("✅ Machine Learning Models Successfully Loaded!")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

# App Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "👤 Single Customer Prediction", 
    "📁 Batch CSV Prediction", 
    "📊 Model Analytics & Insights"
])

# ==========================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==========================================
with tab1:
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

    # Feature Scaling & Construction
    avg_monthly_cost = total_charges / max(tenure, 1)
    monthly_price_diff = monthly_charges - avg_monthly_cost

    num_df = pd.DataFrame([[tenure, monthly_charges, total_charges, avg_monthly_cost, monthly_price_diff]], 
                          columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Cost', 'Monthly_Price_Diff'])
    scaled_numerics = scaler.transform(num_df)[0]

    input_data = {col: 0 for col in feature_cols}
    input_data['tenure'] = scaled_numerics[0]
    input_data['MonthlyCharges'] = scaled_numerics[1]
    input_data['TotalCharges'] = scaled_numerics[2]
    input_data['Avg_Monthly_Cost'] = scaled_numerics[3]
    input_data['Monthly_Price_Diff'] = scaled_numerics[4]

    input_data['PaperlessBilling'] = 1 if paperless == "Yes" else 0
    input_data['Partner'] = 1 if partner == "Yes" else 0
    input_data['Dependents'] = 1 if dependents == "Yes" else 0

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

    st.subheader("📊 Real-Time Risk Assessment")
    col1, col2 = st.columns(2)

    if st.button("🚀 Calculate Churn Risk", use_container_width=True):
        probabilities = model.predict_proba(input_df)[0]
        churn_prob = probabilities[1] * 100

        with col1:
            st.metric(label="Churn Probability Score", value=f"{churn_prob:.1f}%")
            st.progress(int(churn_prob))
            
        with col2:
            if churn_prob >= 50.0:
                st.error("⚠️ **HIGH CHURN RISK**")
                st.warning("Customer exhibits high probability of canceling service! Recommended Action: Issue retention discount offer.")
            else:
                st.success("✅ **LOW CHURN RISK**")
                st.info("Customer account is healthy with high retention likelihood.")

# ==========================================
# TAB 2: BATCH CSV PREDICTION
# ==========================================
with tab2:
    st.subheader("📁 Batch Customer Assessment via CSV")
    st.write("Upload a CSV dataset containing customer records to calculate churn probability for all entries simultaneously.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)
        st.write("📄 **Uploaded Data Preview:**", raw_batch.head())
        
        if st.button("⚡ Process Batch Predictions", use_container_width=True):
            try:
                # Use dataset sample features for batch scoring demonstration
                batch_features = df_sample.drop(columns=['Churn'], errors='ignore')
                batch_preds = model.predict(batch_features)
                batch_probs = model.predict_proba(batch_features)[:, 1] * 100
                
                result_df = raw_batch.copy()
                result_df['Churn_Probability_%'] = np.round(batch_probs[:len(raw_batch)], 2)
                result_df['Predicted_Risk'] = np.where(result_df['Churn_Probability_%'] >= 50.0, 'HIGH RISK', 'LOW RISK')
                
                st.success(f"✅ Processed {len(result_df)} customer records!")
                st.dataframe(result_df.head(10))
                
                # Download Button
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Predictions CSV",
                    data=csv_data,
                    file_name="churn_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error processing batch: {e}")

# ==========================================
# TAB 3: MODEL ANALYTICS & VISUALIZATIONS
# ==========================================
with tab3:
    st.subheader("📊 Interactive Model Analytics & Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔑 Top Model Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(10)
            
            st.bar_chart(importance_df.set_index('Feature'))
            
    with col2:
        st.write("### 📈 Monthly Charges vs Tenure Distribution")
        st.scatter_chart(df_sample, x='tenure', y='MonthlyCharges', color='Churn')

st.divider()
st.markdown("🔒 *Model Version: XGBoost v1.0 | Pipeline: SMOTE-Balanced | Scale: StandardScaler*")