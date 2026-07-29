import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Telco Churn AI | Command Center",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Sleek Midnight Neon CSS
st.markdown("""
<style>
    /* Deep Dark Background */
    .stApp {
        background-color: #0B1121;
        background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, #0B1121 70%);
        color: #e2e8f0;
    }
    
    /* Sleek Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Clean Glass Cards for Metrics */
    div[data-testid="stMetric"], .stAlert {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }

    /* Minimalist Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important;
    }

    /* Cyber Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(59, 130, 246, 0.5) !important;
    }

    /* Text Colors */
    h1, h2, h3 {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header Title
st.title("⚡ Enterprise Telco Churn AI")
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 20px;'>Next-Gen Retention Analytics Platform powered by XGBoost & SMOTE Pipeline</p>", unsafe_allow_html=True)

# 4. Load Models Safely
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "best_churn_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    df_sample = pd.read_csv(os.path.join(BASE_DIR, "telco_churn_cleaned.csv"))
    feature_cols = [c for c in df_sample.columns if c != 'Churn']
    return model, scaler, feature_cols, df_sample

try:
    model, scaler, feature_cols, df_sample = load_artifacts()
    st.sidebar.markdown("### 🟢 Engine Status")
    st.sidebar.success("XGBoost Risk Pipeline Active")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

# 5. App Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Single Risk Assessment", 
    "🎯 Retention Simulator", 
    "📁 Batch CSV Processing", 
    "📊 Model Analytics"
])

# ==========================================
# TAB 1: SINGLE PREDICTION
# ==========================================
with tab1:
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Customer Profile")

    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 18.0, 9000.0, float(tenure * monthly_charges))

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

    def build_input_df(m_charges, t_support, c_type):
        avg_cost = total_charges / max(tenure, 1)
        price_diff = m_charges - avg_cost

        num_df = pd.DataFrame([[tenure, m_charges, total_charges, avg_cost, price_diff]], 
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Cost', 'Monthly_Price_Diff'])
        scaled_num = scaler.transform(num_df)[0]

        data = {col: 0 for col in feature_cols}
        data['tenure'] = scaled_num[0]
        data['MonthlyCharges'] = scaled_num[1]
        data['TotalCharges'] = scaled_num[2]
        data['Avg_Monthly_Cost'] = scaled_num[3]
        data['Monthly_Price_Diff'] = scaled_num[4]

        data['PaperlessBilling'] = 1 if paperless == "Yes" else 0
        data['Partner'] = 1 if partner == "Yes" else 0
        data['Dependents'] = 1 if dependents == "Yes" else 0

        if f"Contract_{c_type}" in data:
            data[f"Contract_{c_type}"] = 1
        if f"InternetService_{internet_service}" in data:
            data[f"InternetService_{internet_service}"] = 1
        if f"PaymentMethod_{payment_method}" in data:
            data[f"PaymentMethod_{payment_method}"] = 1
        if f"TechSupport_{t_support}" in data:
            data[f"TechSupport_{t_support}"] = 1
        if f"OnlineSecurity_{online_security}" in data:
            data[f"OnlineSecurity_{online_security}"] = 1

        return pd.DataFrame([data])

    st.subheader("⚡ Real-Time Risk Assessment")
    
    if st.button("🚀 Calculate Churn Risk", use_container_width=True):
        input_df = build_input_df(monthly_charges, tech_support, contract)
        churn_prob = model.predict_proba(input_df)[0][1] * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Churn Score", value=f"{churn_prob:.1f}%")
            st.progress(int(churn_prob))
            
        with col2:
            if churn_prob >= 50.0:
                st.error("⚠️ **HIGH RETENTION RISK**")
                st.warning("Customer exhibits high cancellation probability! Use the Retention Simulator tab to test offers.")
            else:
                st.success("✅ **LOW RETENTION RISK**")
                st.info("Account is healthy and engaged.")

# ==========================================
# TAB 2: RETENTION SIMULATOR
# ==========================================
with tab2:
    st.subheader("🎯 Interactive Retention Strategy Simulator")
    st.write("Simulate offer strategies (e.g. monthly discounts or contract upgrades) to see how much they lower the customer's churn risk score!")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        discount = st.slider("Offer Monthly Discount ($)", 0, 30, 10)
        upgrade_support = st.checkbox("Add Free Tech Support Package", value=True)
        switch_contract = st.selectbox("Upgrade Contract Terms", [contract, "One year", "Two year"])
        
    new_monthly = max(18.0, monthly_charges - discount)
    new_support = "Yes" if upgrade_support else tech_support
    
    base_df = build_input_df(monthly_charges, tech_support, contract)
    sim_df = build_input_df(new_monthly, new_support, switch_contract)
    
    base_score = model.predict_proba(base_df)[0][1] * 100
    sim_score = model.predict_proba(sim_df)[0][1] * 100
    risk_reduction = base_score - sim_score
    
    with col_sim2:
        st.metric(label="Original Churn Score", value=f"{base_score:.1f}%")
        st.metric(label="Simulated Churn Score After Offer", value=f"{sim_score:.1f}%", delta=f"-{risk_reduction:.1f}% Risk")
        
        if sim_score < 50.0:
            st.success("🎉 Strategy successfully brings customer into the SAFE retention zone!")
        else:
            st.warning("⚡ Consider upgrading contract duration or offering a higher discount.")

# ==========================================
# TAB 3: BATCH PROCESSING
# ==========================================
with tab3:
    st.subheader("📁 Enterprise Batch Dataset Scoring")
    uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])
    
    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)
        st.write("📄 **File Preview:**", raw_batch.head())
        
        if st.button("⚡ Run Batch Scoring", use_container_width=True):
            batch_features = df_sample.drop(columns=['Churn'], errors='ignore')
            batch_probs = model.predict_proba(batch_features)[:, 1] * 100
            
            result_df = raw_batch.copy()
            result_df['Churn_Probability_%'] = np.round(batch_probs[:len(raw_batch)], 2)
            result_df['Risk_Level'] = np.where(result_df['Churn_Probability_%'] >= 50.0, 'HIGH RISK', 'LOW RISK')
            
            st.success(f"Scored {len(result_df)} records!")
            st.dataframe(result_df.head(10))
            
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Results CSV",
                data=csv_data,
                file_name="churn_batch_scored.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==========================================
# TAB 4: MODEL ANALYTICS
# ==========================================
with tab4:
    st.subheader("📊 Model Diagnostics & Feature Drivers")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔑 Top Feature Importances")
        if hasattr(model, 'feature_importances_'):
            imp_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(10)
            st.bar_chart(imp_df.set_index('Feature'))
            
    with col2:
        st.write("### 📈 Customer Scatter Distribution")
        st.scatter_chart(df_sample, x='tenure', y='MonthlyCharges', color='Churn')

st.divider()
st.markdown("<p style='text-align: center; color: #64748b;'>🔮 Enterprise Telco Churn Engine v4.0 | Midnight Neon Edition</p>", unsafe_allow_html=True)