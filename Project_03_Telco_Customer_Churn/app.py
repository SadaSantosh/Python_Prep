import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Telco Churn AI | Enterprise Risk Command Center",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ADAPTIVE CSS: Master Glass Pane Layout
st.markdown("""
<style>
    /* FIX: Transparent Header to prevent white blocks at the top */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Base Background Image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1920&auto=format&fit=crop') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }
    
    /* Remove default app background so the image shines through */
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    
    /* THE MAGIC FIX: Master Glass Pane for Main Content */
    /* This creates a solid, frosted backing behind ALL text so it is readable in Light AND Dark mode */
    .block-container {
        background-color: color-mix(in srgb, var(--background-color) 85%, transparent) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-radius: 24px !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 15%, transparent) !important;
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4) !important;
        padding: 3rem 4rem !important;
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        max-width: 90% !important; /* Keeps the glass pane centered with the background showing on edges */
    }
    
    /* ADAPTIVE FLOATING SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: color-mix(in srgb, var(--background-color) 85%, transparent) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid color-mix(in srgb, var(--text-color) 15%, transparent) !important;
    }

    /* ADAPTIVE GLASS CARDS */
    div[data-testid="stMetric"], .stAlert {
        background-color: color-mix(in srgb, var(--secondary-background-color) 60%, transparent) !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        border-color: var(--primary-color) !important;
    }

    /* MODERN NEON TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: transparent;
        padding: 10px 0px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--text-color) !important;
        background-color: color-mix(in srgb, var(--secondary-background-color) 60%, transparent) !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 15%, transparent) !important;
        font-weight: 700;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }

    /* GLOWING GRADIENT BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #d946ef 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 20px rgba(217, 70, 239, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 8px 30px rgba(217, 70, 239, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header Title
st.title("⚡ Enterprise Telco Churn AI Command Center")
st.markdown("<p style='color: var(--text-color); opacity: 0.8; font-size: 1.1rem; margin-bottom: 25px;'>Next-Gen Retention Analytics Platform powered by XGBoost & SMOTE Pipeline</p>", unsafe_allow_html=True)
st.divider()

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
    "📁 Batch Financial Risk", 
    "📊 Model Analytics"
])

# ==========================================
# TAB 1: SINGLE PREDICTION + AI REPORT
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

        if f"Contract_{c_type}" in data: data[f"Contract_{c_type}"] = 1
        if f"InternetService_{internet_service}" in data: data[f"InternetService_{internet_service}"] = 1
        if f"PaymentMethod_{payment_method}" in data: data[f"PaymentMethod_{payment_method}"] = 1
        if f"TechSupport_{t_support}" in data: data[f"TechSupport_{t_support}"] = 1
        if f"OnlineSecurity_{online_security}" in data: data[f"OnlineSecurity_{online_security}"] = 1
        return pd.DataFrame([data])

    st.subheader("⚡ Real-Time Probability Assessment")
    
    if st.button("🚀 Calculate Churn Risk", use_container_width=True):
        input_df = build_input_df(monthly_charges, tech_support, contract)
        churn_prob = model.predict_proba(input_df)[0][1] * 100
        ltv_at_risk = monthly_charges * 12 

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Predicted Churn Score", value=f"{churn_prob:.1f}%")
            st.metric(label="12-Month Revenue at Risk", value=f"${ltv_at_risk:,.2f}")
            st.progress(int(churn_prob))
            
        with col2:
            st.write("### 🤖 AI Risk Analysis Report")
            reasons = []
            if contract == "Month-to-month": reasons.append("- **High-Risk Contract:** Month-to-month billing creates low switching friction.")
            if internet_service == "Fiber optic": reasons.append("- **Service Type:** Fiber optic customers historically show higher churn volatility.")
            if tech_support == "No": reasons.append("- **Lack of Support:** Customer has no tech support, reducing overall satisfaction.")
            if tenure < 12: reasons.append("- **New Customer:** Account is less than a year old, a critical drop-off period.")
            if payment_method == "Electronic check": reasons.append("- **Payment Method:** Electronic check users show higher default rates.")
            
            if churn_prob >= 50.0:
                st.error("⚠️ **HIGH RETENTION RISK**")
                if reasons:
                    st.write("**Primary Risk Drivers Identified:**")
                    for r in reasons: st.write(r)
            else:
                st.success("✅ **LOW RETENTION RISK**")
                st.write("Customer exhibits strong loyalty indicators. Existing plan is optimal.")
                
            report_text = f"AI RISK ANALYSIS REPORT\n----------------------\nChurn Probability: {churn_prob:.1f}%\n12-Month LTV at Risk: ${ltv_at_risk:,.2f}\n\nPrimary Risk Drivers:\n" + "\n".join(reasons)
            st.download_button("📥 Export AI Report (.txt)", data=report_text, file_name="AI_Churn_Report.txt")

# ==========================================
# TAB 2: RETENTION SIMULATOR
# ==========================================
with tab2:
    st.subheader("🎯 Interactive What-If Retention Strategy Simulator")
    st.write("Simulate offer strategies to see how much they lower the customer's churn risk score!")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        discount = st.slider("Offer Monthly Discount ($)", 0, int(monthly_charges), 10)
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
        st.metric(label="Simulated Score After Offer", value=f"{sim_score:.1f}%", delta=f"-{risk_reduction:.1f}% Risk")
        
        if sim_score < 50.0: st.success("🎉 This retention strategy successfully brings the customer into the SAFE zone!")
        else: st.warning("⚡ Consider upgrading contract duration or offering a higher discount to lower risk further.")

# ==========================================
# TAB 3: BATCH FINANCIAL RISK
# ==========================================
with tab3:
    st.subheader("📁 Batch Scoring & Financial Risk Impact")
    uploaded_file = st.file_uploader("Upload customer CSV file to calculate total revenue at risk", type=["csv"])
    
    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)
        
        if st.button("⚡ Run Financial Batch Scoring", use_container_width=True):
            batch_features = df_sample.drop(columns=['Churn'], errors='ignore')
            batch_probs = model.predict_proba(batch_features)[:, 1] * 100
            
            result_df = raw_batch.copy()
            result_df['Churn_Probability_%'] = np.round(batch_probs[:len(raw_batch)], 2)
            result_df['Risk_Level'] = np.where(result_df['Churn_Probability_%'] >= 50.0, 'HIGH RISK', 'LOW RISK')
            
            if 'MonthlyCharges' in result_df.columns:
                high_risk_users = result_df[result_df['Risk_Level'] == 'HIGH RISK']
                annual_revenue_at_risk = high_risk_users['MonthlyCharges'].sum() * 12
                
                col_fin1, col_fin2, col_fin3 = st.columns(3)
                col_fin1.metric("Total Customers Scored", len(result_df))
                col_fin2.metric("High Risk Accounts", len(high_risk_users))
                col_fin3.metric("Annual Revenue at Risk", f"${annual_revenue_at_risk:,.2f}", delta="-High Financial Threat", delta_color="inverse")
            
            st.dataframe(result_df.head(10))
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Scored Financial Results", data=csv_data, file_name="churn_financial_batch_scored.csv", mime="text/csv", use_container_width=True)

# ==========================================
# TAB 4: MODEL ANALYTICS
# ==========================================
with tab4:
    st.subheader("📊 Model Diagnostics & Feature Drivers")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🔑 Top Feature Importances")
        if hasattr(model, 'feature_importances_'):
            imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
            st.bar_chart(imp_df.set_index('Feature'))
    with col2:
        st.write("### 📈 Customer Scatter Distribution")
        st.scatter_chart(df_sample, x='tenure', y='MonthlyCharges', color='Churn')

# ==========================================
# FOOTER / CREATOR BRANDING
# ==========================================
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: var(--text-color); opacity: 0.6; font-size: 0.9rem;'>🔮 Enterprise Telco Churn Engine v6.1 | Master Glass Edition</p>
    <p style='font-size: 1.1rem; font-weight: 700; color: #a855f7; margin-top: -10px;'>
        🚀 Developed by Sada Santosh Kalmath
    </p>
</div>
""", unsafe_allow_html=True)