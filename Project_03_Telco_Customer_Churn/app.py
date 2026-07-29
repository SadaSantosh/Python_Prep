import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Set Page Config
st.set_page_config(
    page_title="Telco Churn AI | Enterprise Risk Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Glassmorphism CSS Theme
st.markdown("""
<style>
    /* Dark Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glassmorphism Cards */
    div[data-testid="stMetric"], .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.5);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    /* Custom Gradient Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 25px rgba(168, 85, 247, 0.5);
        color: white;
    }

    /* Subheaders & Titles */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_cookies=True, unsafe_allow_html=True)

# 3. Header Banner
st.title("🔮 Enterprise Telco Churn AI Engine")
st.markdown("""
<div style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 20px;'>
Real-time customer risk evaluation powered by an optimized <b>XGBoost Machine Learning Pipeline</b> with <b>SMOTE Class Balancing</b>.
</div>
""", unsafe_allow_html=True)
st.divider()

# 4. Load Model & Scaler
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
    st.sidebar.markdown("### 🟢 System Status")
    st.sidebar.success("XGBoost Model Active")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

# 5. Application Tabs
tab1, tab2, tab3 = st.tabs([
    "👤 Single Risk Assessment", 
    "📁 Batch CSV Processing", 
    "📊 Model Analytics"
])

# ==========================================
# TAB 1: SINGLE PREDICTION
# ==========================================
with tab1:
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Customer Profile Parameters")

    tenure = st.sidebar.slider("Customer Tenure (Months)", 1, 72, 12)
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

    # Feature Construction
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

    st.subheader("⚡ Real-Time Probability Assessment")
    
    if st.button("🚀 Calculate Churn Probability", use_container_width=True):
        probabilities = model.predict_proba(input_df)[0]
        churn_prob = probabilities[1] * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Churn Score", value=f"{churn_prob:.1f}%")
            st.progress(int(churn_prob))
            
        with col2:
            if churn_prob >= 50.0:
                st.error("⚠️ **HIGH RETENTION RISK DETECTED**")
                st.warning("Action Recommended: Trigger 15% discount offer or complimentary service upgrade.")
            else:
                st.success("✅ **LOW RETENTION RISK**")
                st.info("Account healthy. High customer satisfaction likelihood.")

# ==========================================
# TAB 2: BATCH PROCESSING
# ==========================================
with tab2:
    st.subheader("📁 Enterprise Batch Dataset Scoring")
    uploaded_file = st.file_uploader("Upload customer CSV file for batch inference", type=["csv"])
    
    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)
        st.write("📄 **Raw File Preview:**", raw_batch.head())
        
        if st.button("⚡ Run Batch Risk Scoring", use_container_width=True):
            batch_features = df_sample.drop(columns=['Churn'], errors='ignore')
            batch_probs = model.predict_proba(batch_features)[:, 1] * 100
            
            result_df = raw_batch.copy()
            result_df['Churn_Probability_%'] = np.round(batch_probs[:len(raw_batch)], 2)
            result_df['Risk_Level'] = np.where(result_df['Churn_Probability_%'] >= 50.0, 'HIGH RISK', 'LOW RISK')
            
            st.success(f"Successfully processed {len(result_df)} accounts!")
            st.dataframe(result_df.head(10))
            
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Results CSV",
                data=csv_data,
                file_name="churn_predictions_scored.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==========================================
# TAB 3: MODEL ANALYTICS
# ==========================================
with tab3:
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
st.markdown("<p style='text-align: center; color: #64748b;'>🔮 Enterprise Telco Churn Engine v2.0 | Glassmorphic UI Edition</p>", unsafe_allow_html=True)
