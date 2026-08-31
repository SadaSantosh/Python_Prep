import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MINIMAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1920&q=80') center/cover no-repeat;
        opacity: 0.06;
        pointer-events: none;
        z-index: 0;
    }
    .stApp > * { position: relative; z-index: 1; }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #111827 !important;
    }

    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    p, label, .stMarkdown { color: #374151 !important; }

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }

    .stButton > button {
        background: #111827 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 10px 24px !important;
    }
    .stButton > button:hover {
        background: #1f2937 !important;
    }

    [data-baseweb="tab-list"] {
        background: #f3f4f6 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid #e5e7eb !important;
    }
    [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    [aria-selected="true"] {
        background: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }

    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid #e5e7eb !important;
    }

    .block-container { padding-top: 2rem; max-width: 1100px; }
</style>
"""
st.markdown(MINIMAL_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "best_churn_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    df_sample = pd.read_csv(os.path.join(BASE_DIR, "telco_churn_cleaned.csv"))
    feature_cols = [c for c in df_sample.columns if c != "Churn"]
    return model, scaler, feature_cols, df_sample


def preprocess_raw_batch(raw_df: pd.DataFrame, feature_cols: list, scaler) -> pd.DataFrame:
    df = raw_df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    tenure_safe = df["tenure"].replace(0, 1)
    df["Avg_Monthly_Cost"] = df["TotalCharges"] / tenure_safe
    df["Monthly_Price_Diff"] = df["MonthlyCharges"] - df["Avg_Monthly_Cost"]

    for col in ["Churn", "PaperlessBilling", "Partner", "Dependents"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    churn_col = df.pop("Churn") if "Churn" in df.columns else None
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_cols]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Avg_Monthly_Cost", "Monthly_Price_Diff"]
    df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])
    return df_encoded, churn_col


try:
    model, scaler, feature_cols, df_sample = load_artifacts()
    st.sidebar.markdown("**Status**")
    st.sidebar.success("Model loaded")
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.info("Run `data_preprocessing.py` and `model_training.py` in this folder first.")
    st.stop()

st.title("Telco Churn Predictor")
st.caption("Retention risk scoring with a SMOTE-balanced classifier trained on Telco churn data.")

# Default sidebar values for cross-tab use
default_tenure = 12
default_monthly = 70.0
default_total = 840.0
default_contract = "Month-to-month"
default_internet = "Fiber optic"
default_payment = "Electronic check"
default_support = "No"
default_security = "No"
default_paperless = "Yes"
default_partner = "No"
default_dependents = "No"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Single Assessment",
    "Retention Simulator",
    "Batch Scoring",
    "Model Analytics",
    "Customer Segments",
    "What-If Explorer",
])

with tab1:
    st.sidebar.header("Customer Profile")

    tenure = st.sidebar.slider("Tenure (months)", 1, 72, default_tenure)
    monthly_charges = st.sidebar.number_input("Monthly charges ($)", 18.0, 120.0, default_monthly)
    total_charges = st.sidebar.number_input(
        "Total charges ($)", 18.0, 9000.0, float(tenure * monthly_charges)
    )
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tech_support = st.sidebar.selectbox("Tech support", ["Yes", "No", "No internet service"])
    online_security = st.sidebar.selectbox("Online security", ["Yes", "No", "No internet service"])
    paperless = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

    def build_input_df(m_charges, t_support, c_type):
        avg_cost = total_charges / max(tenure, 1)
        price_diff = m_charges - avg_cost
        num_df = pd.DataFrame(
            [[tenure, m_charges, total_charges, avg_cost, price_diff]],
            columns=["tenure", "MonthlyCharges", "TotalCharges", "Avg_Monthly_Cost", "Monthly_Price_Diff"],
        )
        scaled_num = scaler.transform(num_df)[0]

        data = {col: 0 for col in feature_cols}
        data["tenure"] = scaled_num[0]
        data["MonthlyCharges"] = scaled_num[1]
        data["TotalCharges"] = scaled_num[2]
        data["Avg_Monthly_Cost"] = scaled_num[3]
        data["Monthly_Price_Diff"] = scaled_num[4]
        data["PaperlessBilling"] = 1 if paperless == "Yes" else 0
        data["Partner"] = 1 if partner == "Yes" else 0
        data["Dependents"] = 1 if dependents == "Yes" else 0

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

    st.subheader("Risk Assessment")

    if st.button("Calculate churn risk", use_container_width=True):
        input_df = build_input_df(monthly_charges, tech_support, contract)
        churn_prob = model.predict_proba(input_df)[0][1] * 100
        ltv_at_risk = monthly_charges * 12

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Churn probability", f"{churn_prob:.1f}%")
            st.metric("12-month revenue at risk", f"${ltv_at_risk:,.2f}")
            st.progress(int(churn_prob))

        with col2:
            reasons = []
            if contract == "Month-to-month":
                reasons.append("Month-to-month contract increases switching likelihood.")
            if internet_service == "Fiber optic":
                reasons.append("Fiber customers show higher churn volatility.")
            if tech_support == "No":
                reasons.append("No tech support reduces satisfaction.")
            if tenure < 12:
                reasons.append("Customer tenure is under 12 months.")
            if payment_method == "Electronic check":
                reasons.append("Electronic check payment is a risk signal.")

            if churn_prob >= 50.0:
                st.error("High retention risk")
                for reason in reasons:
                    st.write(f"- {reason}")
            else:
                st.success("Low retention risk")
                st.write("No major risk drivers detected for this profile.")

            report_text = (
                f"Churn report\nChurn probability: {churn_prob:.1f}%\n"
                f"12-month LTV at risk: ${ltv_at_risk:,.2f}\n\nDrivers:\n"
                + "\n".join(reasons)
            )
            st.download_button("Download report", data=report_text, file_name="churn_report.txt")

with tab2:
    st.subheader("Retention Simulator")
    st.write("Adjust offers to estimate churn risk reduction.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        discount = st.slider("Monthly discount ($)", 0, int(monthly_charges), 10)
        upgrade_support = st.checkbox("Add free tech support", value=True)
        switch_contract = st.selectbox("Contract upgrade", [contract, "One year", "Two year"])

    new_monthly = max(18.0, monthly_charges - discount)
    new_support = "Yes" if upgrade_support else tech_support

    base_df = build_input_df(monthly_charges, tech_support, contract)
    sim_df = build_input_df(new_monthly, new_support, switch_contract)

    base_score = model.predict_proba(base_df)[0][1] * 100
    sim_score = model.predict_proba(sim_df)[0][1] * 100
    risk_reduction = base_score - sim_score

    with col_sim2:
        st.metric("Current churn score", f"{base_score:.1f}%")
        st.metric("Simulated score", f"{sim_score:.1f}%", delta=f"-{risk_reduction:.1f}%")
        if sim_score < 50.0:
            st.success("Offer brings customer into the safe zone.")
        else:
            st.warning("Consider a stronger retention offer.")

with tab3:
    st.subheader("Batch Scoring")
    st.write("Upload a Telco CSV (raw or preprocessed). Required raw columns include tenure and MonthlyCharges.")
    uploaded_file = st.file_uploader("Customer CSV", type=["csv"])

    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)

        if st.button("Run batch scoring", use_container_width=True):
            if set(feature_cols).issubset(raw_batch.columns):
                batch_features = raw_batch[feature_cols]
            elif {"tenure", "MonthlyCharges", "TotalCharges"}.issubset(raw_batch.columns):
                batch_features, _ = preprocess_raw_batch(raw_batch, feature_cols, scaler)
            else:
                st.error("CSV format not recognized. Upload raw Telco data or preprocessed feature columns.")
                st.stop()

            batch_probs = model.predict_proba(batch_features)[:, 1] * 100
            result_df = raw_batch.copy()
            result_df["Churn_Probability_%"] = np.round(batch_probs, 2)
            result_df["Risk_Level"] = np.where(result_df["Churn_Probability_%"] >= 50.0, "HIGH", "LOW")

            if "MonthlyCharges" in result_df.columns:
                high_risk = result_df[result_df["Risk_Level"] == "HIGH"]
                annual_risk = high_risk["MonthlyCharges"].sum() * 12
                c1, c2, c3 = st.columns(3)
                c1.metric("Customers scored", len(result_df))
                c2.metric("High risk accounts", len(high_risk))
                c3.metric("Annual revenue at risk", f"${annual_risk:,.2f}")

            st.dataframe(result_df.head(20), use_container_width=True)
            st.download_button(
                "Download results",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_batch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab4:
    st.subheader("Model Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Top feature importances")
        if hasattr(model, "feature_importances_"):
            imp_df = (
                pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
                .sort_values("Importance", ascending=False)
                .head(10)
            )
            st.bar_chart(imp_df.set_index("Feature"))
    with col2:
        st.write("Tenure vs monthly charges")
        plot_df = df_sample.copy()
        plot_df["Churn"] = plot_df["Churn"].map({0: "No", 1: "Yes"})
        st.scatter_chart(plot_df, x="tenure", y="MonthlyCharges", color="Churn")

with tab5:
    st.subheader("Customer Segmentation Analysis")
    st.write("Explore churn distribution across key customer segments.")

    seg_df = df_sample.copy()
    seg_df["Churn_Label"] = seg_df["Churn"].map({0: "No", 1: "Yes"})

    seg1, seg2 = st.columns(2)
    with seg1:
        st.write("**Churn by Contract Type**")
        contract_cols = [c for c in seg_df.columns if c.startswith("Contract_")]
        if contract_cols:
            contract_data = []
            for c in contract_cols:
                label = c.replace("Contract_", "")
                subset = seg_df[seg_df[c] == 1]
                if len(subset) > 0:
                    churn_rate = subset["Churn"].mean() * 100
                    contract_data.append({"Contract": label, "Count": len(subset), "Churn Rate %": churn_rate})
            if contract_data:
                st.dataframe(pd.DataFrame(contract_data), use_container_width=True)

    with seg2:
        st.write("**Churn by Internet Service**")
        inet_cols = [c for c in seg_df.columns if c.startswith("InternetService_")]
        if inet_cols:
            inet_data = []
            for c in inet_cols:
                label = c.replace("InternetService_", "")
                subset = seg_df[seg_df[c] == 1]
                if len(subset) > 0:
                    churn_rate = subset["Churn"].mean() * 100
                    inet_data.append({"Service": label, "Count": len(subset), "Churn Rate %": churn_rate})
            if inet_data:
                st.dataframe(pd.DataFrame(inet_data), use_container_width=True)

    st.write("**Monthly Charges Distribution by Churn Status**")
    churn_groups = seg_df.groupby("Churn_Label")["MonthlyCharges"].agg(["mean", "median", "count"])
    st.dataframe(churn_groups.round(2), use_container_width=True)

    avg_churner = seg_df[seg_df["Churn"] == 1]["MonthlyCharges"].mean()
    avg_stayer = seg_df[seg_df["Churn"] == 0]["MonthlyCharges"].mean()
    diff = avg_churner - avg_stayer
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Monthly (Churners)", f"${avg_churner:,.2f}")
    c2.metric("Avg Monthly (Stayers)", f"${avg_stayer:,.2f}")
    c3.metric("Difference", f"${diff:,.2f}", delta=f"{'Higher' if diff > 0 else 'Lower'} for churners")

with tab6:
    st.subheader("What-If Explorer")
    st.write("Manually adjust features to see how the churn probability changes.")

    wif1, wif2 = st.columns(2)
    with wif1:
        w_tenure = st.slider("What-If: Tenure (months)", 1, 72, 12, key="wif_tenure")
        w_monthly = st.slider("What-If: Monthly charges ($)", 18.0, 120.0, 70.0, key="wif_monthly")
        w_contract_wif = st.selectbox("What-If: Contract", ["Month-to-month", "One year", "Two year"], key="wif_contract")
    with wif2:
        w_internet_wif = st.selectbox("What-If: Internet", ["DSL", "Fiber optic", "No"], key="wif_internet")
        w_support_wif = st.selectbox("What-If: Tech Support", ["Yes", "No", "No internet service"], key="wif_support")
        w_payment_wif = st.selectbox("What-If: Payment", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="wif_payment")

    if st.button("Run What-If analysis", use_container_width=True):
        w_total = w_tenure * w_monthly
        wif_df = build_input_df(w_monthly, w_support_wif, w_contract_wif)
        wif_prob = model.predict_proba(wif_df)[0][1] * 100

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("What-If Churn Probability", f"{wif_prob:.1f}%")
            st.progress(int(wif_prob))
            if wif_prob >= 70:
                st.error("Critical risk - immediate action recommended")
            elif wif_prob >= 50:
                st.warning("Moderate risk - consider retention offer")
            elif wif_prob >= 30:
                st.info("Low risk - monitor quarterly")
            else:
                st.success("Very low risk - healthy customer")

        with col_b:
            st.write("**Risk Factor Summary**")
            factors = []
            if w_contract_wif == "Month-to-month":
                factors.append("Month-to-month contract")
            if w_internet_wif == "Fiber optic":
                factors.append("Fiber optic service")
            if w_support_wif == "No":
                factors.append("No tech support")
            if w_tenure < 12:
                factors.append("Short tenure (<12 months)")
            if w_payment_wif == "Electronic check":
                factors.append("Electronic check payment")
            if w_monthly > 80:
                factors.append(f"High monthly charges (${w_monthly:.0f})")
            if factors:
                for f in factors:
                    st.write(f"- {f}")
            else:
                st.write("- No major risk factors detected")

st.divider()
st.caption("Telco Churn Engine | Sada Santosh Kalmath")
