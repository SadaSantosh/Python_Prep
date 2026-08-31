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
    .stApp { background-color: #fafafa; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    h1, h2, h3 { color: #111827; font-weight: 600; letter-spacing: -0.02em; }
    p, label, .stMarkdown { color: #374151; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
    }
    .stButton > button {
        background: #111827 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover { background: #374151 !important; }
    .block-container { padding-top: 2rem; max-width: 1100px; }
</style>
"""
st.markdown(MINIMAL_CSS, unsafe_allow_html=True)

st.title("Telco Churn Predictor")
st.caption("Retention risk scoring with a SMOTE-balanced classifier trained on Telco churn data.")


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

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Assessment",
    "Retention Simulator",
    "Batch Scoring",
    "Model Analytics",
])

with tab1:
    st.sidebar.header("Customer profile")

    tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly charges ($)", 18.0, 120.0, 70.0)
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

    st.subheader("Risk assessment")

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
    st.subheader("Retention simulator")
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
    st.subheader("Batch scoring")
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
    st.subheader("Model analytics")
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

st.divider()
st.caption("Telco Churn Engine · Sada Santosh Kalmath")
