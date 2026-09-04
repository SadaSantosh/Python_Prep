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

NEUMORPHIC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --neu-bg: #e0e0e0;
        --neu-soft: #eaeaea;
        --neu-panel: #e6e6e6;
        --neu-light: #ffffff;
        --neu-shadow: #c0c0c0;
        --neu-text: #2b2b2b;
        --neu-muted: #565656;
        --neu-radius: 15px;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: var(--neu-bg) !important;
        color: var(--neu-text);
        font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }

    h1, h2, h3, h4 {
        color: var(--neu-text) !important;
        font-weight: 650 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: var(--neu-muted) !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] p { color: #6a6a6a !important; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--neu-bg) !important;
        border-right: none !important;
    }
    [data-testid="stSidebarContent"] { padding: 0.6rem 0.5rem 2rem 0.5rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--neu-text) !important; }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, var(--neu-soft), #d6d6d6) !important;
        border: none !important;
        border-radius: var(--neu-radius) !important;
        padding: 1.1rem 1.3rem !important;
        box-shadow: 8px 8px 16px var(--neu-shadow), -8px -8px 16px var(--neu-light) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--neu-muted) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
        color: var(--neu-text) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] p { font-weight: 600 !important; }

    /* ---------- Buttons (raised / extruded) ---------- */
    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(145deg, var(--neu-soft), #d4d4d4) !important;
        color: var(--neu-text) !important;
        border: none !important;
        border-radius: var(--neu-radius) !important;
        font-weight: 600 !important;
        box-shadow: 7px 7px 14px var(--neu-shadow), -7px -7px 14px var(--neu-light) !important;
        transition: box-shadow 0.18s ease !important;
    }
    [data-testid="stButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 5px 5px 10px var(--neu-shadow), -5px -5px 10px var(--neu-light) !important;
    }
    [data-testid="stButton"] > button:active,
    [data-testid="stDownloadButton"] > button:active,
    [data-testid="stFormSubmitButton"] > button:active {
        box-shadow: inset 5px 5px 10px var(--neu-shadow), inset -5px -5px 10px var(--neu-light) !important;
    }

    /* ---------- Inputs (inset / pressed) ---------- */
    [data-testid="stNumberInputContainer"],
    [data-testid="stTextAreaRootElement"],
    [data-testid="stTextInputRootElement"],
    [data-testid="stDateInput"] [role="group"] {
        background-color: var(--neu-panel) !important;
        border: none !important;
        border-radius: var(--neu-radius) !important;
        box-shadow: inset 5px 5px 10px var(--neu-shadow), inset -5px -5px 10px var(--neu-light) !important;
        overflow: hidden !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--neu-text) !important;
    }
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] { background: transparent !important; color: var(--neu-muted) !important; }

    /* Select (react-aria combobox) */
    [data-testid="stSelectbox"] [role="group"] {
        background-color: var(--neu-panel) !important;
        border: none !important;
        border-radius: var(--neu-radius) !important;
        box-shadow: inset 5px 5px 10px var(--neu-shadow), inset -5px -5px 10px var(--neu-light) !important;
    }
    [data-testid="stSelectbox"] input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--neu-text) !important;
    }
    [data-testid="stSelectbox"] button {
        background: transparent !important;
        border: none !important;
        color: var(--neu-muted) !important;
    }

    [data-testid="stCheckbox"] label { color: var(--neu-text) !important; }

    /* ---------- Tabs ---------- */
    [data-testid="stTabs"] [role="tablist"] {
        background: #d3d3d3 !important;
        border-radius: var(--neu-radius) !important;
        padding: 0.35rem !important;
        box-shadow: inset 3px 3px 7px #bebebe, inset -3px -3px 7px #f2f2f2 !important;
        gap: 0.35rem !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"] {
        border-radius: var(--neu-radius) !important;
        color: var(--neu-muted) !important;
        font-weight: 500 !important;
        transition: all 0.18s ease !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
        background: linear-gradient(145deg, #ededed, #d5d5d5) !important;
        color: var(--neu-text) !important;
        font-weight: 600 !important;
        box-shadow: 5px 5px 10px var(--neu-shadow), -5px -5px 10px var(--neu-light) !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"] .react-aria-SelectionIndicator { display: none !important; }

    /* ---------- Surfaces: uploader / dataframe / charts / alerts / expanders ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--neu-panel) !important;
        border: 1px dashed #aaaaaa !important;
        border-radius: var(--neu-radius) !important;
        box-shadow: inset 4px 4px 8px var(--neu-shadow), inset -4px -4px 8px var(--neu-light) !important;
        color: var(--neu-muted) !important;
    }
    [data-testid="stDataFrame"] {
        background: var(--neu-panel) !important;
        border: none !important;
        border-radius: var(--neu-radius) !important;
        overflow: hidden !important;
        box-shadow: 8px 8px 16px var(--neu-shadow), -8px -8px 16px var(--neu-light) !important;
    }
    [data-testid="stPlotlyChart"] {
        background: var(--neu-panel) !important;
        border-radius: var(--neu-radius) !important;
        padding: 0.4rem 0.6rem !important;
        box-shadow: 8px 8px 16px var(--neu-shadow), -8px -8px 16px var(--neu-light) !important;
    }
    [data-testid="stAlert"] {
        border-radius: var(--neu-radius) !important;
        box-shadow: 6px 6px 12px rgba(150, 150, 150, 0.35), -6px -6px 12px var(--neu-light) !important;
    }
    [data-testid="stExpander"] {
        background: var(--neu-panel) !important;
        border: none !important;
        border-radius: var(--neu-radius) !important;
        overflow: hidden !important;
        box-shadow: 6px 6px 12px var(--neu-shadow), -6px -6px 12px var(--neu-light) !important;
    }
    [data-testid="stExpander"] summary { color: var(--neu-text) !important; font-weight: 600 !important; }

    hr { border-color: rgba(90, 90, 90, 0.25) !important; }
</style>
"""
st.markdown(NEUMORPHIC_CSS, unsafe_allow_html=True)


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

    def build_input_df(m_charges, t_support, c_type, *, t_v=None, total_v=None, inet_v=None, pay_v=None):
        """Build one model-ready row. Optional overrides let the What-If Explorer
        test profiles that differ from the sidebar defaults."""
        tenure_v = tenure if t_v is None else t_v
        charges_total = total_charges if total_v is None else total_v
        internet = internet_service if inet_v is None else inet_v
        payment = payment_method if pay_v is None else pay_v

        avg_cost = charges_total / max(tenure_v, 1)
        price_diff = m_charges - avg_cost
        num_df = pd.DataFrame(
            [[tenure_v, m_charges, charges_total, avg_cost, price_diff]],
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
        if f"InternetService_{internet}" in data:
            data[f"InternetService_{internet}"] = 1
        if f"PaymentMethod_{payment}" in data:
            data[f"PaymentMethod_{payment}"] = 1
        if f"TechSupport_{t_support}" in data:
            data[f"TechSupport_{t_support}"] = 1
        if f"OnlineSecurity_{online_security}" in data:
            data[f"OnlineSecurity_{online_security}"] = 1
        return pd.DataFrame([data])

    st.subheader("Risk Assessment")

    if st.button("Calculate churn risk", width="stretch"):
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
        upgrade_options = [contract] + [c for c in ["One year", "Two year", "Month-to-month"] if c != contract]
        switch_contract = st.selectbox("Contract upgrade", upgrade_options)

    new_monthly = max(18.0, monthly_charges - discount)
    new_support = "Yes" if upgrade_support else tech_support

    base_df = build_input_df(monthly_charges, tech_support, contract)
    sim_df = build_input_df(new_monthly, new_support, switch_contract)

    base_score = model.predict_proba(base_df)[0][1] * 100
    sim_score = model.predict_proba(sim_df)[0][1] * 100

    with col_sim2:
        st.metric("Current churn score", f"{base_score:.1f}%")
        st.metric(
            "Simulated score",
            f"{sim_score:.1f}%",
            delta=f"{sim_score - base_score:+.1f}%",
            delta_color="inverse",
        )
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

        if st.button("Run batch scoring", width="stretch"):
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

            st.dataframe(result_df.head(20), width="stretch")
            st.download_button(
                "Download results",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_batch_results.csv",
                mime="text/csv",
                width="stretch",
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
                st.dataframe(pd.DataFrame(contract_data), width="stretch")

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
                st.dataframe(pd.DataFrame(inet_data), width="stretch")

    st.write("**Monthly Charges Distribution by Churn Status**")
    churn_groups = seg_df.groupby("Churn_Label")["MonthlyCharges"].agg(["mean", "median", "count"])
    st.dataframe(churn_groups.round(2), width="stretch")

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
        w_contract_wif = st.selectbox(
            "What-If: Contract", ["Month-to-month", "One year", "Two year"], key="wif_contract"
        )
    with wif2:
        w_internet_wif = st.selectbox("What-If: Internet", ["DSL", "Fiber optic", "No"], key="wif_internet")
        w_support_wif = st.selectbox("What-If: Tech Support", ["Yes", "No", "No internet service"], key="wif_support")
        w_payment_wif = st.selectbox(
            "What-If: Payment",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="wif_payment",
        )

    if st.button("Run What-If analysis", width="stretch"):
        wif_df = build_input_df(
            w_monthly,
            w_support_wif,
            w_contract_wif,
            t_v=w_tenure,
            total_v=w_tenure * w_monthly,
            inet_v=w_internet_wif,
            pay_v=w_payment_wif,
        )
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
