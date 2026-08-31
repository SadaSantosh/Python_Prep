import json
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="ValuaAI",
    page_icon="🏠",
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
        background: url('https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=1920&q=80') center/cover no-repeat;
        opacity: 0.05;
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
    model = joblib.load(os.path.join(BASE_DIR, "real_estate_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "real_estate_scaler.pkl"))
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"mae": 21169.73, "r2": 0.9835}
    return model, scaler, metrics


model, scaler, metrics = load_artifacts()
mae = metrics["mae"]
r2_score_val = metrics["r2"]

st.title("ValuaAI")
st.caption("Real estate valuation using a Random Forest regression model.")

with st.sidebar:
    st.header("Settings")
    st.write(f"Model R²: {r2_score_val * 100:.2f}%")
    currency_symbol = st.selectbox("Currency", ["$ (USD)", "₹ (INR)", "€ (EUR)"])
    multiplier, curr_prefix = 1.0, "$"
    if "INR" in currency_symbol:
        multiplier, curr_prefix = 83.0, "₹"
    elif "EUR" in currency_symbol:
        multiplier, curr_prefix = 0.92, "€"

# Initialize default values BEFORE tabs so all tabs can reference them
default_sqft = 2200
default_bedrooms = 3
default_bathrooms = 2
default_location_score = 7
default_age = 5

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Valuation", "Batch Processing", "Market Map", "Mortgage Calculator", "Comparable Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        sqft = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=default_sqft, step=50)
        bedrooms = st.slider("Bedrooms", 1, 8, default_bedrooms)
        bathrooms = st.slider("Bathrooms", 1, 6, default_bathrooms)
    with col2:
        location_score = st.slider("Location score (1-10)", 1, 10, default_location_score)
        age = st.slider("Property age (years)", 0, 50, default_age)

    renovate = st.checkbox("Simulate renovation")
    extra_sqft = extra_baths = location_boost = 0
    if renovate:
        r1, r2, r3 = st.columns(3)
        extra_sqft = r1.slider("Added sqft", 0, 1500, 300, step=50)
        extra_baths = r2.slider("Added bathrooms", 0, 3, 1)
        location_boost = r3.slider("Location upgrade", 0, 3, 1)

    if st.button("Estimate value", use_container_width=True):
        base_df = pd.DataFrame([{
            "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "age": age, "location_score": location_score,
        }])
        base_price = model.predict(scaler.transform(base_df))[0] * multiplier
        error_margin = mae * multiplier

        st.metric("Estimated value", f"{curr_prefix}{base_price:,.0f}")
        st.write(
            f"Confidence range (+/-MAE): "
            f"{curr_prefix}{base_price - error_margin:,.0f} - {curr_prefix}{base_price + error_margin:,.0f}"
        )

        if renovate:
            sim_df = pd.DataFrame([{
                "sqft": sqft + extra_sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms + extra_baths,
                "age": max(0, age - 2),
                "location_score": min(10, location_score + location_boost),
            }])
            sim_price = model.predict(scaler.transform(sim_df))[0] * multiplier
            st.success(
                f"Post-renovation value: {curr_prefix}{sim_price:,.0f} "
                f"(+{curr_prefix}{sim_price - base_price:,.0f})"
            )

        importance_df = pd.DataFrame({
            "Feature": ["Square feet", "Bedrooms", "Bathrooms", "Age", "Location"],
            "Importance": model.feature_importances_ * 100,
        }).sort_values("Importance")
        fig = px.bar(
            importance_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Greys",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#374151", margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Batch Valuation")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    req_cols = ["sqft", "bedrooms", "bathrooms", "age", "location_score"]

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if all(c in batch_df.columns for c in req_cols):
            preds = model.predict(scaler.transform(batch_df[req_cols])) * multiplier
            batch_df["Valuation"] = preds
            batch_df["Lower_Bound"] = preds - (mae * multiplier)
            batch_df["Upper_Bound"] = preds + (mae * multiplier)

            st.subheader("Results preview")
            st.dataframe(batch_df.head(10), use_container_width=True)

            k1, k2, k3 = st.columns(3)
            k1.metric("Listings", len(batch_df))
            k2.metric("Portfolio value", f"{curr_prefix}{batch_df['Valuation'].sum():,.0f}")
            k3.metric("Average value", f"{curr_prefix}{batch_df['Valuation'].mean():,.0f}")

            st.download_button(
                "Download CSV",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="valuaai_batch_report.csv",
                use_container_width=True,
            )
        else:
            st.error(f"Missing columns. Required: {req_cols}")

with tab3:
    st.subheader("Regional Price Map (synthetic)")
    np.random.seed(42)
    lat_center, lon_center = 37.7749, -122.4194
    geo_points = 150
    geo_df = pd.DataFrame({
        "sqft": np.random.randint(1000, 4500, geo_points),
        "bedrooms": np.random.randint(2, 5, geo_points),
        "bathrooms": np.random.randint(1, 4, geo_points),
        "age": np.random.randint(1, 30, geo_points),
        "location_score": np.random.randint(3, 10, geo_points),
        "lat": lat_center + np.random.normal(0, 0.05, geo_points),
        "lon": lon_center + np.random.normal(0, 0.05, geo_points),
    })
    geo_df["Valuation"] = (
        model.predict(scaler.transform(geo_df[req_cols])) * multiplier
    )

    fig_map = px.scatter_mapbox(
        geo_df, lat="lat", lon="lon", size="Valuation", color="Valuation",
        color_continuous_scale="Greys", size_max=15, zoom=10,
        mapbox_style="carto-positron",
        hover_data={"Valuation": ":.0f", "sqft": True, "lat": False, "lon": False},
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#374151",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    st.subheader("Mortgage Calculator")
    st.write("Estimate monthly payments based on property value and financing terms.")

    mc1, mc2 = st.columns(2)
    with mc1:
        prop_value = st.number_input("Property value ($)", min_value=50000, max_value=5000000, value=500000, step=10000, key="mort_prop")
        down_payment_pct = st.slider("Down payment (%)", 0, 50, 20, key="mort_down")
        loan_term_years = st.selectbox("Loan term", [15, 20, 30], key="mort_term")
    with mc2:
        interest_rate = st.slider("Interest rate (%)", 1.0, 10.0, 6.5, 0.1, key="mort_rate")
        property_tax_rate = st.slider("Annual property tax rate (%)", 0.0, 3.0, 1.1, 0.1, key="mort_tax")
        insurance_annual = st.number_input("Annual insurance ($)", min_value=0, max_value=10000, value=1500, step=100, key="mort_ins")

    down_payment = prop_value * (down_payment_pct / 100)
    loan_amount = prop_value - down_payment
    monthly_rate = (interest_rate / 100) / 12
    num_payments = loan_term_years * 12

    if monthly_rate > 0:
        monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_mortgage = loan_amount / num_payments

    monthly_tax = (prop_value * property_tax_rate / 100) / 12
    monthly_insurance = insurance_annual / 12
    total_monthly = monthly_mortgage + monthly_tax + monthly_insurance

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Down Payment", f"{curr_prefix}{down_payment:,.0f}")
    m2.metric("Loan Amount", f"{curr_prefix}{loan_amount:,.0f}")
    m3.metric("Monthly Mortgage", f"{curr_prefix}{monthly_mortgage:,.0f}")
    m4.metric("Total Monthly Payment", f"{curr_prefix}{total_monthly:,.0f}")

    st.write("**Payment Breakdown**")
    breakdown_df = pd.DataFrame({
        "Component": ["Principal & Interest", "Property Tax", "Insurance"],
        "Monthly Amount": [monthly_mortgage, monthly_tax, monthly_insurance],
    })
    fig_pie = px.pie(breakdown_df, values="Monthly Amount", names="Component",
                     color_discrete_sequence=["#374151", "#6b7280", "#9ca3af"])
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#374151", margin=dict(l=0, r=0, t=20, b=0),
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    total_interest = (monthly_mortgage * num_payments) - loan_amount
    st.info(f"**Total interest over {loan_term_years} years:** {curr_prefix}{total_interest:,.0f}  |  **Total cost:** {curr_prefix}{(loan_amount + total_interest + down_payment + insurance_annual * loan_term_years):,.0f}")

with tab5:
    st.subheader("Comparable Property Analysis")
    st.write("See how your property compares to similar listings in the area.")

    comp_sqft = st.slider("Comparable area (sqft)", 500, 10000, default_sqft, step=50, key="comp_sqft")
    comp_bedrooms = st.slider("Comparable bedrooms", 1, 8, default_bedrooms, key="comp_bed")
    comp_bathrooms = st.slider("Comparable bathrooms", 1, 6, default_bathrooms, key="comp_bath")
    comp_location = st.slider("Comparable location (1-10)", 1, 10, default_location_score, key="comp_loc")
    num_comps = st.slider("Number of comparables", 5, 50, 15, key="num_comps")

    np.random.seed(42)
    comps = pd.DataFrame({
        "sqft": np.random.randint(max(500, comp_sqft - 500), comp_sqft + 500, num_comps),
        "bedrooms": np.random.randint(max(1, comp_bedrooms - 1), min(8, comp_bedrooms + 2), num_comps),
        "bathrooms": np.random.randint(max(1, comp_bathrooms - 1), min(6, comp_bathrooms + 2), num_comps),
        "age": np.random.randint(0, 30, num_comps),
        "location_score": np.random.randint(max(1, comp_location - 2), min(10, comp_location + 3), num_comps),
    })
    comps["Valuation"] = model.predict(scaler.transform(comps[["sqft", "bedrooms", "bathrooms", "age", "location_score"]])) * multiplier
    comps["Price_per_sqft"] = comps["Valuation"] / comps["sqft"]

    target_input = pd.DataFrame([{"sqft": comp_sqft, "bedrooms": comp_bedrooms, "bathrooms": comp_bathrooms, "age": default_age, "location_score": comp_location}])
    target_val = model.predict(scaler.transform(target_input))[0] * multiplier
    target_ppsf = target_val / comp_sqft

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Your Estimate", f"{curr_prefix}{target_val:,.0f}")
    k2.metric("Avg Comparable", f"{curr_prefix}{comps['Valuation'].mean():,.0f}")
    k3.metric("Your Price/sqft", f"{curr_prefix}{target_ppsf:,.0f}")
    k4.metric("Avg Comp Price/sqft", f"{curr_prefix}{comps['Price_per_sqft'].mean():,.0f}")

    diff_pct = ((target_val - comps['Valuation'].mean()) / comps['Valuation'].mean()) * 100
    if abs(diff_pct) < 5:
        st.success(f"Your property is priced competitively ({diff_pct:+.1f}% vs average comparable)")
    elif diff_pct > 0:
        st.warning(f"Your property is {diff_pct:+.1f}% above comparable average - verify premium features")
    else:
        st.info(f"Your property is {diff_pct:+.1f}% below comparable average - potential value opportunity")

    st.dataframe(comps.sort_values("Valuation", ascending=False).reset_index(drop=True), use_container_width=True)

    fig_comp = px.scatter(comps, x="sqft", y="Valuation", color="Price_per_sqft",
                          color_continuous_scale="Greys", hover_data=["bedrooms", "bathrooms"])
    fig_comp.add_scatter(x=[comp_sqft], y=[target_val], mode="markers", marker=dict(size=15, color="red", symbol="star"), name="Your Property")
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#374151", margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.divider()
st.caption("ValuaAI | Sada Santosh Kalmath")
