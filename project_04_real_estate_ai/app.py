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

st.markdown(
    """
    <style>
        .stApp { background-color: #fafafa; }
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        h1, h2, h3 { color: #111827; font-weight: 600; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
        }
        .stButton > button {
            background: #111827 !important;
            color: #ffffff !important;
            border-radius: 6px !important;
        }
        .block-container { padding-top: 2rem; max-width: 1100px; }
    </style>
    """,
    unsafe_allow_html=True,
)


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

tab1, tab2, tab3 = st.tabs(["Valuation", "Batch processing", "Market map"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        sqft = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=2200, step=50)
        bedrooms = st.slider("Bedrooms", 1, 8, 3)
        bathrooms = st.slider("Bathrooms", 1, 6, 2)
    with col2:
        location_score = st.slider("Location score (1–10)", 1, 10, 7)
        age = st.slider("Property age (years)", 0, 50, 5)

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
            f"Confidence range (±MAE): "
            f"{curr_prefix}{base_price - error_margin:,.0f} — {curr_prefix}{base_price + error_margin:,.0f}"
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
    st.subheader("Batch valuation")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    req_cols = ["sqft", "bedrooms", "bathrooms", "age", "location_score"]

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if all(c in batch_df.columns for c in req_cols):
            preds = model.predict(scaler.transform(batch_df[req_cols])) * multiplier
            batch_df["AI_Valuation"] = preds
            batch_df["Lower_Bound"] = preds - (mae * multiplier)
            batch_df["Upper_Bound"] = preds + (mae * multiplier)

            st.subheader("Results preview")
            st.dataframe(batch_df.head(10), use_container_width=True)

            k1, k2, k3 = st.columns(3)
            k1.metric("Listings", len(batch_df))
            k2.metric("Portfolio value", f"{curr_prefix}{batch_df['AI_Valuation'].sum():,.0f}")
            k3.metric("Average value", f"{curr_prefix}{batch_df['AI_Valuation'].mean():,.0f}")

            st.download_button(
                "Download CSV",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="valuaai_batch_report.csv",
                use_container_width=True,
            )
        else:
            st.error(f"Missing columns. Required: {req_cols}")

with tab3:
    st.subheader("Regional price map (synthetic)")
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

st.divider()
st.caption("ValuaAI · Sada Santosh Kalmath")
