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
        /* Liquid Glass — Glassmorphism UI */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 50%, #fce7f3 100%);
            font-family: 'Inter', sans-serif;
        }
        .stApp::before {
            content: '';
            position: fixed;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle at 30% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
                        radial-gradient(circle at 70% 80%, rgba(236,72,153,0.06) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        .stApp > * { position: relative; z-index: 1; }

        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.55) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
            border-right: 1px solid rgba(255,255,255,0.6) !important;
            box-shadow: 4px 0 30px rgba(0,0,0,0.05) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdown"] {
            color: #1e1b4b !important;
        }

        h1, h2, h3 {
            color: #1e1b4b !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }

        /* Glass card metrics */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.6) !important;
            backdrop-filter: blur(16px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
            border: 1px solid rgba(255,255,255,0.7) !important;
            border-radius: 16px !important;
            padding: 20px 24px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.8) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 40px rgba(0,0,0,0.1) !important;
        }

        /* Glass button */
        .stButton > button {
            background: linear-gradient(135deg, rgba(99,102,241,0.85), rgba(139,92,246,0.85)) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            border-radius: 12px !important;
            font-weight: 500 !important;
            padding: 10px 24px !important;
            backdrop-filter: blur(8px) !important;
            box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
            transition: all 0.25s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(79,70,229,0.95), rgba(124,58,237,0.95)) !important;
            box-shadow: 0 6px 28px rgba(99,102,241,0.45) !important;
            transform: translateY(-1px) !important;
        }

        /* Glass tabs */
        [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.4) !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 14px !important;
            padding: 4px !important;
            border: 1px solid rgba(255,255,255,0.6) !important;
        }
        [data-baseweb="tab"] { border-radius: 10px !important; font-weight: 500 !important; }
        [aria-selected="true"] {
            background: rgba(255,255,255,0.7) !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
        }

        /* Glass input fields */
        .stTextInput > div > div,
        .stNumberInput > div > div,
        .stSelectbox > div > div,
        .stSlider > div > div {
            background: rgba(255,255,255,0.5) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255,255,255,0.7) !important;
            border-radius: 10px !important;
        }

        .stDataFrame {
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.06) !important;
            border: 1px solid rgba(255,255,255,0.6) !important;
        }

        .stAlert {
            background: rgba(255,255,255,0.55) !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.6) !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.04) !important;
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
