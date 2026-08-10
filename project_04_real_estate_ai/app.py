import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64

# 1. Page Configuration
st.set_page_config(
    page_title="ValuaAI — Sada Santosh Kalmath Enterprise Valuation Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Function to Load Background Image (translucent cityscape)
def add_bg_image():
    # Use an uncompressed, high-end, dark cityscape image URL
    bg_img_url = "https://images.unsplash.com/photo-1542296332-2e4473faf563?q=80&w=2670&auto=format&fit=crop"
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)), url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #f8fafc !important;
        }}
        
        /* Force sidebar transparency to inherit bg */
        [data-testid="stSidebar"] {{
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(129, 140, 248, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background image
add_bg_image()

# 3. Load Model & Scaler Artifacts
@st.cache_resource
def load_artifacts():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(base_dir, 'real_estate_model.pkl'))
        scaler = joblib.load(os.path.join(base_dir, 'real_estate_scaler.pkl'))
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Artifacts not found. Please run 'train_model.py' first inside the 'real_estate_ai' folder.")
        st.stop()

model, scaler = load_artifacts()

# 4. Premium Custom CSS (Glassmorphism Overhaul)
st.markdown("""
    <style>
    /* Master Glass Pane for Main Content */
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
        max-width: 90% !important;
    }

    /* Transparent Header */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Remove default app background */
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }

    /* Hero Section Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(49, 46, 129, 0.7) 100%);
        backdrop-filter: blur(25px) !important;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* Authorship Signature */
    .author-sig {
        font-size: 0.9rem;
        color: rgba(148, 163, 184, 0.8);
        margin-top: 15px;
        font-style: italic;
    }
    
    /* Input Box styling */
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {
        background-color: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    label[data-testid="stWidgetLabel"] {
        color: #e2e8f0 !important;
        font-weight: 600;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: transparent;
        padding: 10px 0px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8 !important;
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

# 5. Hero Banner Header with Dynamic SADA SIGNATURE
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">💎 ValuaAI — Enterprise Valuation Engine</div>
        <div class="hero-subtitle">Next-Generation Real Estate Intelligence Powered by Random Forest Ensemble ML</div>
        <div class="author-sig">✨ Advanced Analytics Engine Engineered by: SADA SANTOSH KALMATH</div>
    </div>
""", unsafe_allow_html=True)

# 6. Sidebar Controls & Info
with st.sidebar:
    # Portfolio branding image or icon
    st.image("https://cdn-icons-png.flaticon.com/512/602/602275.png", width=80)
    st.title("System Control Panel")
    st.info("Status: **Model Online (R²: 98.35%)**")
    
    st.divider()
    
    # Currency Settings
    st.markdown("### ⚙️ Region Settings")
    currency_symbol = st.selectbox("Currency Display", ["$ (USD)", "₹ (INR)", "€ (EUR)"])
    multiplier = 1.0
    curr_prefix = "$"
    if "INR" in currency_symbol:
        multiplier = 83.0
        curr_prefix = "₹"
    elif "EUR" in currency_symbol:
        multiplier = 0.92
        curr_prefix = "€"

    st.divider()

    st.markdown("### 📈 Market Trend Adjuster")
    st.write("Simulate macro-economic impacts on property values.")
    market_trend_percent = st.slider("Expected Market Growth/Decline (%)", min_value=-25.0, max_value=25.0, value=0.0, step=0.5, format="%.1f%%")
    multiplier = multiplier * (1 + (market_trend_percent / 100))
    
    st.divider()
    st.success(f"**Developer Identification Verified**\n\n©️ Sada Santosh Kalmath Analytics")

# 7. Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🧮 Single Valuation & ROI Simulator", 
    "📁 Enterprise Batch Processing", 
    "🗺️ Geospatial & Market Analytics"
])

# ==========================================
# TAB 1: SINGLE VALUATION & ROI SIMULATOR
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Base Property Characteristics")
        sqft = st.number_input("Total Area (Sqft)", min_value=500, max_value=10000, value=2200, step=50)
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3)
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=2)

    with col2:
        st.subheader("📍 Location & Condition Parameters")
        location_score = st.slider("Location Score Index (1: Poor ➔ 10: Prime)", min_value=1, max_value=10, value=7)
        age = st.slider("Property Age (Years)", min_value=0, max_value=50, value=5)

    st.divider()
    
    # What-If Renovation Simulator Toggle
    st.subheader("🛠️ What-If Renovation & ROI Simulator")
    renovate = st.checkbox("Enable Renovation Expansion Simulation")
    
    extra_sqft, extra_baths, location_boost = 0, 0, 0
    
    if renovate:
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            extra_sqft = st.slider("Add Extra Area (Sqft)", 0, 1500, 300, step=50)
        with r_col2:
            extra_baths = st.slider("Add Extra Bathrooms", 0, 3, 1)
        with r_col3:
            location_boost = st.slider("Upgrade Quality Index (+Points)", 0, 3, 1)

    if st.button("🚀 Run ML Valuation Analysis", use_container_width=True):
        # Base Prediction
        base_df = pd.DataFrame([{'sqft': sqft, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'age': age, 'location_score': location_score}])
        base_price = model.predict(scaler.transform(base_df))[0] * multiplier
        
        # Calculate Confidence Range (±MAE ~21k)
        error_margin = 21169.73 * multiplier
        min_est = base_price - error_margin
        max_est = base_price + error_margin

        # Glassmorphism Valuation Display Card
        st.markdown(f"""
            <style>
            .val-card {{
                background: rgba(129, 140, 248, 0.1) !important;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(129, 140, 248, 0.3) !important;
                padding: 30px;
                border-radius: 16px;
                text-align: center;
                margin: 20px 0;
            }}
            .val-price {{
                font-size: 3.5rem;
                font-weight: 800;
                color: #34d399;
                text-shadow: 0 0 15px rgba(52, 211, 153, 0.4);
            }}
            .val-range {{
                color: #94a3b8;
                font-size: 1.1rem;
            }}
            </style>
            <div class="val-card">
                <h3>Estimated Market Valuation (ML Inference)</h3>
                <div class="val-price">{curr_prefix}{base_price:,.2f}</div>
                <div class="val-range">Confidence Interval (±MAE): {curr_prefix}{min_est:,.2f} — {curr_prefix}{max_est:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # ROI Simulation Output
        if renovate:
            sim_df = pd.DataFrame([{
                'sqft': sqft + extra_sqft, 
                'bedrooms': bedrooms, 
                'bathrooms': bathrooms + extra_baths, 
                'age': max(0, age - 2), 
                'location_score': min(10, location_score + location_boost)
            }])
            sim_price = model.predict(scaler.transform(sim_df))[0] * multiplier
            value_gain = sim_price - base_price
            
            st.success(f"🎉 **Post-Renovation Value:** `{curr_prefix}{sim_price:,.2f}` | **Simulated Equity Gain:** `+{curr_prefix}{value_gain:,.2f}`")

        # Feature Importance Drivers (Purples scale)
        st.divider()
        st.subheader("📊 Primary Valuation Drivers (Random Forest Feature Importance)")
        importance_df = pd.DataFrame({
            'Feature': ['Square Feet', 'Bedrooms', 'Bathrooms', 'Property Age', 'Location Score'],
            'Importance (%)': model.feature_importances_ * 100
        }).sort_values(by='Importance (%)', ascending=True)
        
        fig_imp = px.bar(importance_df, x='Importance (%)', y='Feature', orientation='h', 
                         color='Importance (%)', color_continuous_scale='Purples')
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                             margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Contribution to Price (%)")
        st.plotly_chart(fig_imp, use_container_width=True)

# ==========================================
# TAB 2: ENTERPRISE BATCH PROCESSING
# ==========================================
with tab2:
    st.subheader("📁 Bulk Portfolio CSV Valuation")
    st.write("Upload property portfolio data (CSV format) to compute massive-scale automated valuations.")
    
    uploaded_file = st.file_uploader("Upload Batch File (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        req_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
        
        if all(c in batch_df.columns for c in req_cols):
            # Automated Pipeline: Scale & Predict
            scaled_batch = scaler.transform(batch_df[req_cols])
            batch_preds = model.predict(scaled_batch) * multiplier
            
            batch_df['AI_Valuation'] = batch_preds
            batch_df['Valuation_Lower_Bound'] = batch_preds - (21169.73 * multiplier)
            batch_df['Valuation_Upper_Bound'] = batch_preds + (21169.73 * multiplier)
            
            st.write("### Enterprise Valuation Results (Preview)", batch_df.head(10))
            
            # Summary Performance KPIs
            st.divider()
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total Listings Processed", len(batch_df))
            kpi2.metric("Total Portfolio Asset Value", f"{curr_prefix}{batch_df['AI_Valuation'].sum():,.2f}")
            kpi3.metric("Average Home Valuation", f"{curr_prefix}{batch_df['AI_Valuation'].mean():,.2f}")
            
            csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Final Valuation Analysis (CSV)", data=csv_bytes, file_name="ValuaAI_Batch_Report.csv", use_container_width=True)
        else:
            st.error(f"Error: Missing standard CSV columns: {req_cols}")

# ==========================================
# TAB 3: GEOSPATIAL & MARKET ANALYTICS
# ==========================================
with tab3:
    st.subheader("🗺️ Synthetic Geospatial Market Visualization")
    st.write("Interactive map displaying simulated property valuations generated across regional coordinates.")
    
    # Simulate GeoData based on model range
    np.random.seed(42)
    lat_center, lon_center = 37.7749, -122.4194 # San Francisco base coordinates
    geo_points = 150
    geo_df = pd.DataFrame({
        'sqft': np.random.randint(1000, 4500, geo_points),
        'bedrooms': np.random.randint(2, 5, geo_points),
        'bathrooms': np.random.randint(1, 4, geo_points),
        'age': np.random.randint(1, 30, geo_points),
        'location_score': np.random.randint(3, 10, geo_points),
        'lat': lat_center + np.random.normal(0, 0.05, geo_points),
        'lon': lon_center + np.random.normal(0, 0.05, geo_points)
    })
    
    geo_df['Valuation'] = model.predict(scaler.transform(geo_df[['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']])) * multiplier
    
    # Interactive Mapbox rendering
    fig_map = px.scatter_mapbox(geo_df, lat="lat", lon="lon", size="Valuation", color="Valuation",
                        color_continuous_scale="Viridis", size_max=15, zoom=10, mapbox_style="carto-darkmatter",
                        title="Simulated Regional Property Price Heatmap",
                        hover_data={'Valuation': ':.2f', 'sqft': True, 'lat': False, 'lon': False})
    fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)