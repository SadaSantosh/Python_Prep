import json
import streamlit as st
import joblib
import re
import string
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="PhishShield",
    page_icon="🛡️",
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
        .stTextArea > div > div,
        .stSelectbox > div > div {
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
def load_nlp_artifacts():
    model = joblib.load(BASE_DIR / "spam_model.pkl")
    tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    metrics_path = BASE_DIR / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {"accuracy": 1.0, "vocab_size": 1000}
    return model, tfidf, metrics


model, tfidf, metrics = load_nlp_artifacts()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "http_link", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    return text


def analyze_urls_in_text(raw_text):
    urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", raw_text)
    if not urls:
        return []

    suspicious_tlds = [".xyz", ".top", ".online", ".site", ".club", ".info", ".live", ".cc", ".tk"]
    results = []
    for url in urls:
        flags = []
        if any(url.endswith(tld) or (tld + "/") in url for tld in suspicious_tlds):
            flags.append("Suspicious TLD")
        if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            flags.append("IP-based hostname")
        if len(url) > 55:
            flags.append("Unusually long URL")
        if "@" in url:
            flags.append("Contains @ override")
        results.append({
            "URL": url,
            "Risk_Level": "HIGH" if flags else "LOW",
            "Flags": flags or ["Standard format"],
        })
    return results


st.title("PhishShield")
st.caption("Spam and phishing detection with TF-IDF + Naive Bayes, plus URL heuristics.")

with st.sidebar:
    st.header("Model")
    st.write(f"Accuracy: {metrics['accuracy'] * 100:.1f}%")
    st.write(f"Vocabulary: {metrics['vocab_size']:,} terms")

tab1, tab2, tab3 = st.tabs(["Scanner", "Batch audit", "Metrics"])

with tab1:
    user_input = st.text_area(
        "Message to analyze",
        height=140,
        placeholder="Paste email or SMS content here...",
    )

    if st.button("Analyze message", use_container_width=True):
        if not user_input.strip():
            st.warning("Enter text before scanning.")
        else:
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]
            spam_idx = list(model.classes_).index("spam")
            spam_prob = probabilities[spam_idx] * 100

            if prediction == "spam":
                st.error(f"Spam / phishing detected — {spam_prob:.1f}% confidence")
            else:
                st.success(f"Likely legitimate — {(100 - spam_prob):.1f}% safe")

            st.subheader("URL inspection")
            url_reports = analyze_urls_in_text(user_input)
            if url_reports:
                for rep in url_reports:
                    if rep["Risk_Level"] == "HIGH":
                        st.warning(f"{rep['URL']}: {', '.join(rep['Flags'])}")
                    else:
                        st.info(f"{rep['URL']}: {', '.join(rep['Flags'])}")
            else:
                st.write("No URLs found.")

with tab2:
    uploaded_file = st.file_uploader("CSV with `text` column", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if "text" in batch_df.columns:
            cleaned_batch = batch_df["text"].astype(str).apply(clean_text)
            vec_batch = tfidf.transform(cleaned_batch)
            spam_idx = list(model.classes_).index("spam")

            batch_df["Prediction"] = model.predict(vec_batch)
            batch_df["Spam_Probability_%"] = model.predict_proba(vec_batch)[:, spam_idx] * 100

            st.subheader("Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Scanned", len(batch_df))
            c2.metric("Spam", (batch_df["Prediction"] == "spam").sum())
            c3.metric("Ham", (batch_df["Prediction"] == "ham").sum())

            st.download_button(
                "Download report",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="phishshield_report.csv",
                use_container_width=True,
            )
        else:
            st.error("CSV must include a `text` column.")

with tab3:
    st.subheader("Architecture")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "Multinomial Naive Bayes")
    m2.metric("Features", f"{metrics['vocab_size']:,} TF-IDF terms")
    m3.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")

    st.markdown(
        """
        **Pipeline**
        - Lowercase, URL normalization, punctuation and number removal
        - TF-IDF vectorization with English stop words
        - Naive Bayes classification
        - Rule-based URL heuristics for TLD, IP, and length checks
        """
    )

st.divider()
st.caption("PhishShield · Sada Santosh Kalmath")
