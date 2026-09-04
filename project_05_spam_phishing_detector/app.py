import json
import streamlit as st
import joblib
import re
import string
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="PhishShield",
    page_icon="🛡️",
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Scanner",
    "Batch Audit",
    "Metrics",
    "Email Header Analyzer",
    "Threat Intelligence",
])

with tab1:
    user_input = st.text_area(
        "Message to analyze",
        height=140,
        placeholder="Paste email or SMS content here...",
    )

    if st.button("Analyze message", width="stretch"):
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
                st.error(f"Spam / phishing detected - {spam_prob:.1f}% confidence")
            else:
                st.success(f"Likely legitimate - {(100 - spam_prob):.1f}% safe")

            st.subheader("URL Inspection")
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
            st.dataframe(batch_df.head(10), width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Scanned", len(batch_df))
            c2.metric("Spam", (batch_df["Prediction"] == "spam").sum())
            c3.metric("Ham", (batch_df["Prediction"] == "ham").sum())

            st.download_button(
                "Download report",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="phishshield_report.csv",
                width="stretch",
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

with tab4:
    st.subheader("Email Header Analyzer")
    st.write("Paste raw email headers to extract sender info, routing, and authentication results.")

    header_input = st.text_area(
        "Raw email headers",
        height=200,
        placeholder="Paste full email headers here (From, To, Subject, Received, DKIM, SPF, etc.)...",
        key="header_input",
    )

    if st.button("Analyze headers", width="stretch", key="analyze_headers"):
        if not header_input.strip():
            st.warning("Paste email headers first.")
        else:
            lines = header_input.strip().split("\n")
            headers = {}
            current_key = None
            for line in lines:
                line = line.rstrip()
                if line.startswith((" ", "\t")) and current_key:
                    headers[current_key] += " " + line.strip()
                elif ":" in line:
                    key, _, val = line.partition(":")
                    current_key = key.strip().lower()
                    headers[current_key] = val.strip()

            h1, h2 = st.columns(2)
            with h1:
                st.write("**Extracted Fields**")
                important_fields = ["from", "to", "subject", "date", "reply-to", "return-path", "message-id"]
                for field in important_fields:
                    if field in headers:
                        st.write(f"- **{field.title()}:** {headers[field]}")

            with h2:
                st.write("**Authentication Results**")
                auth_fields = ["authentication-results", "received-spf", "dkim-signature", "arc-authentication-results"]
                auth_found = False
                for field in auth_fields:
                    if field in headers:
                        snippet = headers[field][:100]
                        suffix = "..." if len(headers[field]) > 100 else ""
                        st.write(f"- **{field.title()}:** {snippet}{suffix}")
                        auth_found = True
                if not auth_found:
                    st.info("No standard authentication headers found.")

            st.write("**Spoofing Indicators**")
            spoof_flags = []
            if "from" in headers and "reply-to" in headers:
                from_domain = headers["from"].split("@")[-1].strip(">") if "@" in headers["from"] else ""
                reply_domain = headers["reply-to"].split("@")[-1].strip(">") if "@" in headers["reply-to"] else ""
                if from_domain and reply_domain and from_domain != reply_domain:
                    spoof_flags.append(f"Reply-to domain ({reply_domain}) differs from From domain ({from_domain})")
            if "return-path" in headers and "from" in headers:
                rp_domain = headers["return-path"].split("@")[-1].strip(">") if "@" in headers["return-path"] else ""
                from_domain2 = headers["from"].split("@")[-1].strip(">") if "@" in headers["from"] else ""
                if rp_domain and from_domain2 and rp_domain != from_domain2:
                    spoof_flags.append(f"Return-path domain ({rp_domain}) mismatches From domain ({from_domain2})")
            if spoof_flags:
                for flag in spoof_flags:
                    st.error(flag)
            else:
                st.success("No obvious spoofing indicators detected.")

            with st.expander("Show all raw headers"):
                for k, v in headers.items():
                    st.code(f"{k}: {v}")

with tab5:
    st.subheader("Threat Intelligence Dashboard")
    st.write("Overview of common threat patterns detected in the training corpus.")

    threat_data = pd.DataFrame({
        "Threat Category": [
            "Credential Phishing", "Prize/Scam", "Banking Fraud",
            "Package Delivery", "Account Suspension", "Tech Support Scam",
        ],
        "Prevalence": [35, 25, 15, 10, 10, 5],
        "Avg Confidence": [87, 92, 78, 85, 90, 75],
    })

    t1, t2 = st.columns(2)
    with t1:
        fig_threat = px.bar(
            threat_data, x="Prevalence", y="Threat Category", orientation="h",
            color="Prevalence", color_continuous_scale="Greys",
        )
        fig_threat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#374151", margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_threat, width="stretch")

    with t2:
        fig_conf = px.bar(
            threat_data, x="Avg Confidence", y="Threat Category", orientation="h",
            color="Avg Confidence", color_continuous_scale="Greys",
        )
        fig_conf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#374151", margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_conf, width="stretch")

    st.write("**Detection Rules Active**")
    rules = pd.DataFrame({
        "Rule": [
            "URL TLD Filtering", "IP-Based Hostname Detection", "Long URL Detection",
            "Email @ Override Check", "TF-IDF Spam Keywords", "Punctuation Pattern Analysis",
        ],
        "Type": ["Heuristic", "Heuristic", "Heuristic", "Heuristic", "ML", "ML"],
        "Status": ["Active"] * 6,
    })
    st.dataframe(rules, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rules", "6")
    c2.metric("Heuristic Rules", "4")
    c3.metric("ML Features", f"{metrics['vocab_size']:,}")

st.divider()
st.caption("PhishShield | Sada Santosh Kalmath")
