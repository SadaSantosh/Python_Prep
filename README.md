# 🚀 Applied Machine Learning & Data Science Projects

Welcome to my machine learning portfolio! This repository contains end-to-end data science projects ranging from core data analysis to production-deployed predictive web applications.

👉 **Featured Deployment:** [Live Telco Churn Prediction Engine](https://sadasantosh-telco-churn.streamlit.app)

---

## 📌 Portfolio Index & Overview

| Project                                | Domain                     | Key Technologies                     | Status / Link                                                |
| :------------------------------------- | :------------------------- | :----------------------------------- | :----------------------------------------------------------- |
| **01. Student Performance Classifier** | Education Analytics        | Python, Scikit-Learn, Decision Trees | Completed                                                    |
| **02. Corporate Employee Analytics**   | HR & Workforce Analytics   | Pandas, Seaborn, Attrition EDA       | Completed                                                    |
| **03. Telco Customer Churn Engine**    | Telecom Customer Retention | XGBoost, SMOTE, Streamlit Cloud      | 🚀 [Live App](https://sadasantosh-telco-churn.streamlit.app) |
| **04. Real Estate AI (ValuaAI)**       | Real Estate Valuation      | Random Forest, Plotly, Streamlit     | Completed                                                    |
| **05. Spam & Phishing Detector**      | Cybersecurity / NLP        | TF-IDF, Naive Bayes, Streamlit       | Completed                                                    |

---

## 🎓 Project 01: Student Performance Classifier

### 📌 Project 01 Problem Statement

Predicting student academic outcomes based on demographic, behavioral, and historical academic attributes to enable early intervention for at-risk students.

### 🛠️ Project 01 Key Technical Highlights

- **Task:** Multi-Class / Binary Classification
- **Algorithms Applied:** Logistic Regression, Decision Trees, Random Forest
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-Score evaluation across demographic features.

---

## 🏢 Project 02: Corporate Employee Analytics

### 📌 Project 02 Problem Statement

An exploratory data analysis (EDA) and predictive modeling project analyzing workforce metrics, employee satisfaction scores, and primary drivers of corporate attrition.

### 🛠️ Project 02 Key Technical Highlights

- **Task:** Exploratory Data Analysis & Feature Importance
- **Techniques:** Feature Correlation Heatmaps, Demographic Profiling, Attrition Rate Isolation
- **Data Processing:** Categorical encoding, outlier handling, and statistical hypothesis testing.

---

## 🔮 Project 03: Enterprise Telco Customer Churn Prediction Engine

### 📌 Project 03 Problem Statement

Customer churn poses a significant revenue threat in telecommunications. This project delivers an interactive decision-support engine that enables retention teams to evaluate customer churn risk in real time.

![Telco Churn Dashboard](Project_03_Telco_Customer_Churn/dashboard.png)

### 🛠️ Project 03 Key Technical Highlights

- **Imbalance Handling:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to train balanced models on imbalanced churn data.
- **Model Pipeline:** Trained an **XGBoost Classifier** optimized for high-recall customer risk detection.
- **Deployment:** Deployed an interactive frontend using **Streamlit** on **Streamlit Community Cloud**.
- **UI Design:** **Liquid Glass (Glassmorphism)** frosted-glass interface with blur effects, semi-transparent cards, and gradient accents.
- **Live App Link:** [https://sadasantosh-telco-churn.streamlit.app](https://sadasantosh-telco-churn.streamlit.app)

---

## 🏡 Project 04: ValuaAI — Real Estate Price Estimator

### 📌 Project 04 Problem Statement

A machine learning-powered real estate price estimation engine that enables property valuation with ROI simulation, batch portfolio analysis, and geospatial mapping.

### 🛠️ Project 04 Key Technical Highlights

- **Model:** Random Forest Regression (R² ≈ 98.35%)
- **Features:** Single property inference, renovation simulator, bulk CSV valuation, interactive map visualization.
- **UI Design:** **Liquid Glass (Glassmorphism)** frosted-glass interface with blur effects, semi-transparent cards, and gradient accents.

---

## 🛡️ Project 05: PhishShield — Spam & Phishing Detector

### 📌 Project 05 Problem Statement

An NLP-powered spam and phishing detection system that classifies messages using TF-IDF vectorization and rule-based URL heuristics for real-time threat analysis.

### 🛠️ Project 05 Key Technical Highlights

- **Model:** Multinomial Naive Bayes with TF-IDF features.
- **Features:** Single message scanner, batch CSV audit, URL inspection (suspicious TLD, IP-based, length checks).
- **UI Design:** **Liquid Glass (Glassmorphism)** frosted-glass interface with blur effects, semi-transparent cards, and gradient accents.

---

## ⚙️ Global Local Setup & Running Instructions

To clone and run any of these projects locally on your computer:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SadaSantosh/Python_Prep.git
   cd Python_Prep
   ```

2. **Create a Virtual Environment & Install Dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run a Streamlit App (Projects 03, 04, 05):**

   ```bash
   cd Project_03_Telco_Customer_Churn
   streamlit run app.py
   ```

4. **Train Models from Scratch (optional):**

   ```bash
   cd Project_03_Telco_Customer_Churn
   python data_preprocessing.py
   python model_training.py
   ```

---

## 👤 Author

Engineered by **Sada Santosh Kalmath**
