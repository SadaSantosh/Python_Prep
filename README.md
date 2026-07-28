# 🔮 Enterprise Telco Customer Churn Prediction Engine
![App Preview](Project_03_Telco_Customer_Churn/dashboard.png)
An end-to-end Machine Learning web application that predicts real-time customer churn probability for telecommunications providers. Built using **Python**, **XGBoost**, **SMOTE**, and **Streamlit**, and deployed live on **Streamlit Cloud**.

👉 **[Live Interactive App](https://sadasantosh-telco-churn.streamlit.app)**

---

## 📌 Business Overview & Problem Statement
Customer churn poses a significant revenue threat in the telecommunications industry. Acquiring a new customer can cost up to 5 times more than retaining an existing one. 

This project delivers an interactive decision-support engine that enables retention teams to:
1. Input customer demographic data, tenure, billing history, and subscribed services.
2. Calculate real-time churn risk percentages powered by an optimized machine learning pipeline.
3. Automatically flag high-risk accounts to trigger proactive retention offers.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.11
* **Machine Learning:** XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib
* **Web Framework:** Streamlit
* **Deployment & CI/CD:** Streamlit Community Cloud, GitHub

---

## ⚙️ Machine Learning Pipeline Architecture

1. **Exploratory Data Analysis (EDA):** Cleaned raw Telco customer records, handled missing total charges, and engineered key ratio features (`Avg_Monthly_Cost`, `Monthly_Price_Diff`).
2. **Feature Engineering & Scaling:** One-Hot Encoded categorical variables and standardized numerical features using `StandardScaler`.
3. **Handling Class Imbalance:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on training splits to prevent model bias towards majority non-churning profiles.
4. **Model Training & Evaluation:** Trained an XGBoost Classifier tuned to optimize **ROC-AUC** and **Recall** for high-risk customer identification.

---

## 🚀 Local Setup & Installation

To run this app locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SadaSantosh/Python_Prep.git](https://github.com/SadaSantosh/Python_Prep.git)
   cd Python_Prep