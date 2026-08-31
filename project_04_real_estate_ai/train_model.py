import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
n_samples = 1000

sqft = np.random.randint(600, 4500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 5, n_samples)
age = np.random.randint(0, 40, n_samples)
location_score = np.random.randint(1, 10, n_samples)

price = (
    50000
    + (sqft * 180)
    + (bedrooms * 15000)
    + (bathrooms * 20000)
    - (age * 1200)
    + (location_score * 25000)
    + np.random.normal(0, 15000, n_samples)
)

df = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "location_score": location_score,
    "price": price,
})

X = df[["sqft", "bedrooms", "bathrooms", "age", "location_score"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print("REAL ESTATE AI MODEL EVALUATION")
print("=" * 40)
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R² Score: {r2 * 100:.2f}%")
print("=" * 40)

joblib.dump(model, os.path.join(BASE_DIR, "real_estate_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "real_estate_scaler.pkl"))

metrics = {"mae": mae, "rmse": rmse, "r2": r2}
with open(os.path.join(BASE_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Saved model artifacts and model_metrics.json")
