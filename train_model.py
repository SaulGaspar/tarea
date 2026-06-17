import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("insurance.csv")

# Encoding
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map({"northwest": 0, "northeast": 1, "southeast": 2, "southwest": 3})

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
print(f"MAE: ${mean_absolute_error(y_test, preds):.2f}")
print(f"R²: {r2_score(y_test, preds):.4f}")

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Modelo y scaler guardados.")
