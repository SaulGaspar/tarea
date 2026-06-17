import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("StudentPerformanceFactors.csv")

# Imputar faltantes
imputer = SimpleImputer(strategy='most_frequent')
df[['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']] = imputer.fit_transform(
    df[['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']]
)

# Encoding
low_med = {"Low": 0, "Medium": 1, "High": 2}
for col in ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality']:
    df[col] = df[col].map(low_med)

df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes':1, 'No':0})
df['Internet_Access'] = df['Internet_Access'].map({'Yes':1, 'No':0})
df['Learning_Disabilities'] = df['Learning_Disabilities'].map({'Yes':1, 'No':0})
df['School_Type'] = df['School_Type'].map({'Private':1, 'Public':0})
df['Gender'] = df['Gender'].map({'Female':1, 'Male':0})
df['Peer_Influence'] = df['Peer_Influence'].map({'Negative':0, 'Neutral':1, 'Positive':2})

# Seleccionamos solo las features más importantes
features = ['Hours_Studied', 'Attendance', 'Parental_Involvement', 
            'Access_to_Resources', 'Motivation_Level', 'Previous_Scores']

X = df[features]
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"MAE: {mean_absolute_error(y_test, model.predict(X_test_scaled)):.2f}")
print(f"R²: {r2_score(y_test, model.predict(X_test_scaled)):.4f}")

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Modelo entrenado correctamente con 6 features")