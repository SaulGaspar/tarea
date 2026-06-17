import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Cargar datos
df = pd.read_csv("StudentPerformanceFactors.csv")

# Imputar valores faltantes
imputer = SimpleImputer(strategy='most_frequent')
df[['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']] = imputer.fit_transform(
    df[['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']]
)

# Encoding
ordinal_cols = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 
                'Family_Income', 'Teacher_Quality']
ordinal_encoder = OrdinalEncoder()
df[ordinal_cols] = ordinal_encoder.fit_transform(df[ordinal_cols])

binary_cols = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 
               'School_Type', 'Gender']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Public': 0, 'Private': 1, 'Male': 0, 'Female': 1})

df['Peer_Influence'] = df['Peer_Influence'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
df['Distance_from_Home'] = df['Distance_from_Home'].map({'Near': 0, 'Moderate': 1, 'Far': 2})
df['Parental_Education_Level'] = df['Parental_Education_Level'].map(
    {'High School': 0, 'College': 1, 'Postgraduate': 2}
)

# Features y target
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

# Selección de características con RFE (10 características)
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_].tolist()

# Guardar features seleccionadas
joblib.dump(selected_features, 'features.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo final
final_model = LinearRegression()
final_model.fit(X_train_scaled, y_train)

# Guardar
joblib.dump(final_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Modelo entrenado y guardado correctamente.")
print(f"Características usadas: {selected_features}")