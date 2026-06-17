from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo, scaler y features seleccionadas
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("features.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        
        # Crear vector con las 10 características seleccionadas
        input_data = np.array([[
            int(d["hours_studied"]),
            int(d["attendance"]),
            int(d.get("parental_involvement", "Medium")),  # se convertirá después
            int(d.get("access_to_resources", "Medium")),
            1 if d.get("extracurricular", "Yes") == "Yes" else 0,
            int(d.get("sleep_hours", 7)),
            int(d["previous_scores"]),
            int(d.get("motivation_level", "Medium")),
            1 if d.get("internet_access", "Yes") == "Yes" else 0,
            int(d.get("tutoring_sessions", 2))
        ]])
        
        # Convertir valores categóricos a numéricos
        low_med = {"Low": 0, "Medium": 1, "High": 2}
        input_data[0][2] = low_med.get(d.get("parental_involvement"), 1)
        input_data[0][3] = low_med.get(d.get("access_to_resources"), 1)
        input_data[0][7] = low_med.get(d.get("motivation_level"), 1)
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        score = max(0, min(100, round(float(prediction), 1)))
        return jsonify({"prediction": score, "status": "ok"})
    
    except Exception as e:
        print("Error:", str(e))  # Para ver el error en consola
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)