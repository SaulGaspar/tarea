from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()

        # Mapeo de categorías
        low_med = {"Low": 0, "Medium": 1, "High": 2}

        # Crear input en el orden correcto
        input_data = np.array([[
            float(d["hours_studied"]),
            float(d["attendance"]),
            low_med.get(d.get("parental_involvement"), 1),
            low_med.get(d.get("access_to_resources"), 1),
            low_med.get(d.get("motivation_level"), 1),
            float(d["previous_scores"])
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        score = max(0, min(100, round(float(prediction), 1)))
        return jsonify({"prediction": score, "status": "ok"})

    except Exception as e:
        print("Error en /predict:", str(e))
        return jsonify({"error": str(e), "status": "error"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)