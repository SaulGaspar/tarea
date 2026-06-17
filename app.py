from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapeos
LOW_MED_HIGH = {"Low": 0, "Medium": 1, "High": 2}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        
        input_data = np.array([[ 
            int(d["hours_studied"]),
            int(d["attendance"]),
            LOW_MED_HIGH[d["parental_involvement"]],
            LOW_MED_HIGH[d["access_to_resources"]],
            1 if d.get("extracurricular") == "Yes" else 0,
            int(d.get("sleep_hours", 7)),
            int(d["previous_scores"]),
            LOW_MED_HIGH[d["motivation_level"]],
            1 if d.get("internet_access") == "Yes" else 0,
            int(d.get("tutoring_sessions", 2))
        ]])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        score = max(0, min(100, round(float(prediction), 1)))
        
        return jsonify({"prediction": score, "status": "ok"})
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)