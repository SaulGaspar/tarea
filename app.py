from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age      = float(data["age"])
        sex      = 0 if data["sex"] == "male" else 1
        bmi      = float(data["bmi"])
        children = int(data["children"])
        smoker   = 1 if data["smoker"] == "yes" else 0
        region   = {"northwest": 0, "northeast": 1, "southeast": 2, "southwest": 3}[data["region"]]

        features = np.array([[age, sex, bmi, children, smoker, region]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return jsonify({"prediction": round(float(prediction), 2), "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    app.run(debug=True)
