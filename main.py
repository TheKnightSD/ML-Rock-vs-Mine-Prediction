from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os   

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("sonar_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_file", methods=["POST"])
def predict_file():
    try:
        file = request.files["file"]
        data = pd.read_csv(file, header=None, encoding="latin-1")
        values = data.values.flatten()

        if len(values) != 60:
            return render_template(
                "index.html",
                prediction=f"Error: Expected 60 values, got {len(values)}"
            )

        input_scaled = scaler.transform(values.reshape(1, -1))
        pred = model.predict(input_scaled)

        result = "Mine 💣" if pred[0] == 0 else "Rock 🪨"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Invalid CSV file")

@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        raw = request.form["manual_data"]
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]

        if len(values) != 60:
            return render_template(
                "index.html",
                prediction=f"Error: Expected 60 values, got {len(values)}"
            )

        input_scaled = scaler.transform(np.array(values).reshape(1, -1))
        pred = model.predict(input_scaled)

        result = "Mine 💣" if pred[0] == 0 else "Rock 🪨"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction="Invalid input values")

# ✅ RENDER COMPATIBLE RUN
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
