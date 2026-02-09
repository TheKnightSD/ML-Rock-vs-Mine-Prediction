from flask import Flask, render_template, jsonify
import numpy as np
import pickle
import random

app = Flask(__name__)

# Load ML components
model = pickle.load(open("sonar_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Global score
score = 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/scan", methods=["GET"])
def scan_ocean():
    global score

    # -------- GAME PROBABILITY LOGIC --------
    # 70% chance Rock
    # 30% chance Mine (validated by ML)

    chance = random.random()

    if chance < 0.7:
        # Mostly Rock
        result = "Rock 🪨"

    else:
        # ML-based Mine check
        values = np.array([random.random() for _ in range(60)]).reshape(1, -1)
        values_scaled = scaler.transform(values)
        prediction = model.predict(values_scaled)[0]

        if prediction == 0:
            result = "Mine 💣"
            score += 10
        else:
            result = "Rock 🪨"

    return jsonify({
        "result": result,
        "score": score
    })

@app.route("/reset", methods=["GET"])
def reset_game():
    global score
    score = 0
    return jsonify({"message": "Game reset", "score": score})

if __name__ == "__main__":
    app.run(debug=True)
