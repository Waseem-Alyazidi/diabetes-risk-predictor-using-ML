"""
app.py

Purpose
-------
    Provide a Flask-based web API and frontend integration for predicting 
    diabetes risk levels using a trained Random Forest model.

Description
-----------
    This application:
    1. Loads the trained Random Forest model from 'backend/model.pkl' using dill.
    2. Serves a homepage ('/') rendering an HTML template.
    3. Exposes a '/predict' endpoint that:
       - Accepts JSON input with health and lifestyle features 
         (age, height, weight, bp, glucose, activity, diet, family_history).
       - Preprocesses inputs using utility functions (BMI calculation 
         and categorical encoding).
       - Runs the model to predict the user's diabetes risk level.
       - Returns the prediction as JSON.

Usage
-----
    Run this script to start a Flask development server:
        python app.py

    Send a POST request with JSON data to '/predict' to get a risk prediction.

Dependencies
------------
    - dill
    - numpy
    - flask
    - pathlib
    - utils (calculate_bmi, data_encoding)

Author
------
    Waseem Alyazidi.

Date
----
    2025-09-09.
"""

import numpy as np
import dill # type: ignore
from pathlib import Path
from collections import Counter # For handel dill "name 'Counter' is not defined" error
from flask import Flask, request, jsonify, render_template
from utils import calculate_bmi, data_encoding # type: ignore

app = Flask(__name__, template_folder=r"../frontend/templates", static_folder=r"../frontend/statics")

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
try:
    # Load the model
    with open(MODEL_PATH, mode="rb") as f:
        model = dill.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}.\n")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    import traceback
    data = request.json

    required_fields = ["age", "height", "weight", "bp", "glucose", "activity", "diet", "family_history"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
        
    # Save inputs
    try:
        age = float(data["age"])
        bp = float(data["bp"])
        glucose = float(data["glucose"])

        bmi = calculate_bmi(height = float(data["height"]), weight = float(data["weight"]))
        activity = data_encoding("activity", data["activity"])
        diet = data_encoding("diet", data["diet"])
        family_history = data_encoding("family_history", data["family_history"])

        X_new = np.array([[age, bmi, bp, glucose, activity, diet, family_history]])
        prediction = int(model.predict(X_new)[0])
        if prediction not in [0, 1, 2, 3, 4]:
            prediction = -1
    except Exception as e:
        return jsonify({"error": str(traceback.format_exc())}), 500
    return jsonify({"risk_level": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
