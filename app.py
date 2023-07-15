from flask import Flask, jsonify, request
from classifier import get_prediction
import numpy as np

app = Flask(__name__)

@app.route("/predict-alphabet", methods=["POST"])
def predict_data():
    # if "alphabet" not in request.files:
    #     return jsonify({"error": "No image file found"})

    image = request.files["alphabet"]
    prediction = get_prediction(image)

    # Convert NumPy array to Python list
    prediction = prediction.tolist()

    return jsonify({"prediction": prediction}), 200

if __name__ == "__main__":
    app.run(debug=True)