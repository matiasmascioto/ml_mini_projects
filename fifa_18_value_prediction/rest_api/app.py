"""
REST API
"""

# Imports
from collections import OrderedDict
from flask import Flask, jsonify, request
import pickle
import numpy as np

# Model Features (default value: 0)
FEATURES = OrderedDict([("Overall", 0),
                        ("Potential", 0),
                        ("Wage", 0),
                        ("Special", 0),
                        ("Ball control", 0),
                        ("Composure", 0),
                        ("Reactions", 0),
                        ("Short passing", 0),
                        ("CAM", 0),
                        ("CF", 0),
                        ("CM", 0),
                        ("LAM", 0),
                        ("LCM", 0),
                        ("LM", 0),
                        ("LS", 0),
                        ("RAM", 0),
                        ("RCM", 0),
                        ("RM", 0),
                        ("RS", 0),
                        ("ST", 0)])

app = Flask(__name__)

# Load model
with open("lin_reg_model.pkl", "rb") as f:
    lin_reg = pickle.load(f)


@app.route("/predict", methods=["GET"])
def predict_ml():
    """ """
    for key, value in request.args.items():
        if key in FEATURES.keys():
            FEATURES[key] = int(value)
    prediction = lin_reg.predict([np.asarray(list(FEATURES.values()))])
    prediction = "€{:0,.2f}".format(round(prediction[0][0], 2))
    return jsonify({"value_prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True, port=8080)