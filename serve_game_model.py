from flask import Flask
from flask import jsonify, request
import tensorflow as tf
import numpy as np
import json


app = Flask(__name__)

def createDetailedPredictions(preds):
    print("Probabilities of next moves are in the decreasing order:")
    preds = preds[0]
    preds_with_pos = [(preds[i], i) for i in range(25)]
    preds_with_pos.sort(reverse=True)

    probs = {}
    for i in range(25):
        next_x = preds_with_pos[i][1] // 5
        next_y = preds_with_pos[i][1] % 5
        # print("Next Move: ", f'({next_x} , {next_y})', " | Probability: ", preds_with_pos[i][0])
        probs[str(preds_with_pos[i][0])] = str(next_x) + str(next_y)
    # print("\n", "="*30, "\n")
    return probs

model = tf.keras.models.load_model('./player_1_moves_model.h5')

@app.route('/predict', methods=["POST"])
def index():
    data = dict(request.get_json())
    inp = data.get("state")
    getDetailed = data.get("getDetailed")

    inp = inp.split(",")
    int_input = np.array([int(i) for i in inp])
    int_input = int_input.reshape(1,25)

    preds = model.predict(int_input)
    pred_pos = np.argmax(preds)

    next_x = pred_pos//5
    next_y = pred_pos % 5

    probs = None
    if getDetailed:
        probs = createDetailedPredictions(preds)

    return jsonify(probs)

if __name__ == '__main__':
    app.run(debug=True)