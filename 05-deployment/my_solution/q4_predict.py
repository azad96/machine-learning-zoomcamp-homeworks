import pickle
from flask import Flask, jsonify, request


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
app = Flask(__name__)
model = load_pickle('model1.bin')
dv = load_pickle('dv.bin')

@app.route('/predict', methods=["POST"])
def predict():
    client = request.get_json()

    X = dv.transform(client)
    y_pred = model.predict_proba(X)[:, 1]
    churn = y_pred >= 0.5

    result = {
        'get_subscription_prob': float(y_pred),
        'get_subscription': bool(churn)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
