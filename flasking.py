from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from predict import predict_root

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=["POST"])
@cross_origin()
def index():
    payload = dict(request.json)
    input = payload.get("input")
    prediction = predict_root(input)
    label, probs = prediction[0], prediction[1]
    probs = [float(element) for element in probs]
    return jsonify({"label":label, "probs":probs})

app.run(port=8000)