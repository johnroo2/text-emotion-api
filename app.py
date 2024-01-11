from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from predict import predict_root

app = Flask(__name__, static_url_path='/static')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['ENV'] = 'production'
app.config['MAX_CONTENT_LENGTH'] = 10485760

@app.route("/", methods=["POST"])
@cross_origin()
def index():
    payload = dict(request.json)
    input_data = payload.get("input")
    prediction = predict_root(input_data)
    label, probs = prediction[0], prediction[1]
    probs = [float(element) for element in probs]
    return jsonify({"label": label, "probs": probs})

if __name__ == "__main__":
    app.run()