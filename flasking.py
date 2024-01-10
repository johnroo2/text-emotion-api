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
    return jsonify({"prediction":"hello"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)