from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin
from predict import predict_root

app = Flask(__name__)
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

@app.route('/.well-known/pki-validation/39D3AE96C00146317DED8ECBBA47F9AA.txt')
def serve_txt_file():
    file_path = './.well-known/pki-validation/39D3AE96C00146317DED8ECBBA47F9AA.txt'
    return send_file(file_path, mimetype='text/plain')

if __name__ == "__main__":
    app.run()