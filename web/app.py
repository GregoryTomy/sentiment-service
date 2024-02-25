from flask import Flask, request, send_from_directory
from flask_cors import CORS
from main import get_sentiment

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route("/")
def index():
    return app.send_static_file('sentiment.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        input = request.args.get("input")
    else:
        input = request.get_json(force=True)["input"]
    
    if not input:
        return "No input value found"
    
    return get_sentiment(input)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4000) 