# from flask import Flask, render_template, request, jsonify
# from nn import GetPrediction
# import numpy as np

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     imgArr = np.array(data['image']).reshape(784, 1)
#     prediction = GetPrediction(imgArr)
#     predictedLabel = np.argmax(prediction)
#     print(prediction)
#     return jsonify({'prediction': int(predictedLabel)})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from nn import GetPrediction
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def process_image(image_data):

    print(image_data)
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_bytes))

    return np.array(image)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image']
    processed_data = process_image(image_data)
    print(f"Processed data: {processed_data[:100]}")

    prediction = GetPrediction(np.array(processed_data).reshape(784, 1))
    predicted_label = np.argmax(prediction)
    print(f"Prediction: {prediction}, Predicted label: {predicted_label}")
    
    return jsonify({'prediction': int(predicted_label)})

if __name__ == "__main__":
    app.run(debug=True)
