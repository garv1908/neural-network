from flask import Flask, render_template, request, jsonify
from nn import NeuralNetwork
from PIL import Image
import io
import base64
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

nn = NeuralNetwork()
nn.train(epochs=1)

@app.route("/")
def index():
    return render_template("index.html")

def process_image(image_data):
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    image = image.resize((28, 28))
    image = np.array(image)

    grey_image = image[:, :, 3]

    grey_image = grey_image.astype("float32") / 255

    return grey_image

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image']
    greyscale_image = process_image(image_data)

    prediction = nn.get_prediction(greyscale_image)

    return jsonify(prediction.tolist())

if __name__ == "__main__":
    app.run(debug=True)
