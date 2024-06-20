from flask import Flask, render_template, request, jsonify
from nn import NeuralNetwork
from PIL import Image
import io
import base64
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

nn = NeuralNetwork()
nn.train()

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

    print('data:', data)
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image']
    print('image data:', image_data)
    greyscale_image = process_image(image_data)
    print(f"Processed data: {greyscale_image}")

    prediction = nn.get_prediction(greyscale_image)
    label = np.argmax(prediction)

    print(prediction)
    print(label)
    
    return jsonify({'prediction': int(label)})

if __name__ == "__main__":
    app.run(debug=True)
