from flask import Flask, render_template, request, jsonify
from fnn import FeedForwardNN
from cnn import ConvolutionalNN
from PIL import Image
import io
import os
import base64
import numpy as np

app = Flask(__name__)

nn = None


@app.route("/changeModel", methods=["POST"])
def change_model():
    global nn

    data = str(request.get_data())[2:-1]

    if data == "FNN":
        nn = fnn
    else:
        nn = cnn
    
    return jsonify({"message": "Model changed successfully", "model": data})


@app.route("/")
def index():
    global nn
    global cnn
    global fnn

    if nn is None:
        defaultModel = "cnn"

        # Load default configurations
        cnn = ConvolutionalNN()
        model_filename = "./tmp/loaded_models/convolutional.pkl"

        if os.path.exists(model_filename):
            cnn.load_model(model_filename)
            print("Model loaded from file.")
        else:
            cnn.train(epochs=1)
            print("New model trained and saved to file.")
        # ---------------------------

        # Load other model
        fnn = FeedForwardNN()
        model_filename = "./tmp/loaded_models/feed-forward.pkl"

        if os.path.exists(model_filename):
            fnn.load_model(model_filename)
            print("Model loaded from file.")
        else:
            fnn.train(epochs=1)
            print("New model trained and saved to file.")
        # ---------------------------

        if defaultModel == "cnn":
            nn = cnn
        elif defaultModel == "fnn":
            nn = fnn
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
    greyscale_image_array = process_image(image_data)

    prediction = nn.get_prediction(greyscale_image_array)
    
    if isinstance(prediction, (np.ndarray, np.generic)):
        return jsonify(prediction.tolist())
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True)
