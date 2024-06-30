from flask import Flask, render_template, request, jsonify
from fnn import FeedForwardNN
from PIL import Image
import io
import os
import base64
import numpy as np

app = Flask(__name__)

nn = FeedForwardNN()

model_filename = "./tmp/loaded_models/feed-forward.pkl"

if os.path.exists(model_filename):
    nn = nn.load_model(model_filename)
    print("Model loaded from file.")
else:
    nn.train(epochs=1, save_model=False)
    print("New model trained and saved to file.")

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
