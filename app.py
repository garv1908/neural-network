from flask import Flask, render_template, request, jsonify
from nn import GetPrediction
from PIL import Image
import io
import base64
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def processImage(imageData):

    imageData = imageData.split(',')[1]
    imageBytes = base64.b64decode(imageData)
    image = Image.open(io.BytesIO(imageBytes))

    image = image.resize((28, 28))
    image = np.array(image)

    greyImage = image[:, :, 3]

    greyImage = greyImage.astype("float32") / 255

    return greyImage

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    print('data:', data)
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    imageData = data['image']
    print('image data:', imageData)
    greyscaleImage = processImage(imageData)
    print(f"Processed data: {greyscaleImage}")

    prediction = GetPrediction(greyscaleImage)
    label = np.argmax(prediction)

    print(prediction)
    print(label)
    
    return jsonify({'prediction': int(label)})

if __name__ == "__main__":
    app.run(debug=True)
