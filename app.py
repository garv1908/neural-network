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
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def process_image(image_data):

    print("in process_image")
    image_data = image_data.split(',')[1]
    print("image_data", image_data)
    image_bytes = base64.b64decode(image_data)
    print("image_bytes", image_bytes)

    image = Image.open(io.BytesIO(image_bytes))
    print("image open")
    image.show()
    image = image.resize((28, 28))
    print("image", image)
    image = np.array(image)

    print('shape:', image.shape[:])
    print(image[:, :, 0])
    print(image[:, :, 1])
    print(image[:, :, 2])
    print(image[:, :, 3])
    greyImage = image[:, :, 3]
    plt.imshow(image, cmap="Greys")
    plt.show()
    print(image)
    print(greyImage)
    print('shape:', greyImage.shape[:])
    plt.imshow(greyImage, cmap="Greys")
    plt.show()
    print("np arr greyImage", greyImage)
    greyImage = greyImage.astype("float32") / 255
    print("greyImage as type normalized", greyImage)
    greyImage = np.reshape(greyImage, (greyImage.shape[0] * greyImage.shape[1]))
    print("greyImage as type normalized", greyImage)
    print('shape after:', greyImage.shape[:])

    return greyImage

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print('data:', data)
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image']
    print('image data:', image_data)
    processed_data = process_image(image_data)
    print(f"Processed data: {processed_data[:]}")

    prediction = GetPrediction(processed_data)
    predicted_label = np.argmax(prediction)
    print(f"Prediction: {prediction}, Predicted label: {predicted_label}")
    
    return jsonify({'prediction': int(predicted_label)})

if __name__ == "__main__":
    app.run(debug=True)
