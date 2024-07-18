from data import get_mnist
import numpy as np
# import tensorflow as tf
import pickle
import subprocess
import json

# print('Tensorflow version:', tf.__version__)
# print('Keras version:', tf.keras.__version__)
class ConvolutionalNN():
    def __init__(self):
        self.images, self.labels = get_mnist()
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS = 28, 28, 1
        self.images = np.reshape(self.images, (self.images.shape[0], self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS))

    from tf_cnn_train_model import train

    """
    this commented out code would be directly making a prediction from the loaded tf model
    """
    # def get_prediction(self, img):
    #     img = np.expand_dims(img, axis=-1)
    #     img = np.expand_dims(img, axis=0)
    #     prediction = self.model.predict(img)

    #     return prediction[0]

    """
    below uses cnn.js to get a prediction back from tensorflowjs
    """
    def get_prediction(self, img):

        # Call the Node.js script
        process = subprocess.Popen(['node', 'cnn.js'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        input_data = json.dumps({'image': img.tolist()})
        stdout, stderr = process.communicate(input=input_data.encode())

        if stderr:
            print(f"Error: {stderr.decode()}")
            return None

        response = json.loads(stdout.decode())
        print(response)
        
        if 'error' in response:
            print(f"Error: {response['error']}")
            return None
        return response['prediction'][0]
    
    def load_model(self, filename="./tmp/loaded_models/convolutional.pkl"):
        try:
            with open(filename, "rb") as file:
                self.model = pickle.load(file)
        except:
            print("Error loading the model.")
            print("Erasing...")
            open(filename, 'w').close()
            self.train(epochs=1)
            print("Retrained model.")
            return self
        
    def save_model(self, filename="./tmp/loaded_models/convolutional.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
        print("model saved to:", filename)


if __name__ == "__main__":
    
    cnn = ConvolutionalNN()
    cnn.train(epochs=10)
    

    # show the results
    # while True:
    #     index = int(input("Enter a number (0-59999): "))
        
    #     img = cnn.images[index]
    #     plt.imshow(img.reshape(28, 28), cmap="Greys")

    #     img.shape += (1, )
    #     o = cnn.get_prediction(img)

    #     plt.title(f"Is the number a {o.argmax()}? :)")
    #     plt.show()
