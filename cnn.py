from data import get_mnist
import numpy as np
import pickle
import tensorflow as tf

print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

class ConvolutionalNN():
    def __init__(self):
        self.images, self.labels = get_mnist()
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS = 28, 28, 1
        self.images = np.reshape(self.images, (self.images.shape[0], self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS))

    def train(self, epochs):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Convolution2D(
            input_shape = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS),
            kernel_size = 5,
            filters = 8,
            strides = 1,
            activation = tf.keras.activations.relu,
            kernel_initializer = tf.keras.initializers.VarianceScaling()
        ))

        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size = (2, 2),
            strides = (2, 2),
        ))

        self.model.add(tf.keras.layers.Convolution2D(
            kernel_size = 5,
            filters = 16,
            strides = 1,
            activation = tf.keras.activations.relu,
            kernel_initializer = tf.keras.initializers.VarianceScaling()
        ))

        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size = (2, 2),
            strides = (2, 2),
        ))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(
            units = 128,
            activation = tf.keras.activations.relu
        ))

        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Dense(
            units = 10,
            activation = tf.keras.activations.softmax,
            kernel_initializer = tf.keras.initializers.VarianceScaling()
        ))

        self.model.summary()

        adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

        self.model.compile(
            optimizer = adam_optimizer,
            loss = tf.keras.losses.categorical_crossentropy,
            metrics = ['accuracy']
        )

        training_history = self.model.fit(
            self.images,
            self.labels,
            epochs=epochs
        )

        self.save_model()

    """
    this code directly makes a prediction from the loaded tf model
    """
    def get_prediction(self, img):
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)

        return prediction[0]

    """
    below uses cnn.js to get a prediction back from tensorflowjs
    wouldn't work on my website on vercel because of storage limitations so i moved on
    """
    # def get_prediction(self, img):
    #     import subprocess
    #     import json
    #     # call the node.js script
    #     process = subprocess.Popen(['node', './tfjs/cnn.js'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     input_data = json.dumps({'image': img.tolist()})
    #     stdout, stderr = process.communicate(input=input_data.encode())

    #     if stderr:
    #         print(f"Error: {stderr.decode()}")
    #         return None

    #     response = json.loads(stdout.decode())
    #     print("repsonse: {response}")
        
    #     if 'error' in response:
    #         print(f"Error: {response['error']}")
    #         return None
    #     return response['prediction'][0]
    
    def load_model(self, filename="./loaded_models/convolutional.pkl"):
        try:
            with open(filename, "rb") as file:
                self.model = pickle.load(file)
        except:
            print("Error loading the model.")
            print("Erasing...")
            open(filename, 'w').close()
            print("Proceeding to retrain model...")
            self.train(epochs=10)
            print("Retrained model.")
            return self
        
    def save_model(self, filename="./loaded_models/convolutional.pkl"):
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
