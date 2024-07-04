from data import get_mnist
import numpy as np
import tensorflow as tf
import pickle

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


    def get_prediction(self, img):
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)

        from icecream import ic
        ic(prediction)
        
        return prediction[0]
    
    def load_model(self, filename="./tmp/loaded_models/convolutional.pkl"):
        try:
            with open(filename, "rb") as file:
                self.model = pickle.load(file)
        except:
            print("Error loading the model. Retraining...")
            open(filename, 'w').close()
            self.train(epochs=1)
            return self
        
    def save_model(self, filename="./tmp/loaded_models/convolutional.pkl"):
        # model_filename = "./tmp/loaded_models/convolutional.pkl"
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
        print("model saved to:", filename)


if __name__ == "__main__":
    
    cnn = ConvolutionalNN()
    cnn.train(epochs=1)
    

    # show the results
    # while True:
    #     index = int(input("Enter a number (0-59999): "))
        
    #     img = cnn.images[index]
    #     plt.imshow(img.reshape(28, 28), cmap="Greys")

    #     img.shape += (1, )
    #     o = cnn.get_prediction(img)

    #     plt.title(f"Is the number a {o.argmax()}? :)")
    #     plt.show()
