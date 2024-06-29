from data import get_mnist
import numpy as np
from layer import Layer
from scipy import signal
import matplotlib.pyplot as plt

class ConvolutionalNN(Layer):
    def __init__(self):
        pass

    def train(self, epochs):
        pass

    def get_prediction(self, img):
        pass


if __name__ == "__main__":
    
    cnn = ConvolutionalNN()
    cnn.train(epochs=3)

    # show the results
    while True:
        index = int(input("Enter a number (0-59999): "))
        
        img = cnn.images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        img.shape += (1, )
        o = cnn.get_prediction(img)

        plt.title(f"Is the number a {o.argmax()}? :)")
        plt.show()
