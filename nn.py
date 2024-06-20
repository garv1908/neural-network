from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
class NeuralNetwork():
    def __init__(self):
        self.images, self.labels = get_mnist()

        # initial values for our weights and biases for processing,
        # adjusted later through multiple iterations
        self.w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
        self.w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
        self.b_i_h = np.zeros((20, 1))
        self.b_h_o = np.zeros((10, 1))

        # pre-set values
        self.learn_rate = 0.01
        self.nr_correct = 0
        self.epochs = 3 # number of iterations for each input/sample/image
    
    def train(self):
        for epoch in range(self.epochs):
            for img, l in zip(self.images, self.labels):
                # converting img and l from vectors to matrices
                    # img: 784 -> (784, 1)
                    # l: 10 -> (10, 1)
                img.shape += (1, )
                l.shape += (1, )

                # forward propagation: transform input values to output values
                    # input -> hidden
                    # take inputs (img) and weight matrix (connects both layers),
                    # multiply through matrix multiplication (@),
                    # add bias weights
                h_pre = self.b_i_h + self.w_i_h @ img

                    # normalize values into a specific range (0 <= x <= 1) by applying an activation function (the Sigmoid function)
                h = 1 / (1 + np.exp(-h_pre))

                    # hidden -> output
                    # repeat procedure to get output values
                o_pre = self.b_h_o + self.w_h_o @ h
                o = 1 / (1 + np.exp(-o_pre))

                # error/cost function (calculation)
                    # compare output with label, using mean-squared error
                e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)

                    # how many images are classified correctly
                self.nr_correct += int(np.argmax(o) == np.argmax(l))

                # backward propagation: propagate error from the end back to the start
                    # output -> hidden (error function derivative)
                    # normally requires derivatiave of the error function, but thanks to math,
                    # we don't really need it, and we can write:
                delta_o = o - l

                    # get update value for each weight connecting both layers
                self.w_h_o += -self.learn_rate * delta_o @ np.transpose(h)
                self.b_h_o += -self.learn_rate * delta_o
                    
                    # hidden -> input (activation function derivative)

                    # delta_h shows how each hidden neuron participated towards the error
                delta_h = np.transpose(self.w_h_o) @ delta_o * (h * (1 - h))

                    # calculate update values for weights connecting the input with the hidden layer
                self.w_i_h += -self.learn_rate * delta_h @ np.transpose(img)
                self.b_i_h += -self.learn_rate * delta_h

            # show accuracy for this epoch
            print(f"Accuracy: {round((self.nr_correct / self.images.shape[0]) * 100, 2)}%")
            self.nr_correct = 0

    def get_prediction(self, img):
        # use forward propagation to get output values
        # forward propagation input -> hidden
        h_pre = self.b_i_h + self.w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        
        # forward propagation hidden -> output
        o_pre = self.b_h_o + self.w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        return o


if __name__ == "__main__":

    nn = NeuralNetwork()
    nn.train()

    # show the results
    while True:
        index = int(input("Enter a number (0-59999): "))
        
        img = nn.images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        img.shape += (1, )
        o = nn.get_prediction(img)

        plt.title(f"Is the number a {o.argmax()}? :)")
        plt.show()