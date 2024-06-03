from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""

images, labels = get_mnist()

# initial values for our weights and biases for processing,
# adjusted later through multiple iterations
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (20, 784))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

# pre-set values
learn_rate = 0.01
nr_correct = 0
epochs = 3 # number of iterations for each input/sample/image

for epoch in range(epochs):
    for img, l in zip(images, labels):
        # converting img and l from vectors to matrices
            # img: 784 -> (784, 1)
            # l: 10 -> (10, 1)
        img.shape += (1, )
        l.shape += (1, )
