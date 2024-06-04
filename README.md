# Learning to write a neural network that recognizes hand-written digits!

This is a humble adventure to venture out into the world of machine-learning. I've heavily commented the code, mostly for my own understanding of the concepts, but feel free to run the [nn.py](./nn.py) file and check out how the program performs yourself! It goes through 3 epochs, or iterations, training itself on the MNIST dataset included in `/data`.

## How does it work?

In an overview, the neural network is made of an input layer, an output layer, and an ___x___ amount of hidden layers. This particular model uses just one hidden layer.
The the data is processed first through forward propagation, where the model makes a prediction, and the data is then normalised into a specific range, here using the Sigmoid function. The pipeline here looks like: input -> hidden -> output.

We then carry out an error/cost function that compares the output/prediction of the model with correct label, using the mean-squared error approach.

The cyclic motion of self-learning neural networks comes up when we introduce backward propagation, which resembles the model using this information to then adjust its weights and biases in order to make a more accurate prediction the next time. This particular pipeline looks like: output -> hidden -> input.

## Some final words

This is a good dive into the 'hello-world' for neural networks, and this project has definitely provided me with knowledge that'll definitely help me continue working on more interesting projects in the future!
