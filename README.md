# Learning to write a neural network that recognizes hand-written digits!

[Try here!](https://neural-network-demo.vercel.app)

This is a humble adventure to venture out into the world of machine-learning. I've heavily commented the code in fnn.py, mostly for my own understanding of the concepts, but feel free to run [the file](./fnn.py) and check out how the program performs yourself! It goes through epochs, or iterations, training itself on the MNIST dataset included in `/data`. The rest of the program in fnn.py then asks for a number between 0 to 59999, which is the size of the dataset, and then predicts what the number is based on its pre-adjusted weights and biases.

I've also developed a website using flask that allows you to draw a digit on a canvas! Once you click submit, the image is processed and then sent into the neural network to obtain a prediction. This currently does not perform at the greatest accuracy but I've planned to try my luck at also developing a convolutional neural network (CNN) to see how that would perform in place of a feed-forward neural network (FNN).

The model itself is stored in a pickle file that's been trained previously for easy access.

## How does it work?

### FNN

In an overview, the FNN is made of an input layer, an output layer, and an ___x___ amount of hidden layers. This particular model uses just one hidden layer.
The the data is processed first through forward propagation, where the model makes a prediction, and the data is then normalised into a specific range, here using the Sigmoid function. The pipeline here looks like: input -> hidden -> output.

We then carry out an error/cost function that compares the output/prediction of the model with correct label, using the mean-squared error approach.

The cyclic motion of self-learning neural networks comes up when we introduce backward propagation, which resembles the model using this information to then adjust its weights and biases in order to make a more accurate prediction the next time. This particular pipeline looks like: output -> hidden -> input.

## Some final words

This is a good dive into the 'hello-world' for neural networks, and this project has provided me with knowledge that'll definitely help me continue working on more interesting projects in the future!
