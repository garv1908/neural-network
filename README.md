# Learning to write a neural network that recognizes hand-written digits! (FNN from scratch and CNN w/ TensorFlow)

[Try here!](https://neural-network-demo.vercel.app)

This is a humble adventure to venture out into the world of machine-learning. I've heavily commented the code in fnn.py, mostly for my own understanding of the concepts, but feel free to run [the file](./fnn.py) and check out how the program performs yourself! It goes through epochs, or iterations, training itself on the MNIST dataset included in `/data`. The rest of the program in fnn.py then asks for a number between 0 to 59999, which is the size of the dataset, and then predicts what the number is based on its pre-adjusted weights and biases.

I've also developed a website using flask (within [app.py](app.py)) that allows you to draw a digit on a canvas! Once you click submit, the image is processed and then sent into the neural network to obtain a prediction. I've developed two models, so you can experiment and see (very clearly), how a convolutional neural network (CNN) performs in place of a feed-forward neural network (FNN).

The models themselves are stored in a pickle file, and have been trained previously for easy access.

## How does it work?

### FNN

In an overview, the FNN is made of an input layer, an output layer, and an ___x___ amount of hidden layers. This particular model uses just one hidden layer.
The the data is processed first through forward propagation, where the model makes a prediction, and the data is then normalised into a specific range, here using the Sigmoid function. The pipeline here looks like: input -> hidden -> output.

We then carry out an error/cost function that compares the output/prediction of the model with correct label, using the mean-squared error approach.

The cyclic motion of self-learning neural networks comes up when we introduce backward propagation, which resembles the model using this information to then adjust its weights and biases in order to make a more accurate prediction the next time. This particular pipeline looks like: output -> hidden -> input.

### CNN

A little more complex to understand, so I won't be explaining it in detail here, but there are a lot of fun layers that a bunch of libraries let you experiment with! I used tensorflow and played around with a sequential model with an adam optimizer and used a small dropout rate to prevent overfitting (something that the other model completely lacked).

## Some final words

This is a good dive into the 'hello-world' for neural networks, and this project has provided me with knowledge that'll definitely help me continue working on more interesting projects in the future!
