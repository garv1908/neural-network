import numpy as np
import pathlib

def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]

    # convert grayscale values to floating point values 0 <= x <= 1
    images = images.astype("float32") / 255
    
    # reshape the images array to a [ 7000 x 784 ] matrix
        # images.shape[0]: number of images (i.e. 7000)
        # images.shape[1], images.shape[2]: height, width (i.e. 28, 28)
        # images is originally imported as a [ 7000 x 28 x 28 ] 3D matrix
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
