"""This unit contains utilities for preprocessing
 the MNIST dataset and 
instatiating the model for training
"""
import tensorflow
from tensorflow.keras.datasets import mnist
from VarAutoencoder import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100

def load_mnist():
    """This utility preprocesses the mnist dataset:
    1. Load the mnist dataset
    2. Converts the X_train and X_test sets to float and Normalizes the data /255
    3. Reshapes the data giving it the same dimensions of the encoder input data.
    In this case we need an extra dimension for the scale for the final shape (width, height, scale) """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,)) #we add another dimension
    x_test = x_test.reshape(x_test.shape + (1,)) #we add another dimension

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    """Here we instatiate the model, compile it and
    start training it"""
    autoencoder = VAE(
        input_shape= (28,28,1),
        conv_filters = (32,64,64,64),
        conv_kernels= (3,3,3,3),
        conv_strides= (1,2,2,1),
        latent_space_dim= 2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder







if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("autoencoder")
    autoencoder2 = VAE.load("autoencoder")
    autoencoder2.summary()
    print("Done.")


