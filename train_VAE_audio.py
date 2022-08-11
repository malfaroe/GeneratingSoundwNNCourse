"""This unit contains utilities for  training
the audio digits dataset
"""
import tensorflow
from tensorflow import keras
from VarAutoencoder import VAE
import os
import numpy as np

#from preprocess_pipeline import SPECTROGRAMS_DIR

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 1

def load_fsdd(spectrogram_path):
    """This utility loads the already processed audio
    data in the form of spectrograms
    from the folder path """
    x_train = []

    for root, _, files in os.walk(spectrogram_path):#revisa todos 
            for file in files:
                file_path = os.path.join(root, file)
                spectrogram = np.load(file_path)
                x_train.append(spectrogram)
               
    x_train = np.array(x_train)
    print("First shape:", x_train.shape)
    #Spectrogram has the shape (nr_bins, nr_frames)
    #But VAE with convo layers need arrays of 3d
    #So we need to reshape the spectrogram array adding one dimension
    x_train = x_train[..., np.newaxis] #--->(nr examples(3000), n_bins (256), n_frames(64), 1(newaxis))
    print("Now the shape of train is:", x_train.shape)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    """Here we instatiate the model, compile it and
    start training it"""
    autoencoder = VAE(
        input_shape= (256,64,1),
        conv_filters = (512,256,128,64, 32),
        conv_kernels= (3,3,3,3,3),
        conv_strides= (2,2,2,2,(2,1)),
        latent_space_dim= 128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder



SPECTROGRAMS_DIR = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\SPECTROGRAM_SAVE_DIR"


if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_DIR)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model_audio")
    
    print("Done.")


