

"""Unit for generating audio samples from random selected original samples
Steps:
1. Load the files  already preprocessed as spectrograms + respective min_max_values
2. Select samples randomly for testing
3. Generate the samples from sampled spectrograms
4. Convert to audio
4. Save audio signals
"""


import os
import pickle
import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from VarAutoencoder import VAE
from train_VAE_audio import SPECTROGRAMS_DIR




HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\Samples\original"
SAVE_DIR_GENERATED = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\Samples\generated" #se guardan los samples originales seleccionados random para generar
MIN_MAX_VALUES_SAVE_DIR = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\MIN_MAX_VALUES_SAVE_DIR" #se guardan los samples generados



def load_fsdd(spectrogram_path):
    """This utility loads the already processed audio
    data in the form of spectrograms
    from the folder path """
    x_train = []
    file_paths = []

    for root, _, files in os.walk(spectrogram_path):#revisa todos 
            for file in files:
                file_path = os.path.join(root, file)
                spectrogram = np.load(file_path) #(n_bins, n_frames)
                x_train.append(spectrogram)
                file_paths.append(file_path)
               
    x_train = np.array(x_train)
    print("First shape:", x_train.shape)
    #Spectrogram has the shape (nr_bins, nr_frames)
    #But VAE with convo layers need arrays of 3d
    #So we need to reshape the spectrogram array adding one dimension
    x_train = x_train[..., np.newaxis] #--->(nr examples(3000), n_bins (256), n_frames(64), 1(newaxis))
    print("Now the shape of train is:", x_train.shape)
    return x_train, file_paths




SAVED_MODEL_FOLDER = R"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\model_audio"



if __name__== "__main__":
    vae = VAE.load("model") #loads the model 
    sound_generator = SoundGenerator(vae, HOP_LENGTH)
    print("Loaded...")
