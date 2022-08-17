

"""Unit for generating audio samples from random selected original samples
Steps:
1. Load the files  already preprocessed as spectrograms + respective min_max_values
2. Select samples randomly for testing
3. Generate the samples from sampled spectrograms
4. Convert to audio (from soundgenerate´s generate method)
4. Save audio signals
"""


import os
import pickle
import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from VarAutoencoder import VAE
from train_VAE_audio import SPECTROGRAMS_DIR







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

#Select samples randomly for testing

def select_spectrograms(spectrograms, file_paths,
                        min_max_values, num_spectrograms = 2):
    """Take random samples from the spectrograms"""
    sampled_indexes = np.random.choice(len(spectrograms), num_spectrograms)
    sampled_spectrogams = spectrograms[sampled_indexes]

    #Corta File_paths a solo los samples seleccionados

    file_paths = [file_paths[index] for index in sampled_indexes]
    #print("Prueba de valor min_max:", min_max_values[12])
    #Extrae los min_max_values de los seleccionados
    
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]
    return sampled_spectrogams, sampled_min_max_values


#Saves the generted signals
def save_signals(signals, save_dir, sample_rate = 22050):
    """Signals are comming from the generate method,
    and using the Soundfile write we can save them
    in save_dir using a determined sample rate"""
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)




HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\Samples\original"
SAVE_DIR_GENERATED = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\Samples\generated" #se guardan los samples originales seleccionados random para generar
MIN_MAX_VALUES_FILE= r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\MIN_MAX_VALUES_SAVE_DIR\min_max_values.pkl" 
#MIN_MAX_VALUES_SAVE_DIR = r"/Users/mauricioalfaro/Documents/mae_code/GeneratingSoundwNNCourse/MIN_MAX_VALUES_SAVE_DIR/min_max_values.pkl"
#se guardan los samples generados

SAVED_MODEL_FOLDER = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\model_audio"
#SPECTROGRAM_DIR = r"/Users/mauricioalfaro/Documents/mae_code/GeneratingSoundwNNCourse/SPECTROGRAM_SAVE_DIR"
SPECTROGRAMS_DIR = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\SPECTROGRAM_SAVE_DIR"

if __name__== "__main__":
    """Steps:
1. Load the files  already preprocessed as spectrograms + respective min_max_values
2. Select samples randomly for testing
3. Generate the samples from sampled spectrograms and convert them into audio
4. Convert to audio (from soundgenerate´s generate method) the original samples
4. Save audio signals
"""
#Initialise sound generator
    vae = VAE.load("model") #loads the model 
    sound_generator = SoundGenerator(vae, HOP_LENGTH)
    # 1. Load the files  already preprocessed as spectrograms + respective min_max_values
    #Loads the min_max...
    with open(MIN_MAX_VALUES_FILE, "rb") as f:
        min_max_values = pickle.load(f)
    print("Files loaded...")
    #Loads the spectrograms
    specs, file_paths = load_fsdd(SPECTROGRAMS_DIR)
    print("Spectrograms and file paths loaded...")

    #2. Select samples randomly for testing
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,file_paths,min_max_values, 5)

    #3.Generate the samples from sampled spectrograms: returns converted audio files
    signals, _ = sound_generator.generate(sampled_specs,
                                         sampled_min_max_values)

    #4. Convert the original samples to audio

    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs,
                                                                     sampled_min_max_values)

    #5. Save original and generated audios 

    save_signals(original_signals, SAVE_DIR_ORIGINAL)
    save_signals(signals, SAVE_DIR_GENERATED)

    print("Satic fire successfully executed...")


