"""Unit for generating audio digits using the VAE trained model"""

import librosa
from preprocess_pipeline import MinMaxNormaliser

class SoundGenerator:
    """Class responsible for generating
    the audios from the spectrograms:
    Take the spectrogram as input
    Applies the autoencoder (generates the encoding
    in the latent representation space)
    Applies the decoder to generate an 
    spectrogram
    Finally converts the spec into an audio"""

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self.min_max_normaliser = MinMaxNormaliser(0,1)

    def generate(self, spectrograms, min_max_values):
        #Generates encode and generate with decoder
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        #Convert to audio
        signal = self.convert_spectrograms_to_audio(generated_spectrograms,
        min_max_values)
        return signal, latent_representations

def convert_spectrograms_to_audio(self, min_max_values):
    pass


        

    
