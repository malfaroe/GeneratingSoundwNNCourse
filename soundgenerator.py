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

def convert_spectrograms_to_audio(self, generated_spectrograms,min_max_values):
    """Converts the generated spectrograms
    to audio files:
    1. Reshapes each spec to 2 Dimension arrays [:,:,0] (the incoming input specs will have 3D)
    [:,:,0] : mantains the 1st and second dimension and drops the third
    2. Denormalise 
    3. Convert from log to amplitude
    4. Apply inverse transform (Griffin-Lim)
    5. Appends to list of audios"""

    signals = []
    for spectrogram, min_max_value in zip(generated_spectrograms, min_max_values):
        #Reshaping
        log_spectrogram = spectrogram[:,:,0]
        #Denormalise
        denorm_log_spec = self.min_max_normaliser.denormalise(norm_array = log_spectrogram,
                        original_min = min_max_value["min"], 
                        original_max =  min_max_value["max"])
        #Convert to amplitude
        spec = librosa.db_to_amplitude(denorm_log_spec)
        #Apply inverse transform (Griffin-Lim)
        signal = librosa.istft(spec, hop_length= self.hop_length)
        #Append
        signals.append(signal)

        
        



        

    
