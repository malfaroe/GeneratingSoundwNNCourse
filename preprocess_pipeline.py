"""Pipeline for preprocessing audio signals
for sound generation purposes
Stages:

1. Load a file
2. Pad the signal if necessary
3. Extract the log spectrogram from signal
4. Normalise spectrogram
5. Save the normalised spectrogram
"""
import os
import librosa
import pickle
import numpy as np



class Loader:
    """Loads the audiofile
    Params:
    sample_rate
    duration
    mono: Boolean, indicating if the audio file is 
    in mono format. Otherwise (False) is stereo"""
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
        sr = self.sample_rate,
        duration = self.duration,
        mono = self.mono)[0]
        return signal



class Padder:
    """Padding is the process of filling in with
    some number (ex: zero) the beginning and end of an array
    in order to give it a determined size"""

    def __init__(self, mode = "constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, 
                            (num_missing_items,0),
                             mode = self.mode)
        return padded_array


    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, 
                                (0, num_missing_items),
                                mode = self.mode)
        return padded_array



   

class LogSpectrogramExtractor:
    """Extracts the log spectrogram (in dB) from
    the audiofile signal using the librosa
    amplitude_to_dB method"""

    def __init__(self, frame_size, hop_length): 
        self.frame_size = frame_size
        self.hop_length = hop_length

    #Use the stft for extracting the signal

    def extract(self, signal):
        stft = librosa.stft(signal, 
        n_fft= self.frame_size,
        hop_length= self.hop_length)[0]
        spectrogram = np.abs(stft)
        #Para poder generar un spec que se vea correctamente hago log
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

    

class MinMaxNormaliser:
    """MinMax Normalisation
    so the minimum and maximum
    get squeashed into some predefined
    min_val and max_val"""
    def __init__(self, min_val, max_val):
        """Predefines the maximum and minimun
        after the normalization"""
        self.min =  min_val
        self.max =  max_val

    def normalise(self, array):
        #With this we obtain a [0,1] min max
        norm_array = (array - array.min()) /(array.max() - array.min())
        #One more step for using any predefined min/max
        norm_array = norm_array *(self.max - self.min) - self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) /(self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array

        


class Saver:
    pass


class PreprocessingPipeline:
    pass



##testing
FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 0.74  # in seconds
SAMPLE_RATE = 22050
MONO = True

if __name__ == "__main__":
    arr_m = np.array([1,2,3])
    song_path = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\spoken_data\7_theo_47.wav"
    padder = Padder()
    print("Array with left pad:", padder.left_pad(array = arr_m, num_missing_items = 5))
    print(padder.left_pad(array = arr_m, num_missing_items = 5).shape)
    Loader = Loader(SAMPLE_RATE, DURATION, MONO)
    signal = Loader.load(song_path)
    print(signal.shape)
    ls = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    print(ls.extract(signal).shape)
    Normalizer = MinMaxNormaliser(-1,1)
    print("Normalised array:", Normalizer.normalise(arr_m))
print("Done.")