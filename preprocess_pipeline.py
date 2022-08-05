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
    pass


class MinMaxNormaliser:
    pass


class Saver:
    pass


class PreprocessingPipeline:
    pass


print("Done.")