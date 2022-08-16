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
        hop_length= self.hop_length)[:-1]
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
        norm_array = norm_array *(self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) /(self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array

        


class Saver:
    """Saves features and min, max values
    Atributes:
    Save_feature_dir:directory to store the features
    min_max_values_save_dir: to store the min-max values"""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        """Saves the arrays in a npy format
        using the np.save numpy method"""
        save_path = self._generate_save_path(file_path)
        #Guardamos el arreglo en formato npy usando np.save
        np.save(save_path, feature)
        return save_path
        

    def save_min_max_values(self, min_max_values):
        """Saves the min max values as pkl"""
        save_path = os.path.join(self.min_max_values_save_dir, 
                                "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def _generate_save_path(self, file_path):
        """Generates a path
        for feature arrays to be stored
         as .npy files"""
        #Genera un contenedor para crear el nombre del archivo en el save
        file_name = os.path.split(file_path)[1] # 1 es el tail
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path 


    @staticmethod
    def _save(data, save_path):
        """Method for saving usable
         in and outside any class"""
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

class PreprocessingPipeline:
    """Pipeline for preprocessing audio files
    following these steps:
    1. Load a file
    2. Pad the signal if necessary
    3. Extract the log spectrogram from signal
    4. Normalise spectrogram
    5. Save the normalised spectrogram
    
    Stores also the min and max original values of each 
    array/signal"""

    def __init__(self):
        """Instantiates all the classes we built"""
        self._loader = None
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._num_expected_samples = None

    #Voy a  definir _loader ocmo una propiedad
    @property
    def loader(self):
        return self._loader
    
    #Con un setter decorator voy a permitir
    #modificar los parametros del loader
    #Como por ejemplo el num_expected_samples

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)


    def process(self, audio_files_dir):
        """Loads each file from the directory and applies
        the preprocessing function _process_file
        that contains all the steps
        params:
        audio_files_dir: path of files directory"""

        for root, _, files in os.walk(audio_files_dir):#revisa todos 
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)


    def _process_file(self, file_path):
        """Method for processing 
        individual file
        1. Load
        2. Padd if necesssary
        3. Extract features(spectrogram)
        4. Normalise feature array
        5. Save the min and max values of the feature
        6. Saves the normalised feature array"""
        signal = self._loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())


    def _is_padding_necessary(self, signal):
        """Compares the len of the current signal
        with the len of the expected signal"""
        if len (signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        """Applies right_pad to
        the signal"""
        num_missing_samples = self._num_expected_samples - len (signal) 
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    
    def _store_min_max_value(self, save_path, min_val, max_val):
        """Stores the min and max values of the feature array
        in a dictionary"""
        self.min_max_values[save_path] = {
            "min":min_val,
            "max": max_val
        }



        


if __name__ == "__main__":
    
    ##testing
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74  # in seconds
    SAMPLE_RATE = 22050
    MONO = True
    
    #FILES_DIR= r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\AUDIO_DATA"
    FILES_DIR= r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\AUDIO_DATA"
    MIN_MAX_VALUES_SAVE_DIR = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\MIN_MAX_VALUES_SAVE_DIR"
    SPECTROGRAMS_DIR = r"C:\Users\malfaro\Desktop\mae_code\GeneratingSoundwNNCourse\SPECTROGRAM_SAVE_DIR"

    #Instantiation of classes
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normalizer = MinMaxNormaliser(0,1)
    saver = Saver(SPECTROGRAMS_DIR, MIN_MAX_VALUES_SAVE_DIR)
    #Instanciamos la preprocess pipeline y todos sus atributos
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normalizer
    preprocessing_pipeline.saver = saver
    
    preprocessing_pipeline.process(FILES_DIR)


print("Done.")