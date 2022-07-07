"""First component of the Conv2 autoencoder ensamble
Sources: 
Autoencoder es una red neuronal que implementa un algoritmo
de reduccion dimensional (data compression) a traves de una arquitectura particular
Componentes: ENCODER--BottleNeck---DECODER
En ese codigo se implementa el block encoder (compresor)
https://ai.plainenglish.io/convolutional-autoencoders-cae-with-tensorflow-97e8d8859cbe
https://medium.com/@AnasBrital98/autoencoders-explained-da131e60e02a"""

"""Class autoencoder"""

from tensorflow.keras import Model 
from tensorflow.keras.layers import Input,Conv2D,     ReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K
from keras.datasets import mnist

class Autoencoder:
    """Class autoencoder represents a Convolutional autoencoder
    architecture with mirrored encoder and decoder components
    
    params:
    input_shape: tuple (nr_rows (width), nr_columns (height), nr_channels) 
                ex: with a grayscale image the nr of channels is 1
    conv_filters: tuple of nr of filters on each layer
    conv_kernels: tuple, nr of kernels for each layer. Ex:
    (3,5,2) means first layer with a 3x3 kernel, second layer with 5x5 etc
    conv_strides: nr strides for each layer
    latent_space_dim: dimension of the bottleneck space (compressed data)
    """

    def __init__(self, input_shape, 
                conv_filters,
                conv_kernels, 
                conv_strides, 
                latent_space_dim):
            
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        #Initialize the components
        self.encoder = None #atributo que sera un keras tf model
        self.decoder = None #atributo que sera un keras tf model
        self.model = None #modelo general autoencoder que abarcara toda la arquitectura (encoder +decoder)

        #Private atributes
        self._num_conv_layers = len(conv_filters) #siempre hay tantos filtros como conv layers


        #First we build and initialize the entire architecture with a method called build

        def _build(self):
            """Builds and initialize 
            the entire architecture"""
            self._build_encoder()
            self._build_decoder()
            self._build_model()

        def _build_encoder(self):
            """Builds and assemble the encoder
            components 
            params:
            encoder_input : vector, input data formated by Keras Input method
            conv_layers: creates all the conv layers with a method
            bottleneck: creates bottleneck with a method
            """
            encoder_input = self._add_encoder_input()
            conv_layers = self._add_conv_layers(encoder_input)
            bottleneck = self._add_bottleneck(conv_layers)
            self.encoder = Model(encoder_input, bottleneck, name = "encoder")
            


        self._build() #builds encoder+decoder+model


        pass