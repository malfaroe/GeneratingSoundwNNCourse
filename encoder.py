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
from tensorflow.keras.layers import Input,Conv2D, ReLU, BatchNormalization, Flatten, Dense
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
        self._shape_before_bottleneck = None

        #First we build and initialize the entire architecture with a method called build

        self._build() #builds encoder+decoder+model
        

    def summary(self):
        self.encoder.summary()


    def _build(self):
        """Builds and initialize 
        the entire architecture"""
        self._build_encoder()
        #self._build_decoder()
        #self._build_model()

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
        #Bottleneck recibe el output de conv_layers y entrega el output del encoder
        bottleneck = self._add_bottleneck(conv_layers)
        #Finalmente encoder es el modelo total que usa el metodo bottleneck para
        #entregar su resultado (data comprimida, coded data)
        #Bottleneck es el encoder output
        self.encoder = Model(encoder_input, bottleneck, name = "encoder") 


    def _add_encoder_input(self):
        """Creates an input layer using the keras
        layer called Input """
        return Input(shape = self.input_shape, name = "encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Creates all the convolutional layers of
        the encoder, moving through the network"""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Creates a convolutional block
        consisting of Conv2D + ReLU + 
        Batch Normamization
        Recordar que la dimension del kernel es
        la que le da el nombre a la red conv. En
            este caso el kernel es 2d"""

        layer_number = layer_index +  1 #because 1st layer is the input layer
        conv_layer = Conv2D(filters = self.conv_filters[layer_index],
        kernel_size= self.conv_kernels[layer_index],
        strides = self.conv_strides[layer_index], 
        padding = "same", name = f"encoder_conv_layer_{layer_number}")

        x = conv_layer(x)
        x = ReLU(name = f"encoder_relu_{layer_number}")(x) #(x) means "applied to x"
        x = BatchNormalization(name = f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Final stage: flattens the data and
        adds a Bottleneck, which is a Dense Layer
        Flatten method: flattens the multi-dimensional
            input tensors into a single dimension"""

        #Flatten the data
        x = Flatten()(x)
        #Now pass the flatten data through a dense layer
        x = Dense(self.latent_space_dim, name = "encoder_output")(x)
        return x

    
        
            
if __name__ == "__main__":
    autoencoder = Autoencoder(
    input_shape=(28, 28, 1),
    conv_filters=(32, 64, 64, 64),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(1, 2, 2, 1),latent_space_dim=2)
    autoencoder.summary()
            



        


    print("Done.")