""" Building and assembly of a Conv2 Autoencoder Architecture
Sources: 
Autoencoder es una red neuronal que implementa un algoritmo
de reduccion dimensional (data compression) a traves de una arquitectura particular
Componentes: ENCODER--BottleNeck---DECODER
En ese codigo se implementa el block encoder (compresor)
https://ai.plainenglish.io/convolutional-autoencoders-cae-with-tensorflow-97e8d8859cbe
https://medium.com/@AnasBrital98/autoencoders-explained-da131e60e02a"""

"""Class autoencoder"""

from tensorflow.keras import Model 
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import MeanSquaredError
from keras.datasets import mnist
import numpy as np
import os
import pickle

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
        self.autoencoder = None  #modelo general autoencoder que abarcara toda la arquitectura (encoder +decoder)
        #self.model = None #modelo general autoencoder que abarcara toda la arquitectura (encoder +decoder)

        #Private atributes
        self._num_conv_layers = len(conv_filters) #siempre hay tantos filtros como conv layers
        self._shape_before_bottleneck = None
        self._model_input = None ##Modelo integrado final, por ahora en None

        #First we build and initialize the entire architecture with a method called build

        self._build() #builds encoder+decoder+model
        

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()


    def _build(self):
        """Builds and initialize 
        the entire architecture"""
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()


    

    ###ENCODER ARCHITECTURE
    def _build_encoder(self):
        """Builds and assemble the encoder
        components 
        params:
        encoder_input : vector, input data formated by Keras Input method
        conv_layers: creates all the conv layers with a method
        bottleneck: creates bottleneck with a method
        """
        encoder_input = self._add_encoder_input()
        self._model_input = encoder_input #sera el input del modelo integrado (***)
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
        #saves shape of data before bottleneck in a tuple
        self._shape_before_bottleneck = K.int_shape(x)[1:] 
        #Flatten the data
        x = Flatten()(x)
        #Now pass the flatten data through a dense layer
        x = Dense(self.latent_space_dim, name = "encoder_output")(x)
        return x

    #######
    #DECODER ARCHITECTURE
    
    def _build_decoder(self): 
        """On reverse order we will have:
        - a dense layer
        - reshape to a 3D array
        - Convolutional transposer layer
        - Final output of the network
        The final model will integrate the
        encoder and decoder using the Keras
        Model object: Model(input = encoder_output,
        output = decoder_output)
        """
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layers(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name = "decoder")


    def _add_decoder_input(self):
        """The bottleneck output is the decoder input"""
        return Input(shape = self.latent_space_dim, name = "decoder_input")

    
    def _add_dense_layers(self, decoder_input):
        """Create the decoder dense layer. We
        need to specify the nr of neurons on 
        the layer, which is equal to the shape of last 
        layer before the bottleneck, self._shape_before_bottleneck.
        The nr of neurons will be equal to the total elements of the shape:
        Ex: _shape_before_bottleneck = [1,2,4] then nr_neurons = 1x2x4 = 8
        Inorder to multiply the dims of an  array we use Numpy prod 
        """
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name = "decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        """From the dense layer we will received a flatten structure
        so we need to turn it back to a 3D array with the same shape of
        the encoder before we apply there the conv layer (shape before
        bottleneck). In order to
        do that we will use the Keras leyer named Reshape """

        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer


    def _add_conv_transpose_layers(self, x):
        """Adds conv transpose blocks
        Loops through all the cnv layers in
        reverse order and stops at the first layer
        (we will use that for another operation
        param:
        x: reshaped_layer"""
        #Iterate in reverse order leaving the first layer out...
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
         

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same", 
            name = f"decoder_conv_transpose_layer_{layer_num}")

        x = conv_transpose_layer(x)
        #Apply non-linearity
        x = ReLU(name = f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name = f"decoder_bn_{layer_num}")(x)
        return x
         

    def _add_decoder_output(self, x):
        """Final convolutional layer
        to be added to decoder but with
        sigmoid activation
        The shape of the output data here will be
        equal to that of the initial input
        Ex: [28,28,1]. If we want the third dimension to
        be 1, then filters must be set to 1
        Besides, the indexes of conv kernels and strides 
        to use are the
        first ones on each of those
        vectores, so the index  will be [0], porque estamos emulando
        la primera layer (input)"""
        conv_transpose_layer = Conv2DTranspose(
        filters = 1,
        kernel_size= self.conv_kernels[0],
        strides = self.conv_strides[0], 
        padding = "same", 
        name = f"encoder_conv_layer_{self._num_conv_layers}")
        ##Apply to the incoming graph of layers (x)
        x = conv_transpose_layer(x)
        #Sigmoid
        output_layer = Activation("sigmoid", name = "sigmoid_layer")(x)
        return output_layer

        

    #Building the autoencoder as the integrated structure
    
    def _build_autoencoder(self): #(***: modelo integrado)
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.autoencoder = Model(model_input, model_output, name = "autoencoder")


    #Additional methods for training the model...
    """We need to compile the model and then training it..."""
    def compile(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate = learning_rate)
        mse_loss = MeanSquaredError()
        self.autoencoder.compile(optimizer = optimizer, loss = mse_loss)

    def train(self, x_train, batch_size, epochs):
        """Como autoencoders trata de reconstruir un input
        se usa el mismo input como referencia para estimar el error
        . Es decir el y es igual al x en la instaciacion de fit"""
        self.autoencoder.fit(x_train, 
        x_train,
        batch_size = batch_size, epochs = epochs,
        shuffle = True)


    ##SAVING AND LOADING BACK UTILITIES
    def save(self, save_folder = "."):
        """Utility for saving the trained model:
        Once we have trained the model we need to save
        it along with all the information about its
        parameters (num kernels, strides,etc) and
        the respective weights"""
        self._create_if_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_if_doesnt_exist(self, folder):
        """Checks if the save folder exists and creates
        one in case it doesnt"""
        if not os.path.exists(folder):
            os.makedirs(folder)
    


    def _save_parameters(self, save_folder):
        """Saves all the model parameters in
        save_folder creating the file parameters.pkl"""
        parameters = [self.input_shape,
        self.conv_filters,
                self.conv_kernels, 
                self.conv_strides, 
                self.latent_space_dim]
            
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.autoencoder.save_weights(save_path)



    @classmethod #permite acceder auna funcion de la clase sin instanciarla
                #En vez de self usa cls
    def load(cls, save_folder = "."):
        """Loads the saved parameters and the weights
        the creates an auoencoder object using the parameters"""
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        #Ahora creamos una instancia de autoencoder pasandole los parametros
        autoencoder = Autoencoder(*parameters)
        #Cargamos los weights ahora
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, weights_path):
        self.autoencoder.load_weights(weights_path)




    

            
if __name__ == "__main__":
    autoencoder = Autoencoder(
    input_shape=(28, 28, 1),
    conv_filters=(32, 64, 64, 64),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(1, 2, 2, 1),
    latent_space_dim=2)
    autoencoder.summary()
            



        


    print("Done.")