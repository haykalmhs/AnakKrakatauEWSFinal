import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def create_autoencoder(input_shape=(128, 32, 1), seed=46):
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    # Input layer
    inp = Input(shape=input_shape)

    # Encoder layers
    e = Conv2D(8, (7, 5), strides=[2,2], activation='elu', kernel_initializer=initializer, padding='same')(inp)
    e = Conv2D(16, (5, 3), strides=[2,2], activation='elu', kernel_initializer=initializer, padding='same')(e)
    e = Conv2D(32, (5, 3), strides=[2,2], activation='elu', kernel_initializer=initializer, padding='same')(e)
    e = Conv2D(64, (5, 3), strides=[2,2], activation='elu', kernel_initializer=initializer, padding='same')(e)

    # Intermediate layers
    shape_before_flattening = K.int_shape(e)
    encoded1 = Flatten()(e)
    encoded2 = Dense(24, activation="elu")(encoded1)
    fc = Dense(np.prod(shape_before_flattening[1:]), activation="elu")(encoded2)

    # Decoder layers
    d = Reshape(shape_before_flattening[1:])(fc)
    d = Conv2DTranspose(32, (5, 3), strides=[2,2], activation='elu', kernel_initializer=initializer, padding='same')(d)
    d = Conv2DTranspose(16, (5, 3), strides=[2,2], activation='elu', kernel_initializer=initializer, padding='same')(d)
    d = Conv2DTranspose(8, (5, 3), strides=[2,2], activation='elu', kernel_initializer=initializer, padding='same')(d)
    decoded = Conv2DTranspose(1, (7, 5), strides=[2,2], activation='linear', kernel_initializer=initializer, padding='same')(d)

    # Autoencoder model
    autoencoder = Model(inputs=inp, outputs=decoded, name="autoencoder")
    encoder = Model(inputs=inp, outputs=encoded2, name="encoder")

    return autoencoder, encoder
