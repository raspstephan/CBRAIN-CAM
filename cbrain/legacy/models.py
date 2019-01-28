"""
The keras models are defined in this file.

Author: Stephan Rasp
"""

from .imports import *
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from .losses import *
act_dict = keras.activations.__dict__
lyr_dict = keras.layers.__dict__

L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
P0 = 1e5


class PartialReLU(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a = x[:, :60]
        b = K.maximum(x[:, 60:120], 0)
        c = x[:, 120:]
        return K.concatenate([a, b, c], 1)

    def compute_output_shape(self, input_shape):
        return input_shape

class QLayer(Layer):
    def __init__(self, fsub, fdiv, hyai, hybi, **kwargs):
        super().__init__(**kwargs)
        self.fsub, self.fdiv, self.hyai, self.hybi = fsub, fdiv, np.array(hyai), np.array(hybi)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, arrs):
        L_V = 2.501e6 ; L_I = 3.337e5; L_S = L_V + L_I
        C_P = 1.00464e3; G = 9.80616; P0 = 1e5

        f, a = arrs
        # Get pressure difference 
        PS = f[:, 90] * self.fdiv[90] + self.fsub[90]
        P = P0 * self.hyai + PS[:, None] * self.hybi
        dP = P[:, 1:] - P[:, :-1]

        # Get Convective integrals
        iPHQ = a[:, 30:60]*dP/G/L_S
        vPHQ = K.sum(iPHQ, 1)
        absvPHQ = K.sum(K.abs(iPHQ),1)
        # Sum for convective moisture
        dQCONV = vPHQ

        # Get surface flux
        LHFLX = (f[:, 93] * self.fdiv[93] + self.fsub[93]) / L_V

        # Get precipitation sink
        TOT_PRECL = a[:, 64] / (24*3600*2e-2)

        # Total differences to be corrected --> factor. Correct everything involved
        #pdb.set_trace()
        dQ = dQCONV - LHFLX + TOT_PRECL
        absTOT = absvPHQ + K.abs(TOT_PRECL)
        # Correct PHQ
        fact = dQ[:, None] * K.abs(iPHQ) / absTOT[:, None]
        b = a[:, 30:60] - fact / dP*G*L_S
        # Correct precipitation sink
        fact = dQ[:] * K.abs(TOT_PRECL) / absTOT[:]
        c = a[:, 64] - fact * (24*3600*2e-2)

        return K.concatenate([a[:, :30], b, a[:, 60:64], c[:, None]], 1)

    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv), 
                  'hyai': list(self.hyai), 'hybi': list(self.hybi)}
        base_config = super(QLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

class ELayer(Layer):
    def __init__(self, fsub, fdiv, hyai, hybi, **kwargs):
        super().__init__(**kwargs)
        self.fsub, self.fdiv, self.hyai, self.hybi = fsub, fdiv, np.array(hyai), np.array(hybi)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, arrs):
        L_V = 2.501e6 ; L_I = 3.337e5; L_S = L_V + L_I
        C_P = 1.00464e3; G = 9.80616; P0 = 1e5

        f, a = arrs
        # Get pressure difference 
        PS = f[:, 90] * self.fdiv[90] + self.fsub[90]
        P = P0 * self.hyai + PS[:, None] * self.hybi
        dP = P[:, 1:] - P[:, :-1]

        # Get Convective integrals
        iTPHY, iPHQ = a[:, :30]*dP/G, a[:, 30:60]*dP/G/L_S*L_V
        vTPHY, vPHQ = K.sum(iTPHY, 1), K.sum(iPHQ, 1)
        absvTPHY, absvPHQ = K.sum(K.abs(iTPHY),1), K.sum(K.abs(iPHQ),1)

        # Get surface fluxes
        SHFLX = f[:, 92] * self.fdiv[92] + self.fsub[92]
        LHFLX = (f[:, 93] * self.fdiv[93] + self.fsub[93])

        # Radiative fluxes
        dERADFLX = K.sum(a[:, -5:-1], 1) * 1e3
        absRADFLX = K.sum(K.abs(a[:, -5:-1]), 1) * 1e3

        # Total differences to be corrected --> factor. Correct heating and 2D terms
        dE = vTPHY - SHFLX - dERADFLX + vPHQ - LHFLX
        absTOT = absvTPHY + absRADFLX
        # Correct TPHY
        fact = dE[:, None] * K.abs(iTPHY) / absTOT[:, None]
        b = a[:, :30] - fact / dP*G
        # Correct Radiative fluxes
        fact = dE[:, None] * K.abs(a[:, -5:-1]) * 1e3 / absTOT[:, None]
        c = a[:, -5:-1] + fact / 1e3

        return K.concatenate([b, a[:, 30:60], c, a[:, -1][:, None]], 1)

    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv), 
                  'hyai': list(self.hyai), 'hybi': list(self.hybi)}
        base_config = super(ELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def act_layer(act):
    """Helper function to return regular and advanced activation layers"""
    return Activation(act) if act in act_dict.keys() else lyr_dict[act]()


def fc_model(feature_shape, target_shape, hidden_layers, lr, loss,
             activation='relu', batch_norm=False, dr=None, l2=None, 
             partial_relu=False, eq=False, fsub=None, fdiv=None):
    """Creates a simple fully connected neural net and compiles it

    Args:
        feature_shape: Shape of input
        target_shape: Shape of output
        hidden_layers: list with hidden nodes
        lr: learning rate for Adam optimizer
        loss: loss function
        activation: Keras activation function
        batch_norm: Add batch normalization

    Returns:
        model: compiled Keras model
    """
    if l2 is not None:
        l2 = keras.regularizers.l2(l2)
    
    inp = Input(shape=(feature_shape,))
    # First hidden layer
    x = Dense(hidden_layers[0], kernel_regularizer=l2)(inp)
    x = act_layer(activation)(x)
    if batch_norm:
        x = BatchNormalization()(x)    
    # if dr is not None:
    #     x = Dropout(dr)(x)
    # All other hidden layers
    if len(hidden_layers) > 1:
        for h in hidden_layers[1:]:
            x = Dense(h, kernel_regularizer=l2)(x)
            x = act_layer(activation)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            if dr is not None:
                x = Dropout(dr)(x)
    # Output layer
    x = Dense(target_shape, activation='linear', kernel_regularizer=l2)(x)

    if partial_relu:
        x = PartialReLU()(x)
    if eq:
        with open(os.path.join(os.path.dirname(__file__), 'hyai_hybi.pkl'), 'rb') as f:
            hyai, hybi = pickle.load(f)
        x = QLayer(fsub, fdiv, hyai, hybi)([inp, x])
        x = ELayer(fsub, fdiv, hyai, hybi)([inp, x])
    
    # Compile model
    model = Model(inp, x)
    model.compile(Adam(lr), loss=loss, metrics=metrics)
    return model


def conv_model(feature_shape_conv, feature_shape_1d, target_shape, feature_maps,
               hidden_layers, lr, loss, kernel_size=3, stride=1, batch_norm=False,
               activation='relu', tile=False, locally_connected=False, padding='same',
               use_bias=True, dr=None):
    """
    Convolutional model
    """
    Conv = LocallyConnected1D if locally_connected else Conv1D

    # Use the functional API
    # First, the convolutional part
    inp_conv = Input(shape=(feature_shape_conv[0],
                            feature_shape_conv[1],))
    # First convolutional layer
    x_conv = Conv(
        feature_maps[0], kernel_size, strides=stride, padding=padding, use_bias=use_bias)(inp_conv)
    x_conv = act_layer(activation)(x_conv)
    if batch_norm: x_conv = BatchNormalization(axis=-1)(x_conv)

    if len(feature_maps) > 1:
        for fm in feature_maps[1:]:
            x_conv = Conv(
                fm, kernel_size, strides=stride, padding=padding, use_bias=use_bias)(x_conv)
            x_conv = act_layer(activation)(x_conv)
            if batch_norm: x_conv = BatchNormalization(axis=-1)(x_conv)
    x_conv = Flatten()(x_conv)

    if not tile:
        # Then the linear path
        inp_1d = Input(shape=(feature_shape_1d,))

        # Concatenate the two
        x = Concatenate()([x_conv, inp_1d])
        inps = [inp_conv, inp_1d]
    else:
        x = x_conv
        inps = inp_conv

    # Fully connected layers at the end
    for h in hidden_layers:
        x = Dense(h)(x)
        x = act_layer(activation)(x)
        if batch_norm: x = BatchNormalization()(x)
        if dr is not None: x = Dropout(dr)(x)

    # Final linear layer
    x = Dense(target_shape, activation='linear')(x)

    # Define the model
    model = Model(inps, x)

    # Compile
    model.compile(Adam(lr), loss=loss, metrics=metrics)
    return model

