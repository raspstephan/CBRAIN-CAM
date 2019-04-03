"""
Define all different types of models.

Created on 2019-01-28-13-17
Author: Stephan Rasp, raspstephan@gmail.com
"""

from .imports import *
from .cam_constants import *
from tensorflow.keras.layers import *
from .layers import *


def act_layer(act):
    """Helper function to return regular and advanced activation layers"""
    act = Activation(act) if act in tf.keras.activations.__dict__.keys() \
        else tf.keras.layers.__dict__[act]()
    return act


def fc_model(input_shape, output_shape, hidden_layers, activation, conservation_layer=False,
             inp_sub=None, inp_div=None, norm_q=None):
    inp = Input(shape=(input_shape,))

    # First hidden layer
    x = Dense(hidden_layers[0])(inp)
    x = act_layer(activation)(x)

    # Remaining hidden layers
    for h in hidden_layers[1:]:
        x = Dense(h)(x)
        x = act_layer(activation)(x)

    if conservation_layer:
        x = SurRadLayer(inp_sub, inp_div, norm_q)([inp, x])
        x = MassConsLayer(inp_sub, inp_div, norm_q)([inp, x])
        out = EntConsLayer(inp_sub, inp_div, norm_q)([inp, x])

    else:
        out = Dense(output_shape)(x)

    return tf.keras.models.Model(inp, out)


