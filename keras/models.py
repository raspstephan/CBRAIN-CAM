"""
The keras models are defined in this file.

Author: Stephan Rasp
"""

import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Input, Flatten, Concatenate, \
    BatchNormalization, LocallyConnected1D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from losses import *


def fc_model(feature_shape, target_shape, hidden_layers, lr, loss,
             activation='relu', batch_norm=False):
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
    # First hidden layer
    model = Sequential([
        Dense(hidden_layers[0], input_shape=(feature_shape,),
              activation=activation)
    ])
    if batch_norm:
        model.add(BatchNormalization())
    # All other hidden layers
    if len(hidden_layers) > 1:
        for h in hidden_layers[1:]:
            model.add(Dense(h, activation=activation))
            if batch_norm:
                model.add(BatchNormalization())
    # Output layer
    model.add(Dense(target_shape, activation='linear'))

    # Compile model
    model.compile(Adam(lr), loss=loss, metrics=metrics)
    return model


def conv_model(feature_shape_conv, feature_shape_1d, target_shape, feature_maps,
               hidden_layers, lr, loss, kernel_size=3, batch_norm=False,
               activation='relu'):
    """
    TODO


    """

    # Use the functional API
    # First, the convolutional part
    inp_conv = Input(shape=(feature_shape_conv[0],
                            feature_shape_conv[1],))
    # First convolutional layer
    x_conv = Conv1D(feature_maps[0], kernel_size, padding='same',
                    activation=activation)(inp_conv)
    if len(feature_maps) > 1:
        for fm in feature_maps[1:]:
            x_conv = Conv1D(fm, kernel_size, padding='same',
                            activation=activation)(x_conv)
    x_conv = Flatten()(x_conv)

    # Then the linear path
    inp_1d = Input(shape=(feature_shape_1d,))

    # Concatenate the two
    x = Concatenate()([x_conv, inp_1d])
    if batch_norm:
        x = BatchNormalization()(x)

    # Fully connected layers at the end
    for h in hidden_layers:
        x = Dense(h, activation=activation)(x)
        if batch_norm:
            x = BatchNormalization()(x)

    # Final linear layer
    x = Dense(target_shape, activation='linear')(x)

    # Define the model
    model = Model([inp_conv, inp_1d], x)

    # Compile
    model.compile(Adam(lr), loss=loss, metrics=metrics)
    return model


def conv_model_tile(feature_shape_conv, target_shape,
                    feature_maps, hidden_layers, lr, loss, kernel_size=3,
                    activation='relu', locally_connected=False):
    """
    TODO


    """

    # Use the functional API
    # First, the convolutional part
    inp_conv = Input(shape=(feature_shape_conv[0],
                            feature_shape_conv[1],))
    # First convolutional layer
    if locally_connected:
        x = LocallyConnected1D(
            feature_maps[0],
            kernel_size,
            padding='valid',
            activation=activation,
        )(inp_conv)
    else:
        x = Conv1D(
            feature_maps[0],
            kernel_size,
            padding='same',
            activation=activation,
        )(inp_conv)
    if len(feature_maps) > 1:
        for fm in feature_maps[1:]:
            if locally_connected:
                x = LocallyConnected1D(
                    fm,
                    kernel_size,
                    padding='valid',
                    activation=activation,
                )(x)
            else:
                x = Conv1D(
                    fm,
                    kernel_size,
                    padding='same',
                    activation=activation,
                )(x)
    x = Flatten()(x)

    # Fully connected layers at the end
    for h in hidden_layers:
        x = Dense(h, activation=activation)(x)

    # Final linear layer
    x = Dense(target_shape, activation='linear')(x)

    # Define the model
    model = Model(inp_conv, x)

    # Compile
    model.compile(Adam(lr), loss=loss, metrics=metrics)
    return model
