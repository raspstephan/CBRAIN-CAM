"""
The keras models are defined in this file.

Author: Stephan Rasp
"""

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from losses import *


def fc_model(feature_shape, target_shape, hidden_layers, lr, loss):
    """Creates a simple fully connected neural net and compiles it

    Args:
        feature_shape: Shape of input
        target_shape: Shape of output
        hidden_layers: list with hidden nodes
        lr: learning rate for Adam optimizer
        loss: loss function

    Returns:

    """
    # First hidden layer
    model = Sequential([
        Dense(hidden_layers[0], input_shape=(feature_shape,), activation='relu')
    ])
    # All other hidden layers
    if len(hidden_layers) > 1:
        for h in hidden_layers[1:]:
            model.add(Dense(h, activation='relu'))
    # Output layer
    model.add(Dense(target_shape, activation='linear'))

    # Compile model
    model.compile(Adam(lr), loss=loss, metrics=metrics)
    return model

