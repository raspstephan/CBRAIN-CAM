import numpy as np
import tensorflow as tf
import numpy as np
import random
from dataLoad import *
from utils import *

slim = tf.contrib.slim

#from tensorflow.contrib.keras.python import keras
#from tensorflow.python.platform import test
#from keras import backend as K


import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)
from keras.layers.local import LocallyConnected2D

from keras.models import Model, Sequential
from keras.losses import mean_absolute_percentage_error, mean_squared_logarithmic_error

from keras.optimizers import Adam

