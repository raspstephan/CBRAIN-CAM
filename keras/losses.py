"""Define custom metrics

Author: Stephan Rasp
"""

import keras
import keras.backend as K


def rmse(y_true, y_pred):
    """This is the regular_loss

    Args:
        y_true:
        y_pred:

    Returns:

    """
    return K.sqrt(K.reduce_mean())