"""Define custom metrics

Author: Stephan Rasp
"""

import keras
import keras.backend as K
from keras.losses import mse


# Define custom losses
def rmse(y_true, y_pred):
    """Regular loss in tensorboard"""
    return K.sqrt(mse(y_true, y_pred))


def log_loss(y_true, y_pred):
    return K.log(rmse(y_true, y_pred) + 1e-20) / K.log(10.)


def total_error(y_true, y_pred):
    """
    I think this is simply the variance, but as a sum...
    """
    return K.sum(K.square(y_true - K.mean(y_true)))


def unexplained_error(y_true, y_pred):
    """
    I think this is simply the squared error, but the sum not the mean as in mse
    """
    return K.sum(K.square(y_true - y_pred))


def rsquared(y_true, y_pred):
    """
    1 - ratio(squared error over variance)
    """
    return (1. - (unexplained_error(y_true, y_pred) /
                  total_error(y_true, y_pred)))


def total_error_avgAx0(y_true, y_pred):
    """
    """
    avgY = K.mean(y_true, axis=0, keepdims=True)   # 0 is sample axis
    return K.sum(K.square(y_true - avgY))


def rsquared_avgAx0(y_true, y_pred):
    return (1. - (unexplained_error(y_true, y_pred) /
                  total_error_avgAx0(y_true, y_pred)))


def var_true(y_true, y_pred): return K.mean(K.var(y_true, axis=(0,1)))
def var_pred(y_true, y_pred): return K.mean(K.var(y_pred, axis=(0,1)))


def var_ratio(y_true, y_pred):
    vt = var_true(y_true, y_pred)
    vp = var_pred(y_true, y_pred)
    return vp / vt


def var_loss(y_true, y_pred):
    vt = var_true(y_true, y_pred)
    vp = var_pred(y_true, y_pred)
    return K.square(vp - vt)


def mse_var(ratio):
    """
    specify ratio to multiply var loss with
    """
    def loss(y_true, y_pred):
        return K.mean(mse(y_true, y_pred)) + ratio * var_loss(y_true, y_pred)
    return loss


# Define metrics list
all_metrics = [rmse, log_loss, total_error, unexplained_error, rsquared,
               total_error_avgAx0, rsquared_avgAx0, var_ratio, var_loss,
               mse_var, mse_var(10)]
metrics = [rmse, log_loss, var_ratio, mse, var_loss]
