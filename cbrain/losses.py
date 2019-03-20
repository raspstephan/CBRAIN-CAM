"""Define custom metrics

Author: Stephan Rasp
"""

import tensorflow as tf
import tensorflow.math as tfm
import tensorflow.keras
# tgb - 2/11/2019 - For consistency with eager execution
#import keras
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse


# Define custom losses

# tgb - 2/11/2019 - Custom loss function = al*MSE+(1-al)*Conservation_residual
# Ideally would name it something else
# Keeping its original name here for convenience sake
def customLoss(input_tensor,fsub,fdiv,normq,hyai,hybi,alpha = 0.5):

        # tgb - 2/5/2019 - Loss function written above
    def lossFunction(y_true,y_pred):    
        loss = tfm.multiply(alpha, mse(y_true, y_pred))
        loss += tfm.multiply(tfm.subtract(1.0,alpha), \
                            massent_res(input_tensor,y_pred,fsub,fdiv,normq,hyai,hybi))
        return loss

    # tgb - 2/5/2019 - Mass and enthalpy residual function
    # Adapted from massent_check by converting numpy to tensorflow
    def massent_res(x,y,fsub,fdiv,normq,hyai,hybi):

        # 0) Constants
        G = 9.80616; # Reference gravity constant [m.s-2]
        L_F = 3.337e5; # Latent heat of fusion of water [W.kg-1]
        L_V = 2.501e6; # Latent heat of vaporization of water [W.kg-1]
        L_S = L_F+L_V; # Latent heat of sublimation of water [W.kg-1]
        P0 = 1e5; # Reference surface pressure [Pa]   

        # WATER&ENTHALPY) Get non-dimensional pressure differences (p_tilde above)
        # In the input vector, PS is the 151st element after 
        # the first elements = [QBP, ..., VBP with shape 30*5=150]
        PS = tfm.add( tfm.multiply( x[:,300], fdiv[300]), fsub[300])
        # Reference for calculation of d_pressure is cbrain/models.py (e.g. QLayer)
        P = tfm.add( tfm.multiply( P0, hyai), \
        tfm.multiply( PS[:,None], hybi))
        dP = tfm.subtract( P[:, 1:], P[:, :-1])
        # norm_output = dp_norm * L_V/G so dp_norm = norm_output * G/L_V
        dP_NORM = tfm.divide( \
        tfm.multiply(normq[:30], \
                  G),\
        L_V)
        # dp_tilde = dp/dp_norm
        dP_TILD = tfm.divide( dP, dP_NORM)

        # WATER.1) Calculate water vertical integral from level 1 to level 30
        WATVEC = tfm.multiply( dP_TILD, tfm.add(tfm.add(y[:, :30],\
                                                        y[:, 30:60]),\
                                                y[:, 60:90]))
        WATINT = tfm.reduce_sum( WATVEC, axis=1)

        # WATER.2) Calculate forcing on the right-hand side (Net Evaporation-Precipitation)
        # E-P is already normalized to units W.m-2 in the output vector
        # so all we need to do is input-unnormalize LHF that is taken from the input vector
        LHF = tfm.add( tfm.multiply( x[:,303], fdiv[303]), fsub[303])
        # Note that total precipitation = PRECT + 1e-3*PRECTEND in the CAM model
        # PRECTEND already multiplied by 1e-3 in output vector so no need to redo it
        PREC = tfm.add( y[:, 214], y[:, 215])

        # WATER.FINAL) Residual = E-P-DWATER/DT
        WATRES = tfm.add(tfm.add(LHF,\
                                 tfm.negative(PREC)),\
                         tfm.negative(WATINT))

        # ENTHALPY.1) Calculate net energy input from phase change and precipitation
        # PHAS = Lf/Lv*((PRECST+PRECSTEN)-(PRECT+PRECTEND))
        PHAS = tfm.divide( tfm.multiply( tfm.subtract(\
                                              tfm.add( y[:,216], y[:,217]),\
                                              tfm.add( y[:,214], y[:,215])),\
                                      L_F),\
                         L_V)

        # ENTHALPY.2) Calculate net energy input from radiation, sensible heat flux and turbulent KE
        # 2.1) RAD = FSNT-FSNS-FLNT+FLNS
        RAD = tfm.add(\
                      tfm.subtract( y[:,210], y[:,211]),\
                      tfm.subtract( y[:,213], y[:,212]))
        # 2.2) Unnormalize sensible heat flux
        SHF = tfm.add( tfm.multiply( x[:,302], fdiv[302]), fsub[302])
        # 2.3) Net turbulent kinetic energy dissipative heating is the column-integrated 
        # turbulent kinetic energy energy dissipative heating
        KEDVEC = tfm.multiply( dP_TILD, y[:, 180:210])
        KEDINT = tfm.reduce_sum( KEDVEC, axis=1)

        # ENTHALPY.3) Calculate tendency of normalized column water vapor due to phase change
        # 3.1) Column water vapor is the column integral of specific humidity
        PHQVEC = tfm.multiply( dP_TILD, y[:, :30])
        PHQINT = tfm.reduce_sum( PHQVEC, axis=1)
        # 3.2) Multiply by L_S/L_V to normalize (explanation above)
        SPDQINT = tfm.divide( tfm.multiply( tfm.subtract(\
                                                     PHQINT, LHF),\
                                        L_S),\
                           L_V)

        # ENTHALPY.4) Same operation for liquid water tendency but multiplied by L_F/L_V
        SPDQCINT = tfm.divide( tfm.multiply(\
                                      tfm.reduce_sum(\
                                             tfm.multiply( dP_TILD, y[:, 30:60]),\
                                             axis=1),\
                                      L_F),\
                         L_V)

        # ENTHALPY.5) Same operation for temperature tendency
        DTINT = tfm.reduce_sum( tfm.multiply( dP_TILD[:, :30], y[:, 90:120]), axis=1)

        # ENTHALPY.FINAL) Residual = SPDQ+SPDQC+DTINT-RAD-SHF-PHAS
        ENTRES = tfm.add(tfm.add(tfm.add(tfm.add(tfm.add(tfm.add(SPDQINT,\
                                                                 SPDQCINT),\
                                                         DTINT),\
                                                 tfm.negative(RAD)),\
                                         tfm.negative(SHF)),\
                                 tfm.negative(PHAS)),\
                         tfm.negative(KEDINT))
        # Return sum of water and enthalpy square residuals
        return tfm.add( tfm.square(WATRES), tfm.square(ENTRES))

    return lossFunction

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
               mse_var, mse_var(10), customLoss]
metrics = [rmse, log_loss, var_ratio, mse, var_loss]
