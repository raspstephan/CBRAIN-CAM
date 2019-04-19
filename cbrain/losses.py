"""
Custom losses.
tgb - 4/8/2019 - Merged Stephan's re-written conserving layers

"""
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow.keras
# tgb - 2/11/2019 - For consistency with eager execution
#import keras
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from .layers import compute_dP_tilde
from .imports import *
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from .cam_constants import *

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

def mass_res(inp, pred, inp_div, inp_sub, norm_q, hyai, hybi, noadiab=False):
    # Input
    PS_idx = 150 if noadiab else 300
    LHFLX_idx = 153 if noadiab else 303

    # Output
    PHQ_idx = slice(0, 30)
    PHCLDLIQ_idx = slice(30, 60)
    PHCLDICE_idx = slice(60, 90)
    PRECT_idx = 214
    PRECTEND_idx = 215

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute water integral
    WATINT = K.sum(dP_tilde *(pred[:, PHQ_idx] + pred[:, PHCLDLIQ_idx] + pred[:, PHCLDICE_idx]), axis=1)

    # 3. Compute latent heat flux and precipitation forcings
    LHFLX = inp[:, LHFLX_idx] * inp_div[LHFLX_idx] + inp_sub[LHFLX_idx]
    PREC = pred[:, PRECT_idx] + pred[:, PRECTEND_idx]

    # 4. Compute water mass residual
    WATRES = LHFLX - PREC - WATINT

    return K.square(WATRES)


def ent_res(inp, pred, inp_div, inp_sub, norm_q, hyai, hybi, noadiab=False):
    # Input
    PS_idx = 150 if noadiab else 300
    SHFLX_idx = 152 if noadiab else 302
    LHFLX_idx = 153 if noadiab else 303

    # Output
    PHQ_idx = slice(0, 30)
    PHCLDLIQ_idx = slice(30, 60)
    PHCLDICE_idx = slice(60, 90)
    TPHYSTND_idx = slice(90, 120)
    DTVKE_idx = slice(180, 210)
    FSNT_idx = 210
    FSNS_idx = 211
    FLNT_idx = 212
    FLNS_idx = 213
    PRECT_idx = 214
    PRECTEND_idx = 215
    PRECST_idx = 216
    PRECSTEND_idx = 217

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute net energy input from phase change and precipitation
    PHAS = L_I / L_V * (
            (pred[:, PRECST_idx] + pred[:, PRECSTEND_idx]) -
            (pred[:, PRECT_idx] + pred[:, PRECTEND_idx])
    )

    # 3. Compute net energy input from radiation, SHFLX and TKE
    RAD = (pred[:, FSNT_idx] - pred[:, FSNS_idx] -
           pred[:, FLNT_idx] + pred[:, FLNS_idx])
    SHFLX = (inp[:, SHFLX_idx] * inp_div[SHFLX_idx] +
             inp_sub[SHFLX_idx])
    KEDINT = K.sum(dP_tilde * pred[:, DTVKE_idx], 1)

    # 4. Compute tendency of vapor due to phase change
    LHFLX = (inp[:, LHFLX_idx] * inp_div[LHFLX_idx] +
             inp_sub[LHFLX_idx])
    VAPINT = K.sum(dP_tilde * pred[:, PHQ_idx], 1)
    SPDQINT = (VAPINT - LHFLX) * L_S / L_V

    # 5. Same for cloud liquid water tendency
    SPDQCINT = K.sum(dP_tilde * pred[:, PHCLDLIQ_idx], 1) * L_I / L_V

    # 6. And the same for T but remember residual is still missing
    DTINT = K.sum(dP_tilde * pred[:, TPHYSTND_idx], 1)

    # 7. Compute enthalpy residual
    ENTRES = SPDQINT + SPDQCINT + DTINT - RAD - SHFLX - PHAS - KEDINT

    return K.square(ENTRES)

# tgb - 4/18/2019 - Add radiation loss
def lw_res(inp, pred, inp_div, inp_sub, norm_q, hyai, hybi, noadiab=False):
    # Input
    PS_idx = 150 if noadiab else 300
    
    # Output
    QRL_idx = slice(120, 150)
    FLNS_idx = 213
    FLNT_idx = 212

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute longwave integral
    LWINT = K.sum(dP_tilde *pred[:, QRL_idx], axis=1)

    # 3. Compute net longwave flux from lw fluxes at top and bottom
    LWNET = pred[:, FLNS_idx] - pred[:, FLNT_idx]

    # 4. Compute water mass residual
    LWRES = LWINT-LWNET

    return K.square(LWRES)

def sw_res(inp, pred, inp_div, inp_sub, norm_q, hyai, hybi, noadiab=False):
    # Input
    PS_idx = 150 if noadiab else 300
    
    # Output
    QRS_idx = slice(150, 180)
    FSNS_idx = 211
    FSNT_idx = 210

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute longwave integral
    SWINT = K.sum(dP_tilde *pred[:, QRS_idx], axis=1)

    # 3. Compute net longwave flux from lw fluxes at top and bottom
    SWNET = pred[:, FSNT_idx] - pred[:, FSNS_idx]

    # 4. Compute water mass residual
    SWRES = SWINT-SWNET

    return K.square(SWRES)


# tgb - 4/18/2019 - Add radiation loss
class WeakLoss():
    def __init__(self, inp_tensor, inp_div, inp_sub, norm_q, hyai, hybi,
                 alpha_mass=0.125, alpha_ent=0.125,
                 alpha_lw=0.125, alpha_sw=0.125,
                 name='weak_loss', noadiab=False):
        self.inp_tensor, self.inp_div, self.inp_sub, self.norm_q, self.hyai, self.hybi,\
        self.alpha_mass, self.alpha_ent, self.alpha_lw, self.alpha_sw = \
            inp_tensor, inp_div, inp_sub, norm_q, hyai, hybi,\
        alpha_mass, alpha_ent, alpha_lw, alpha_sw
        self.alpha_mse = 1 - (alpha_mass + alpha_ent + alpha_sw + alpha_lw)
        self.__name__ = name
        self.noadiab = noadiab

    def __call__(self, y_true, y_pred):
        loss = self.alpha_mse * mse(y_true, y_pred)
        loss += self.alpha_mass * mass_res(self.inp_tensor, y_pred, self.inp_div, self.inp_sub,
                                           self.norm_q, self.hyai, self.hybi, self.noadiab)
        loss += self.alpha_ent * ent_res(self.inp_tensor, y_pred, self.inp_div, self.inp_sub,
                                         self.norm_q, self.hyai, self.hybi, self.noadiab)
        loss += self.alpha_lw * lw_res(self.inp_tensor, y_pred, self.inp_div, self.inp_sub,
                                           self.norm_q, self.hyai, self.hybi, self.noadiab)
        loss += self.alpha_sw * sw_res(self.inp_tensor, y_pred, self.inp_div, self.inp_sub,
                                         self.norm_q, self.hyai, self.hybi, self.noadiab)
        return loss


# Define metric list and loss dictionary 
loss_dict = {'weak_loss': mse, 'mass_loss': mse, 'ent_loss': mse,\
             'lw_loss': mse, 'sw_loss':mse}
all_metrics = [rmse, log_loss, total_error, unexplained_error, rsquared,
               total_error_avgAx0, rsquared_avgAx0, var_ratio, var_loss,
               mse_var, mse_var(10)]
metrics = [rmse, log_loss, var_ratio, mse, var_loss]
