"""
Custom losses.

Created on 2019-04-03-11-18
Author: Stephan Rasp, raspstephan@gmail.com
"""
from .layers import compute_dP_tilde
from .imports import *
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from .cam_constants import *

def mass_res(inp, pred, inp_div, inp_sub, norm_q, noadiab=False):
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
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q)

    # 2. Compute water integral
    WATINT = K.sum(dP_tilde *(pred[:, PHQ_idx] + pred[:, PHCLDLIQ_idx] + pred[:, PHCLDICE_idx]), axis=1)

    # 3. Compute latent heat flux and precipitation forcings
    LHFLX = inp[:, LHFLX_idx] * inp_div[LHFLX_idx] + inp_sub[LHFLX_idx]
    PREC = pred[:, PRECT_idx] + pred[:, PRECTEND_idx]

    # 4. Compute water mass residual
    WATRES = LHFLX - PREC - WATINT

    return K.square(WATRES)


def ent_res(inp, pred, inp_div, inp_sub, norm_q, noadiab=False):
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
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q)

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


class WeakLoss():
    def __init__(self, inp_tensor, inp_div, inp_sub, norm_q, alpha_mass=0.25, alpha_ent=0.25,
                 name='weak_loss', noadiab=False):
        self.inp_tensor, self.inp_div, self.inp_sub, self.norm_q, self.alpha_mass, self.alpha_ent = \
            inp_tensor, inp_div, inp_sub, norm_q, alpha_mass, alpha_ent
        self.alpha_mse = 1 - (alpha_mass + alpha_ent)
        self.__name__ = name
        self.noadiab = noadiab

    def __call__(self, y_true, y_pred):
        loss = self.alpha_mse * mse(y_true, y_pred)
        loss += self.alpha_mass * mass_res(self.inp_tensor, y_pred, self.inp_div, self.inp_sub,
                                           self.norm_q, self.noadiab)
        loss += self.alpha_ent * ent_res(self.inp_tensor, y_pred, self.inp_div, self.inp_sub,
                                         self.norm_q, self.noadiab)
        return loss


loss_dict = {'weak_loss': mse, 'mass_loss': mse, 'ent_loss': mse}

