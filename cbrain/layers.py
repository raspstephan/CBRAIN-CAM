"""
Define custom Keras layers

Created on 2019-04-03-10-56
Author: Stephan Rasp, raspstephan@gmail.com
"""

from .imports import *
from .cam_constants import *
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


# Helpers
def compute_dP_tilde(PS, PS_div, PS_sub, norm_q):
    """
    Computes dP_tilde in Tom's notation.
    PS is the normalized pressure as it is used in the input.
    PS_mult and PS_add are the corresponding values to unnormalize PS.
    WARNING: Really not sure about norm_q
    """
    PS = PS * PS_div + PS_sub
    P = P0 * hyai + PS[:, None] * hybi
    dP = P[:, 1:] - P[:, :-1]
    dP_norm = norm_q * G / L_V   # Why L_V?
    dP_tilde = dP / dP_norm
    return dP_tilde


# Layers
class SurRadLayer(Layer):
    def __init__(self, inp_sub, inp_div, norm_q, hyai=hyai, hybi=hybi, **kwargs):
        """
        Call using ([input, output])
        Assumes
        prior: [PHQ_nores, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FLNT, PRECT, PRECTEND, PRECST, PRECSTEN]
        Returns
        post(erior): [PHQ_nores, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        """
        self.inp_sub, self.inp_div, self.norm_q, self.hyai, self.hybi = \
            np.array(inp_sub), np.array(inp_div), np.array(norm_q), np.array(hyai), np.array(hybi)
        # Define variable indices here
        # Input
        self.PS_idx = 300
        # Output
        self.QRL_idx = slice(118, 148)  # Odd numbers because residuals
        self.QRS_idx = slice(148, 178)  # for Q and T are still missing
        self.FSNT_idx = 208
        self.FLNT_idx = 209
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_sub': list(self.inp_sub), 'inp_div': list(self.inp_div),
                  'norm_q': list(self.norm_q), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, arrs):
        inp, prior = arrs

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(
            inp[:, self.PS_idx],
            self.inp_div[self.PS_idx], self.inp_sub[self.PS_idx],
            self.norm_q
        )

        # 2. Compute radiative integrals
        SWINT = K.sum(dP_tilde * prior[:, self.QRS_idx], 1)
        LWINT = K.sum(dP_tilde * prior[:, self.QRL_idx], 1)

        # 3. Infer surface fluxes from residual
        FSNS = prior[:, self.FSNT_idx] - SWINT
        FLNS = prior[:, self.FLNT_idx] + LWINT

        # 4. Concatenate output vector
        post = tf.concat([
            prior[:, :self.FLNT_idx], FSNS[:, None],
            prior[:, self.FLNT_idx][:, None], FLNS[:, None],
            prior[:, (self.FLNT_idx + 1):]
        ], axis=1)
        return post

    def compute_output_shape(self, input_shape):
        """Input shape + 2"""
        return (input_shape[0][0], input_shape[0][1] + 2)


class MassConsLayer(Layer):
    def __init__(self, inp_sub, inp_div, norm_q, hyai=hyai, hybi=hybi, **kwargs):
        """
        Call using ([input, output])
        Assumes
        prior: [PHQ_nores, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        Returns
        post(erior): [PHQ, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        """
        self.inp_sub, self.inp_div, self.norm_q, self.hyai, self.hybi = \
            np.array(inp_sub), np.array(inp_div), np.array(norm_q), np.array(hyai), np.array(hybi)
        # Define variable indices here
        # Input
        self.PS_idx = 300
        self.LHFLX_idx = 303
        # Output
        self.PHQ_idx = slice(0, 29)  # Residual still missing
        self.PHCLDLIQ_idx = slice(29, 59)
        self.PHCLDICE_idx = slice(59, 89)
        self.PRECT_idx = 212
        self.PRECTEND_idx = 213

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_sub': list(self.inp_sub), 'inp_div': list(self.inp_div),
                  'norm_q': list(self.norm_q), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, arrs):
        inp, prior = arrs

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(
            inp[:, self.PS_idx],
            self.inp_div[self.PS_idx], self.inp_sub[self.PS_idx],
            self.norm_q
        )

        # 2. Compute vertical cloud water integral
        CLDINT = K.sum(dP_tilde *
                       (prior[:, self.PHCLDLIQ_idx] + prior[:, self.PHCLDICE_idx]),
                       axis=1)

        # 3. Compute water vapor integral
        VAPINT = K.sum(dP_tilde[:, :29] * prior[:, self.PHQ_idx], 1)

        # 4. Compute forcing (see Tom's note for details, I am just copying)
        LHFLX = (inp[:, self.LHFLX_idx] * self.inp_div[self.LHFLX_idx] +
                 self.inp_sub[self.LHFLX_idx])
        PREC = prior[:, self.PRECT_idx] + prior[:, self.PRECTEND_idx]

        # 5. Compute water vapor tendency at level 30 as residual
        PHQ30 = (LHFLX - PREC - CLDINT - VAPINT) / dP_tilde[:, 29]

        # 6. Concatenate output vector
        post = tf.concat([
            prior[:, self.PHQ_idx], PHQ30[:, None],
            prior[:, 29:]
        ], axis=1)
        return post

    def compute_output_shape(self, input_shape):
        """Input shape + 1"""
        return (input_shape[0][0], input_shape[0][1] + 1)


class EntConsLayer(Layer):
    def __init__(self, inp_sub, inp_div, norm_q, hyai=hyai, hybi=hybi, **kwargs):
        """
        Call using ([input, output])
        Assumes
        prior: [PHQ, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        Returns
        post(erior): [PHQ, PHCLDLIQ, PHCLDICE, TPHYSTND,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        """
        self.inp_sub, self.inp_div, self.norm_q, self.hyai, self.hybi = \
            np.array(inp_sub), np.array(inp_div), np.array(norm_q), np.array(hyai), np.array(hybi)
        # Define variable indices here
        # Input
        self.PS_idx = 300
        self.SHFLX_idx = 302
        self.LHFLX_idx = 303

        # Output
        self.PHQ_idx = slice(0, 30)
        self.PHCLDLIQ_idx = slice(30, 60)
        self.TPHYSTND_idx = slice(90, 119)  # Residual still missing
        self.DTVKE_idx = slice(179, 209)
        self.FSNT_idx = 209
        self.FSNS_idx = 210
        self.FLNT_idx = 211
        self.FLNS_idx = 212
        self.PRECT_idx = 213
        self.PRECTEND_idx = 214
        self.PRECST_idx = 215
        self.PRECSTEND_idx = 216

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_sub': list(self.inp_sub), 'inp_div': list(self.inp_div),
                  'norm_q': list(self.norm_q), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, arrs):
        inp, prior = arrs

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(
            inp[:, self.PS_idx],
            self.inp_div[self.PS_idx], self.inp_sub[self.PS_idx],
            self.norm_q
        )

        # 2. Compute net energy input from phase change and precipitation
        PHAS = L_I / L_V * (
                (prior[:, self.PRECST_idx] + prior[:, self.PRECSTEND_idx]) -
                (prior[:, self.PRECT_idx] + prior[:, self.PRECTEND_idx])
        )

        # 3. Compute net energy input from radiation, SHFLX and TKE
        RAD = (prior[:, self.FSNT_idx] - prior[:, self.FSNS_idx] -
               prior[:, self.FLNT_idx] + prior[:, self.FLNS_idx])
        SHFLX = (inp[:, self.SHFLX_idx] * self.inp_div[self.SHFLX_idx] +
                 self.inp_sub[self.SHFLX_idx])
        KEDINT = K.sum(dP_tilde * prior[:, self.DTVKE_idx], 1)

        # 4. Compute tendency of vapor due to phase change
        LHFLX = (inp[:, self.LHFLX_idx] * self.inp_div[self.LHFLX_idx] +
                 self.inp_sub[self.LHFLX_idx])
        VAPINT = K.sum(dP_tilde * prior[:, self.PHQ_idx], 1)
        SPDQINT = (VAPINT - LHFLX) * L_S / L_V

        # 5. Same for cloud liquid water tendency
        SPDQCINT = K.sum(dP_tilde * prior[:, self.PHCLDLIQ_idx], 1) * L_I / L_V

        # 6. And the same for T but remember residual is still missing
        DTINT = K.sum(dP_tilde[:, :29] * prior[:, self.TPHYSTND_idx], 1)

        # 7. Compute DT30 as residual
        DT30 = (
                       PHAS + RAD + SHFLX + KEDINT - SPDQINT - SPDQCINT - DTINT
               ) / dP_tilde[:, 29]

        # 8. Concatenate output vector
        post = tf.concat([
            prior[:, :119], DT30[:, None], prior[:, 119:]
        ], axis=1)
        return post

    def compute_output_shape(self, input_shape):
        """Input shape + 1"""
        return (input_shape[0][0], input_shape[0][1] + 1)


layer_dict = {
    'SurRadLayer': SurRadLayer,
    'MassConsLayer': MassConsLayer,
    'EntConsLayer': EntConsLayer
}
