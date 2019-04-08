"""
Define all different types of models.

<<<<<<< HEAD
Author: Stephan Rasp
tgb - 2/7/2019 - Adding mass and enthalpy conservation layers as models
tgb - 2/7/2019 - Replacing keras with tf.keras to avoid incompatibilities when using tensorflow's eager execution
"""

from .imports import *
#import keras
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from .losses import *
from .cam_constants import *
from .layers import *
act_dict = tensorflow.keras.activations.__dict__
lyr_dict = tensorflow.keras.layers.__dict__

L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
P0 = 1e5

# tgb - 2/11/2019 - Add surface radiation layer from notebook 005
class SurRadLay(Layer):
    
    def __init__(self, fsub, fdiv, normq, hyai, hybi, output_dim, **kwargs):
        self.fsub = fsub # Subtraction for normalization of inputs 
        self.fdiv = fdiv # Division for normalization of inputs
        self.normq = normq # Normalization of output's water concentration
        self.hyai = hyai # CAM constants to calculate d_pressure
        self.hybi = hybi # CAM constants to calculate d_pressure
        self.output_dim = output_dim # Dimension of output
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!
    
    # tgb - 2/6/2019 - following https://github.com/keras-team/keras/issues/4871
    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv),
                  'normq': list(self.normq), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi), 'output_dim': self.output_dim}
        base_config = super(SurRadLay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, arrs):
    
        # Split between the inputs inp & the output of the densely connected
        # neural network, massout
        inp, densout = arrs
        
        # 0) Constants
        C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
        G = 9.80616; # Reference gravity constant [m.s-2]
        L_V = 2.501e6; # Latent heat of vaporization of water [W.kg-1]
        P0 = 1e5; # Reference surface pressure [Pa]

        # 1) Get non-dimensional pressure differences (p_tilde above)
        PS = tfm.add( tfm.multiply( inp[:,300], self.fdiv[300]), self.fsub[300])
        # Reference for calculation of d_pressure is cbrain/models.py (e.g. QLayer)
        P = tfm.add( tfm.multiply( P0, self.hyai), \
                    tfm.multiply( PS[:,None], self.hybi))
        dP = tfm.subtract( P[:, 1:], P[:, :-1])
        # norm_output = dp_norm * L_V/G so dp_norm = norm_output * G/L_V
        dP_NORM = tfm.divide( \
                             tfm.multiply(self.normq[:30], \
                                          G),\
                             L_V)
        # dp_tilde = dp/dp_norm
        dP_TILD = tfm.divide( dP, dP_NORM)

        # 2) Radiative integrals
        SWVEC = tfm.multiply( dP_TILD, densout[:, 148:178])
        SWINT = tfm.reduce_sum( SWVEC, axis=1)

        LWVEC = tfm.multiply( dP_TILD, densout[:, 118:148])
        LWINT = tfm.reduce_sum( LWVEC, axis=1) # LW integral

        # 3) Infer surface radiative fluxes from radiative integrals and TOA radiative fluxes
        FSNS = tfm.subtract( densout[:,208], SWINT) # FSNS = FSNT-SWINT
        FLNS = tfm.add( densout[:,209], LWINT) # FLNS = FLNT+LWINT

        # 4) Concatenate the input of the dense layer with 
        # the net surface radiative fluxes to form 
        # the output of the surface radiation layer
        out = tf.concat([densout[:, :209], tf.expand_dims(FSNS, axis=1),\
                               densout[:, 209:210], tf.expand_dims(FLNS, axis=1),\
                              densout[:, 210:]], 1)
        
        return out
    
    def compute_output_shape(self, input_shape):
        # tgb - 2/7/2019 - Wrap the returned output shape in Tensorshape
        # to avoid problems with custom layers & eager execution
        # https://github.com/tensorflow/tensorflow/issues/20805
        return tf.TensorShape((input_shape[0][0], self.output_dim)) 
    # The layer takes inputs from the previous layer that have shape 124
    # and outputs y of shape 126 to be fed to the mass cons. layers


# tgb - 2/5/2019 - Adapated the mass conservation layer to new input format
# tgb - 2/11/2019 - Did it again to predict radiation and convection separately
class MasConsLay(Layer):
    
    def __init__(self, fsub, fdiv, normq, hyai, hybi, output_dim, **kwargs):
        self.fsub = fsub # Subtraction for normalization of inputs 
        self.fdiv = fdiv # Division for normalization of inputs
        self.normq = normq # Normalization of output's water concentration
        self.hyai = hyai # CAM constants to calculate d_pressure
        self.hybi = hybi # CAM constants to calculate d_pressure
        self.output_dim = output_dim # Dimension of output
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!
    
    # tgb - 2/6/2019 - following https://github.com/keras-team/keras/issues/4871
    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv),
                  'normq': list(self.normq), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi), 'output_dim': self.output_dim}
        base_config = super(MasConsLay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, arrs):        
        # Split between the inputs inp & the output of the densely connected
        # neural network, sradout
        inp, sradout = arrs
        
        # 0) Constants
        G = 9.80616; # Reference gravity constant [m.s-2]
        L_V = 2.501e6; # Latent heat of vaporization of water [W.kg-1]
        P0 = 1e5; # Reference surface pressure [Pa]
        
        # 1) Get non-dimensional pressure differences (p_tilde above)
        # In the input vector, PS is the 151st element after 
        # the first elements = [QBP, ..., VBP with shape 30*5=150]
        PS = tfm.add( tfm.multiply( inp[:,300], self.fdiv[300]), self.fsub[300])
        # Reference for calculation of d_pressure is cbrain/models.py (e.g. QLayer)
        P = tfm.add( tfm.multiply( P0, self.hyai), \
                    tfm.multiply( PS[:,None], self.hybi))
        dP = tfm.subtract( P[:, 1:], P[:, :-1])
        # norm_output = dp_norm * L_V/G so dp_norm = norm_output * G/L_V
        dP_NORM = tfm.divide( \
                             tfm.multiply(self.normq[:30], \
                                   G), L_V)
        # dp_tilde = dp/dp_norm
        # Wondering about broadcasting here...
        # tf.div or simply \ would support broadcasting 
        dP_TILD = tfm.divide( dP, dP_NORM)
        
        # 2) Calculate cloud water vertical integral from level 1 to level 30
        # The indices are tricky here because we are missing del(q_v)@(level 30)
        # so e.g. q_liq@(level 1) is the 30th element of the output of the 
        # previous dense layer
        CLDVEC = tfm.multiply( dP_TILD, \
                                  tfm.add( sradout[:, 29:59], sradout[:, 59:89]))
        CLDINT = tfm.reduce_sum( CLDVEC, axis=1)
        
        # 3) Calculate water vapor vertical integral from level 1 to level 29
        VAPVEC = tfm.multiply( dP_TILD[:, :29], \
                                  sradout[:, :29])
        VAPINT = tfm.reduce_sum( VAPVEC, axis=1)
        
        # 4) Calculate forcing on the right-hand side (Net Evaporation-Precipitation)
        # E-P is already normalized to units W.m-2 in the output vector
        # so all we need to do is input-unnormalize LHF that is taken from the input vector
        LHF = tfm.add( tfm.multiply( inp[:,303], self.fdiv[303]), self.fsub[303])
        # Note that total precipitation = PRECT + 1e-3*PRECTEND in the CAM model
        # PRECTEND already multiplied by 1e-3 in output vector so no need to redo it
        # tgb - 2/8/2019 - This is the only line modified from the large-scale version 002
        PREC = tfm.add( sradout[:, 212], sradout[:, 213])
        
        # 5) Infer water vapor tendency at level 30 as a residual
        # Composing tfm.add 3 times because not sure how to use tfm.add_n
        DELQV30 = tfm.divide( \
                             tfm.add( tfm.add( tfm.add (\
                                                        LHF, tfm.negative(PREC)), \
                                              tfm.negative(CLDINT)), \
                                     tfm.negative(VAPINT)), \
                             dP_TILD[:, 29])
        
        # 6) Concatenate the water tendencies with the newly inferred tendency
        # to get the final vector out of shape (#samples,125) with
        # [DELQ, DELCLDLIQ, DELCLDICE, 
        # TPHYSTND\{TPHYSTND AT SURFACE}, FSNT, FSNS, FLNT, FLNS, PRECT PRECTEND]
        # Uses https://www.tensorflow.org/api_docs/python/tf/concat
        DELQV30 = tf.expand_dims(DELQV30,1) # Adds dimension=1 to axis=1
        out = tf.concat([sradout[:, :29], DELQV30, sradout[:, 29:]], 1)
        return out
    
    def compute_output_shape(self, input_shape):
        # tgb - 2/7/2019 - Wrap the returned output shape in Tensorshape
        # to avoid problems with custom layers & eager execution
        # https://github.com/tensorflow/tensorflow/issues/20805
        return tf.TensorShape((input_shape[0][0], self.output_dim)) 
    # The output has size 125=30*4+6-1
    # and is ready to be fed to the energy conservation layer
    # before we reach the total number of outputs = 126
    
# tgb - 2/5/2019 - Change to adapt to new input format
# tgb - 2/11/2019 - Did it again to conserve enthalpy while predicting both radiation and convection
class EntConsLay(Layer):
    
    def __init__(self, fsub, fdiv, normq, hyai, hybi, output_dim, **kwargs):
        self.fsub = fsub # Subtraction for normalization of inputs 
        self.fdiv = fdiv # Division for normalization of inputs
        self.normq = normq # Normalization of output's water concentration
        self.hyai = hyai # CAM constants to calculate d_pressure
        self.hybi = hybi # CAM constants to calculate d_pressure
        self.output_dim = output_dim # Dimension of output
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!
        
    # tgb - 2/6/2019 - following https://github.com/keras-team/keras/issues/4871
    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv),
                  'normq': list(self.normq), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi), 'output_dim': self.output_dim}
        base_config = super(EntConsLay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, arrs):
        
        # Split between the inputs inp & the output of the densely connected
        # neural network, massout
        inp, massout = arrs
        
        # 0) Constants
        G = 9.80616; # Reference gravity constant [m.s-2]
        L_F = 3.337e5; # Latent heat of fusion of water [W.kg-1]
        L_V = 2.501e6; # Latent heat of vaporization of water [W.kg-1]
        P0 = 1e5; # Reference surface pressure [Pa]
        
        # 1) Get non-dimensional pressure differences (p_tilde above)
        # In the input vector, PS is the 151st element after 
        # the first elements = [QBP, ..., VBP with shape 30*5=150]
        PS = tfm.add( tfm.multiply( inp[:,300], self.fdiv[300]), self.fsub[300])
        # Reference for calculation of d_pressure is cbrain/models.py (e.g. QLayer)
        P = tfm.add( tfm.multiply( P0, self.hyai), \
                    tfm.multiply( PS[:,None], self.hybi))
        dP = tfm.subtract( P[:, 1:], P[:, :-1])
        # norm_output = dp_norm * L_V/G so dp_norm = norm_output * G/L_V
        dP_NORM = tfm.divide( \
                             tfm.multiply(self.normq[:30], \
                                          G),\
                             L_V)
        # dp_tilde = dp/dp_norm
        dP_TILD = tfm.divide( dP, dP_NORM)
        
        # 2) Calculate net energy input from phase change and precipitation
        # PHAS = Lf/Lv*((PRECST+PRECSTEN)-(PRECT+PRECTEND))
        PHAS = tfm.divide( tfm.multiply( tfm.subtract(\
                                                      tfm.add( massout[:,215], massout[:,216]),\
                                                      tfm.add( massout[:,213], massout[:,214])),\
                                        L_F),\
                          L_V)
        
        # 3) Calculate net energy input from radiation, sensible heat flux and turbulent KE
        # 3.1) RAD = FSNT-FSNS-FLNT+FLNS
        RAD = tfm.add(\
                      tfm.subtract( massout[:,209], massout[:,210]),\
                      tfm.subtract( massout[:,212], massout[:,211]))
        # 3.2) Unnormalize sensible heat flux
        SHF = tfm.add( tfm.multiply( inp[:,302], self.fdiv[302]), self.fsub[302])
        # 3.3) Net turbulent kinetic energy dissipative heating is the column-integrated 
        # turbulent kinetic energy energy dissipative heating
        # tbeucler - Be careful here as final KE dissipation is between indices 180-209
        KEDVEC = tfm.multiply( dP_TILD, massout[:, 179:209])
        KEDINT = tfm.reduce_sum( KEDVEC, axis=1)
        
        # 4) Calculate tendency of normalized column water vapor due to phase change
        # 4.1) Unnormalize latent heat flux
        LHF = tfm.add( tfm.multiply( inp[:,303], self.fdiv[303]), self.fsub[303])
        # 4.2) Column water vapor is the column integral of specific humidity
        PHQVEC = tfm.multiply( dP_TILD, massout[:, :30])
        PHQINT = tfm.reduce_sum( PHQVEC, axis=1)
        # 4.3) Multiply by L_S/L_V to normalize (explanation above)
        SPDQINT = tfm.divide( tfm.multiply( tfm.subtract(\
                                                         PHQINT, LHF),\
                                           L_S),\
                             L_V)
        
        # 5) Same operation for liquid water tendency but multiplied by L_F/L_V
        SPDQCINT = tfm.divide( tfm.multiply(\
                                            tfm.reduce_sum(\
                                                           tfm.multiply( dP_TILD, massout[:, 30:60]),\
                                                           axis=1),\
                                            L_F),\
                              L_V)
        
        # 6) Same operation for temperature but only integrate from level 1 to level 29
        DTINT = tfm.reduce_sum( tfm.multiply( dP_TILD[:, :29], massout[:, 90:119]), axis=1)

        # 7) Now calculate dT30 as a residual
        dT30 = tfm.divide(tfm.add(tfm.add(tfm.add(tfm.add(tfm.add(tfm.add(\
                                                                          PHAS,RAD),\
                                                                  SHF),\
                                                          KEDINT),\
                                                  tfm.negative( SPDQINT)),\
                                          tfm.negative( SPDQCINT)),\
                                  tfm.negative( DTINT)),\
                          dP_TILD[:, 29])
        dT30 = tf.expand_dims(dT30,1)

        out = tf.concat([massout[:, :119], dT30, massout[:, 119:]], 1)
        return out
    
    def compute_output_shape(self, input_shape):
        # tgb - 2/7/2019 - Wrap the returned output shape in Tensorshape
        # to avoid problems with custom layers & eager execution
        # https://github.com/tensorflow/tensorflow/issues/20805
        return tf.TensorShape((input_shape[0][0], self.output_dim))
    # and is ready to be used in the cost function

def cons_5dens():
# tgb - 2/7/2019 - Draft of the energy/mass conserving model
# Improve it using fc_model
    inp = Input(shape=(304,))
    densout = Dense(512, activation='linear')(inp)
    densout = LeakyReLU(alpha=0.3)(densout)
    for i in range (4):
        densout = Dense(512, activation='linear')(densout)
        densout = LeakyReLU(alpha=0.3)(densout)
    densout = Dense(156, activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
    massout = MasConsLay(
        input_shape=(156,), fsub=fsub, fdiv=fdiv, normq=normq,\
        hyai=hyai, hybi=hybi, output_dim = 157
    )([inp, densout])
    out = EntConsLay(
        input_shape=(157,), fsub=fsub, fdiv=fdiv, normq=normq,\
        hyai=hyai, hybi=hybi, output_dim = 158
    )([inp, massout])
    
    return tf.keras.models.Model(inp, out)    
    
class PartialReLU(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a = x[:, :60]
        b = K.maximum(x[:, 60:120], 0)
        c = x[:, 120:]
        return K.concatenate([a, b, c], 1)

    def compute_output_shape(self, input_shape):
        return input_shape

class QLayer(Layer):
    def __init__(self, fsub, fdiv, hyai, hybi, **kwargs):
        super().__init__(**kwargs)
        self.fsub, self.fdiv, self.hyai, self.hybi = fsub, fdiv, np.array(hyai), np.array(hybi)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, arrs):
        L_V = 2.501e6 ; L_I = 3.337e5; L_S = L_V + L_I
        C_P = 1.00464e3; G = 9.80616; P0 = 1e5

        f, a = arrs
        # Get pressure difference 
        PS = f[:, 90] * self.fdiv[90] + self.fsub[90]
        P = P0 * self.hyai + PS[:, None] * self.hybi
        dP = P[:, 1:] - P[:, :-1]

        # Get Convective integrals
        iPHQ = a[:, 30:60]*dP/G/L_S
        vPHQ = K.sum(iPHQ, 1)
        absvPHQ = K.sum(K.abs(iPHQ),1)
        # Sum for convective moisture
        dQCONV = vPHQ

        # Get surface flux
        LHFLX = (f[:, 93] * self.fdiv[93] + self.fsub[93]) / L_V

        # Get precipitation sink
        TOT_PRECL = a[:, 64] / (24*3600*2e-2)

        # Total differences to be corrected --> factor. Correct everything involved
        #pdb.set_trace()
        dQ = dQCONV - LHFLX + TOT_PRECL
        absTOT = absvPHQ + K.abs(TOT_PRECL)
        # Correct PHQ
        fact = dQ[:, None] * K.abs(iPHQ) / absTOT[:, None]
        b = a[:, 30:60] - fact / dP*G*L_S
        # Correct precipitation sink
        fact = dQ[:] * K.abs(TOT_PRECL) / absTOT[:]
        c = a[:, 64] - fact * (24*3600*2e-2)

        return K.concatenate([a[:, :30], b, a[:, 60:64], c[:, None]], 1)

    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv), 
                  'hyai': list(self.hyai), 'hybi': list(self.hybi)}
        base_config = super(QLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

class ELayer(Layer):
    def __init__(self, fsub, fdiv, hyai, hybi, **kwargs):
        super().__init__(**kwargs)
        self.fsub, self.fdiv, self.hyai, self.hybi = fsub, fdiv, np.array(hyai), np.array(hybi)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, arrs):
        L_V = 2.501e6 ; L_I = 3.337e5; L_S = L_V + L_I
        C_P = 1.00464e3; G = 9.80616; P0 = 1e5

        f, a = arrs
        # Get pressure difference 
        PS = f[:, 90] * self.fdiv[90] + self.fsub[90]
        P = P0 * self.hyai + PS[:, None] * self.hybi
        dP = P[:, 1:] - P[:, :-1]

        # Get Convective integrals
        iTPHY, iPHQ = a[:, :30]*dP/G, a[:, 30:60]*dP/G/L_S*L_V
        vTPHY, vPHQ = K.sum(iTPHY, 1), K.sum(iPHQ, 1)
        absvTPHY, absvPHQ = K.sum(K.abs(iTPHY),1), K.sum(K.abs(iPHQ),1)

        # Get surface fluxes
        SHFLX = f[:, 92] * self.fdiv[92] + self.fsub[92]
        LHFLX = (f[:, 93] * self.fdiv[93] + self.fsub[93])

        # Radiative fluxes
        dERADFLX = K.sum(a[:, -5:-1], 1) * 1e3
        absRADFLX = K.sum(K.abs(a[:, -5:-1]), 1) * 1e3

        # Total differences to be corrected --> factor. Correct heating and 2D terms
        dE = vTPHY - SHFLX - dERADFLX + vPHQ - LHFLX
        absTOT = absvTPHY + absRADFLX
        # Correct TPHY
        fact = dE[:, None] * K.abs(iTPHY) / absTOT[:, None]
        b = a[:, :30] - fact / dP*G
        # Correct Radiative fluxes
        fact = dE[:, None] * K.abs(a[:, -5:-1]) * 1e3 / absTOT[:, None]
        c = a[:, -5:-1] + fact / 1e3

        return K.concatenate([b, a[:, 30:60], c, a[:, -1][:, None]], 1)

    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv), 
                  'hyai': list(self.hyai), 'hybi': list(self.hybi)}
        base_config = super(ELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

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


