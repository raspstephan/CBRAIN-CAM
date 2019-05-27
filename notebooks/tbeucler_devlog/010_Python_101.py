# tgb - 5/1/2019 - Trains architecture-constrained network with al=0.01 loss function
# tgb - 4/18/2019 - Python script callable from command line
# Follows notebook 010 @ https://github.com/tbeucler/CBRAIN-CAM/blob/master/notebooks/tbeucler_devlog/010_Conserving_Network_Paper_Runs.ipynb

from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()
TRAINDIR = '/local/Tom.Beucler/SPCAM_PHYS/'
DATADIR = '/project/meteo/w2w/A6/S.Rasp/SP-CAM/fluxbypass_aqua/'
PREFIX = '8col009_01_'

import os
os.chdir('/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM')

scale_dict = load_pickle('./nn_config/scale_dicts/009_Wm2_scaling.pkl')
in_vars = load_pickle('./nn_config/scale_dicts/009_Wm2_in_vars.pkl')
out_vars = load_pickle('./nn_config/scale_dicts/009_Wm2_out_vars.pkl')
dP = load_pickle('./nn_config/scale_dicts/009_Wm2_dP.pkl')

train_gen = DataGenerator(
    data_fn = TRAINDIR+PREFIX+'train_shuffle.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+PREFIX+'norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True
)
valid_gen = DataGenerator(
    data_fn = TRAINDIR+PREFIX+'valid.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+PREFIX+'norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=False
)
      
Nep = 20
# Repeat for architecture-constrained network
graph = tf.Graph()
with tf.Session(graph=graph):

    # 1) Create model
    # Unconstrained model with 5 dense layers (Notebook 009)
    inpC = Input(shape=(304,))
    densout = Dense(512, activation='linear')(inpC)
    densout = LeakyReLU(alpha=0.3)(densout)
    for i in range (4):
        densout = Dense(512, activation='linear')(densout)
        densout = LeakyReLU(alpha=0.3)(densout)
    densout = Dense(214, activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
    surfout = SurRadLayer(
        inp_div=train_gen.input_transform.div,
        inp_sub=train_gen.input_transform.sub,
        norm_q=scale_dict['PHQ'],
        hyai=hyai, hybi=hybi
    )([inpC, densout])
    massout = MassConsLayer(
        inp_div=train_gen.input_transform.div,
        inp_sub=train_gen.input_transform.sub,
        norm_q=scale_dict['PHQ'],
        hyai=hyai, hybi=hybi
    )([inpC, surfout])
    enthout = EntConsLayer(
        inp_div=train_gen.input_transform.div,
        inp_sub=train_gen.input_transform.sub,
        norm_q=scale_dict['PHQ'],
        hyai=hyai, hybi=hybi
    )([inpC, massout])
    NNA = tf.keras.models.Model(inpC, enthout)
    print(NNA.summary())
    
    al = 0.01/4
    Loss = WeakLoss(inpC, inp_div=train_gen.input_transform.div,
                            inp_sub=train_gen.input_transform.sub,
                            norm_q=scale_dict['PHQ'],
                            hyai=hyai, hybi=hybi, name='loss',
                            alpha_mass=al, alpha_ent=al,
                            alpha_lw=al, alpha_sw=al)
    
    # 2) Compile model
    NNA.compile(tf.keras.optimizers.RMSprop(), loss=Loss, metrics=[mse])

    # 3) Train model
    NNA.fit_generator(train_gen, epochs=Nep, validation_data=valid_gen)

    # 4) Save model
    path = TRAINDIR+'HDF5_DATA/NNA0.01.h5'
    NNA.save(path)
    print('NN saved in ',path)