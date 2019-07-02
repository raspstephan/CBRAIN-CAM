# tgb - 5/26/2019 - The goal is to average the Jacobian over a large number of timesteps to obtain smooth diagnostics. Following notebook at https://github.com/tbeucler/CBRAIN-CAM/blob/master/notebooks/tbeucler_devlog/014_Developing_Jacobian_diagnostics.ipynb
# tgb - 5/1/2019 - Calculates budget residuals on NNLA for all datasets
# tgb - 4/27/2019 - Calculates precipitation PDF for each network on +0K and +4K
# tgb - 4/24/2019 - Data-scarce using just 1-file --> calculate statistics on +0,1,2,3,4K; NNA version
# tgb - 4/24/2019 - Validate the unconstrained multiple linear regression model on +1,2,3,4K
# tgb - 4/22/2019 - Use +1K as validation dataset
# tgb - 4/19/2019 - The goal is to make a slurm-callable script to calculate the statistics and residuals of all the paper neural networks over the validation dataset. This script is specialized to the +0K experiment.

import tensorflow as tf
tf.enable_eager_execution()

from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
from cbrain.model_diagnostics import *
import tensorflow.math as tfm
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
from numpy import linalg as LA
import matplotlib.pyplot as plt
# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()
TRAINDIR = '/local/Tom.Beucler/SPCAM_PHYS/'
DATADIR = '/project/meteo/w2w/A6/S.Rasp/SP-CAM/fluxbypass_aqua/'
PREFIX = '8col009_01_'
NNs = 'NNA0.01' # NNA0.01

import os
os.chdir('/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM')

print(tf.encode_base64("Loading coordinates..."))
print('Loading coordinates...')
coor = xr.open_dataset("/project/meteo/w2w/A6/S.Rasp/SP-CAM/fluxbypass_aqua/AndKua_aqua_SPCAM3.0_sp_fbp_f4.cam2.h1.0000-01-01-00000.nc",\
                    decode_times=False)
lat = coor.lat; lon = coor.lon;
coor.close();

config_fn = '/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM/pp_config/8col_rad_tbeucler_local_PostProc.yml'
data_fn = '/local/Tom.Beucler/SPCAM_PHYS/8col009_01_valid.nc'
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer}

NN = {};
os.chdir(TRAINDIR+'/HDF5_DATA')
path = TRAINDIR+'HDF5_DATA/'+NNs+'.h5'
print('Loading neural network and model diagnostics object...')
NN = load_model(path,custom_objects=dict_lay)
md = ModelDiagnostics(NN,config_fn,data_fn)
NN.summary()

import h5py

print('Defining function...')
def get_RADCONjacobian(model,inp,md,ind):
# model is the neural network model from inp to out
# inp is the input x generator from the generator (object = gen_obj)
# sample_index is the reference number of the sample for x
# md is the model diagnostics object
# ind is the indices over which the Jacobian is calculated
# x.shape = (#sample,#inputs) so we are evaluating the gradient of
# y(sample_index,:) with respect to x(sample_index,:)
# cf is the conversion factor calculated using CAM constants
    cf = np.zeros((1, md.valid_gen.n_inputs))
    for index in range (md.valid_gen.n_inputs):
        if index<90: cf[0,index]=L_V;
        elif index<120: cf[0,index]=C_P;
        elif index<150: cf[0,index]=1;
        elif index<240: cf[0,index]=L_V*DT;
        elif index<270: cf[0,index]=C_P*DT;
        elif index<301: cf[0,index]=1;
    else: cf[0,index]=DT;
    
    JCON = np.zeros((60,60,len(ind)))
    JRAD = np.zeros((60,60,len(ind)))
    
    J = np.zeros((md.valid_gen.n_outputs,md.valid_gen.n_inputs,len(ind)))
    for count,i in enumerate(ind):
        print('i=',i,'/',len(ind)-1,end="\r")
        with tf.GradientTape(persistent=True) as tape:
            TFinp = tf.convert_to_tensor(np.expand_dims(inp[i,:],axis=0))
            tape.watch(TFinp)
            pred = NN(TFinp)
        J[:,:,i] = tape.jacobian(pred,TFinp,experimental_use_pfor=False)\
        .numpy().squeeze()/(cf*md.valid_gen.input_transform.div)
    
    for i in range (60):
        for j in range(60):
            # Convection
            if (i<30) and (j<30): JCON[i,j] = J[i,j,:] # d(dq/dt)/dq
            elif (i>29) and (j<30): JCON[i,j] = J[90+(i-30),j,:]-\
                J[120+(i-30),j,:]-J[150+(i-30),j,:] # d(dTcon/dt)/dq
            elif (i<30) and (j>29): JCON[i,j] = J[i,90+(j-30),:] # d(dq/dt)/dT
            elif (i>29) and (j>29): JCON[i,j] = J[90+(i-30),90+(j-30),:]-\
                J[120+(i-30),90+(j-30),:]-J[150+(i-30),90+(j-30),:] # d(dTcon/dt)/dT
            # Radiation    
            if (i<30) and (j<30): JRAD[i,j,:] = J[120+i,j,:]
            elif (i>29) and (j<30): JRAD[i,j,:] = J[150+(i-30),j,:]
            elif (i<30) and (j>29): JRAD[i,j,:] = J[120+i,90+(j-30),:]
            elif (i>29) and (j>29): JRAD[i,j,:] = J[150+(i-30),90+(j-30),:]

    return JCON,JRAD

JCONa = np.zeros((md.nlat,60,60))
JRADa = np.copy(JCONa)
with h5py.File('HDF5_DATA/014_'+NNs+'_JCON.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=JCONa)
with h5py.File('HDF5_DATA/014_'+NNs+'_JRAD.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=JRADa)

print('Function defined. Now calculating Jacobian time step by time step')
for itime in range(100):
    print('itime=',itime)
    X, truth = md.valid_gen[itime]
    Xgeo = X.values.reshape(md.nlat, md.nlon, 304)
    for ilat in range(md.nlat):
        print('ilat=',ilat,'/',str(md.nlat-1),end="\r")
        Xlat = Xgeo[ilat,:,:]
        JCON,JRAD = get_RADCONjacobian(NN,Xlat,md,np.arange(0,md.nlon))
        JCONa[ilat,:,:] = JCON.mean(axis=2)/(itime+1)+\
        itime/(itime+1)*JCONa[ilat,:,:]
        JRADa[ilat,:,:] = JRAD.mean(axis=2)/(itime+1)+\
        itime/(itime+1)*JRADa[ilat,:,:]

    print('itime=',itime,'Saving the arrays in HDF5 format')
    with h5py.File('HDF5_DATA/014_'+NNs+'_JCON.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=JCONa)
    with h5py.File('HDF5_DATA/014_'+NNs+'_JRAD.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=JRADa)
        
