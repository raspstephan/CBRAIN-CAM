#!/usr/bin/env python3
import numpy as np
import scipy.io
import h5py
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

file_name = './1028_094639_SPDT,SPDQ_layers_32,32,32,32,32,32__kdr_1.0_ac_0_convo_True_variables_TAP,QAP,OMEGA,SHFLX,LHFLX,dTdt_adiabatic,dQdt_adiabatic_batchsize_512/model.ckpt-515939'
reader = pywrap_tensorflow.NewCheckpointReader(file_name)
var_to_shape_map = reader.get_variable_to_shape_map()
filters = []
biases = []
bad_keys = ['power', 'Adam', 'lr', 'step']
for key in sorted(var_to_shape_map):
  if not any([e in key for e in bad_keys]):
    tensor = reader.get_tensor(key)
    matrix = np.squeeze(np.array(tensor))
    ndims = np.ndim(matrix)
    keyclean = key.replace ('/','_')
    filename = keyclean+'.mat'
    scipy.io.savemat(filename, mdict={keyclean: matrix})#            np.savetxt(filename,matrix)    
