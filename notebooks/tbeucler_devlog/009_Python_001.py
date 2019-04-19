# tgb - 4/17/2019 - Calculates Model Diagnostics statistics in python script
# to call it from command line in order to avoid timeout connection problems
# Follows the notebook 009 @ https://github.com/tbeucler/CBRAIN-CAM/blob/master/notebooks/tbeucler_devlog/009_Generalization_Climate_Change_8col.ipynb

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
# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()
TRAINDIR = '/local/Tom.Beucler/SPCAM_PHYS/'
DATADIR = '/project/meteo/w2w/A6/S.Rasp/SP-CAM/fluxbypass_aqua/'
PREFIX = '8col009_01_'

import os
os.chdir('/local/Tom.Beucler/SPCAM_PHYS/HDF5_DATA')
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer}
C_009 = load_model('C_009.h5',custom_objects=dict_lay)
U_009 = load_model('U_009.h5',custom_objects=dict_lay)

from cbrain.model_diagnostics.model_diagnostics import ModelDiagnostics
config_fn = '/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM/pp_config/8col_rad_tbeucler_local_PostProc.yml'
data_fn = '/local/Tom.Beucler/SPCAM_PHYS/8col009_01_valid.nc'
data4k_fn = '/local/Tom.Beucler/SPCAM_PHYS/8col009_02_valid.nc'
data_test = '/local/Tom.Beucler/SPCAM_PHYS/8col009_03_train.nc'
mdC = ModelDiagnostics(C_009,config_fn,data_fn)
mdU = ModelDiagnostics(U_009,config_fn,data_fn)
mdC4k = ModelDiagnostics(C_009,config_fn,data4k_fn)
mdU4k = ModelDiagnostics(U_009,config_fn,data4k_fn)

mdC.compute_stats()
pickle.dump(mdC.stats,open('mdCstats.pkl','wb'))

mdU.compute_stats()
pickle.dump(mdU.stats,open('mdUstats.pkl','wb'))

mdU4k.compute_stats()
pickle.dump(mdU4k.stats,open('mdU4kstats.pkl','wb'))

mdC4k.compute_stats()
pickle.dump(mdC4k.stats,open('mdC4kstats.pkl','wb'))

