"""
Just all the imports for all other scripts and notebooks.
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
import pandas as pd
import pickle
import pdb
import netCDF4 as nc
import xarray as xr
import h5py
from glob import glob
import sys, os
base_dir = os.getcwd().split('CBRAIN-Keras-Diagnostics/')[0] + 'CBRAIN-Keras-Diagnostics/'
sys.path.append(f'{base_dir}keras_network/')
sys.path.append(f'{base_dir}data_processing/')
from .losses import *
from keras.utils.generic_utils import get_custom_objects
metrics_dict = dict([(f.__name__, f) for f in all_metrics])
get_custom_objects().update(metrics_dict)

from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()

if in_notebook():
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm