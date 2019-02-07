"""
Just all the imports for all other scripts and notebooks.
tgb - 2/7/2019 - Replacing keras with tensorflow.keras for eager execution purposes
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
#import keras
import tensorflow.keras
from tensorflow.keras.callbacks import *
import pandas as pd
import pickle
import pdb
import netCDF4 as nc
import xarray as xr
import h5py
from glob import glob
import sys, os
import seaborn as sns
base_dir = os.getcwd().split('CBRAIN-CAM/')[0] + 'CBRAIN-CAM/'
sys.path.append(f'{base_dir}keras_network/')
sys.path.append(f'{base_dir}data_processing/')
from .losses import *
from .models import PartialReLU, QLayer, ELayer, MasConsLay, EntConsLay
from tensorflow.keras.utils import get_custom_objects
metrics_dict = dict([(f.__name__, f) for f in all_metrics])
get_custom_objects().update(metrics_dict)
get_custom_objects().update({
    'PartialReLU': PartialReLU,
    'QLayer': QLayer,
    'ELayer': ELayer,
    'MasConsLay': MasConsLay,
    'EntConsLay': EntConsLay
    })
from configargparse import ArgParser

from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()

if in_notebook():
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
P0 = 1e5

with open(os.path.join(os.path.dirname(__file__), 'hyai_hybi.pkl'), 'rb') as f:
    hyai, hybi = pickle.load(f)