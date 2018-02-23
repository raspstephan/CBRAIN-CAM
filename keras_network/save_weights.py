"""Little script to save weights from saved model.

Author: Stephan Rasp
"""

from keras.models import load_model
from configargparse import ArgParser
from losses import *
from keras.utils.generic_utils import get_custom_objects
metrics_dict = dict([(f.__name__, f) for f in all_metrics])
get_custom_objects().update(metrics_dict)
import h5py
import os, sys
import netCDF4 as nc
import numpy as np
fmt = '%.6e'


def save2txt(weight_file, save_dir):
    weights = []; biases = []
    with h5py.File(weight_file, 'r') as f:
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']
                       if not 'activation' in n.decode('utf8')]
        for il, l in enumerate(layer_names):
            g = f[l]
            w = g[l + '/kernel:0'][:]
            b = g[l + '/bias:0'][:]
            weights.append(w); biases.append(b)
            np.savetxt(save_dir+f'/layer{il+1}_kernel.txt', w.T, fmt=fmt,
                       delimiter=',')
            np.savetxt(save_dir + f'/layer{il+1}_bias.txt', b.reshape(1, -1),
                       fmt=fmt, delimiter=',')


def save_norm(norm_path, save_dir):
    with nc.Dataset(norm_path) as ds:
        np.savetxt(
            save_dir + '/inp_means.txt', ds['feature_means'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/inp_stds.txt', ds['feature_stds'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/outp_mins.txt', ds['target_mins'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/outp_maxs.txt', ds['target_maxs'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')


def main(inargs):
    """Load saved model and save weights

    Args:
        inargs: Namespace

    """
    if not os.path.exists(inargs.save_dir):
        os.makedirs(inargs.save_dir, exist_ok=True)
    if inargs.model_path is not None:
        model = load_model(inargs.model_path)
        model.save_weights(inargs.save_dir + 'weights.h5')

    if inargs.save_txt:
        save2txt(inargs.save_dir + 'weights.h5', inargs.save_dir)
    if inargs.save_norm is not None:
        save_norm(inargs.save_norm, inargs.save_dir)


if __name__ == '__main__':

    p = ArgParser()
    p.add_argument('--model_path',
                   default=None,
                   type=str,
                   help='Path to model')
    p.add_argument('--save_dir',
                   type=str,
                   help='Path for saved weights.')
    p.add_argument('--save_txt',
                   action='store_true',
                   help='Save weights and biases as textfiles for F90')
    p.set_defaults(save_txt=False)
    p.add_argument('--save_norm',
                   type=str,
                   default=None,
                   help='Save mean and std as textfiles for F90')

    args = p.parse_args()
    main(args)
