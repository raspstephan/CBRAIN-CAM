"""
Functions to save keras model weights to CAM text files.

Created on 2019-01-28-11-28
Author: Stephan Rasp, raspstephan@gmail.com
"""

from .imports import *


fmt = '%.6e'


def save2txt(weight_file, save_dir):
    """Saves model weights to text file"""
    weights = []; biases = []
    with h5py.File(weight_file, 'r') as f:
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']
                       if 'dense' in n.decode('utf8')]
        for il, l in enumerate(layer_names):
            g = f[l]
            w = g[l + '/kernel:0'][:]
            b = g[l + '/bias:0'][:]
            weights.append(w); biases.append(b)
            np.savetxt(save_dir+f'/layer{il+1}_kernel.txt', w.T, fmt=fmt,
                       delimiter=',')
            np.savetxt(save_dir + f'/layer{il+1}_bias.txt', b.reshape(1, -1),
                       fmt=fmt, delimiter=',')


def save_norm(input_transform, output_transform, save_dir):
    """Save normalization arrays to text file."""
    for name, arr in input_transform.transform_arrays.items():
        np.savetxt(save_dir + f'/inp_{name}.txt', arr.reshape(1, -1), fmt=fmt, delimiter=',')
    for name, arr in output_transform.transform_arrays.items():
        np.savetxt(save_dir + f'/out_{name}.txt', arr.reshape(1, -1), fmt=fmt, delimiter=',')
