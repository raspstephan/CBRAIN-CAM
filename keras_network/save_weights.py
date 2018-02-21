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
import numpy as np


def save2txt(weight_file, save_dir):
    fmt = '%.6e'
    weights = []; biases = []
    with h5py.File(weight_file, 'r') as f:
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        for il, l in enumerate(layer_names):
            g = f[l]
            w = g[l + '/kernel:0'][:]
            b = g[l + '/bias:0'][:]
            weights.append(w); biases.append(b)
            np.savetxt(save_dir+f'/layer{il+1}_kernel.txt', w.T, fmt=fmt,
                       delimiter=',')
            np.savetxt(save_dir + f'/layer{il+1}_bias.txt', b.reshape(1, -1),
                       fmt=fmt, delimiter=',')


def main(inargs):
    """Load saved model and save weights

    Args:
        inargs: Namespace

    """

    model = load_model(inargs.model_path)

    model.save_weights(inargs.save_path)

    if inargs.save_txt is not None:
        save2txt(inargs.save_path, inargs.save_txt)


if __name__ == '__main__':

    p = ArgParser()
    p.add_argument('--model_path',
                   type=str,
                   help='Path to model')
    p.add_argument('--save_path',
                   type=str,
                   help='Path for saved weights.')
    p.add_argument('--save_txt',
                   type=str,
                   default=None,
                   help='Save weights and biases as textfiles for F90')

    args = p.parse_args()
    main(args)
