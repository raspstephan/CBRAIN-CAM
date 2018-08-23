"""Little script to save weights from saved model.

Author: Stephan Rasp
"""

from keras.models import load_model
from cbrain.imports import *
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


def save_norm(norm_path, save_dir):
    with nc.Dataset(norm_path) as ds:
        np.savetxt(
            save_dir + '/inp_means.txt', ds['feature_means'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/inp_stds.txt', ds['feature_stds'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/inp_mins.txt', ds['feature_mins'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/inp_maxs.txt', ds['feature_maxs'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/outp_mins.txt', ds['target_mins'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/outp_maxs.txt', ds['target_maxs'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/inp_stds_by_var.txt', ds['feature_stds_by_var'][:].reshape(1, -1),
            fmt=fmt, delimiter=',')
        np.savetxt(
            save_dir + '/inp_max_rs.txt',
            np.maximum(ds['feature_stds_by_var'][:],
                       ds['feature_maxs'][:] - ds['feature_mins'][:]).reshape(1, -1),
            fmt=fmt, delimiter=',')


def main(inargs):
    """Load saved model and save weights

    Args:
        inargs: Namespace

    """
    save_dir = f'./saved_models/{inargs.exp_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model = load_model(f'./saved_models/{inargs.exp_name}.h5')
    model.save_weights(save_dir + 'weights.h5')

    save2txt(save_dir + 'weights.h5', save_dir)

    if inargs.norm_fn is None:
        norm_fn = inargs.train_fn.split('_shuffle')[0] + '_norm.nc'
    else:
        norm_fn = inargs.norm_fn
    save_norm(inargs.data_dir + norm_fn, save_dir)


if __name__ == '__main__':
    p = ArgParser()
    p.add('-c', '--config_file',
          is_config_file=True,
          help='Name of config file.')
    p.add_argument('--exp_name',
                   default=None,
                   type=str,
                   help='Experiment name.')
    p.add_argument('--model_dir',
                   type=str,
                   default='./saved_models/',
                   help='Directory to save model to.')
    p.add_argument('--data_dir',
                   type=str,
                   help='Full outputs file.')
    p.add_argument('--train_fn',
                   type=str,
                   help='Training set file.')
    p.add_argument('--valid_fn',
                   type=str,
                   help='Validation set file.')
    p.add_argument('--norm_fn',
                   type=str,
                   default=None,
                   help='Normalization file. Default: None -> Infer')
    p.add_argument('--log_dir',
                   default=None,
                   type=str,
                   help='TensorBoard log dir')
    p.add_argument('--fsub',
                   default=None,
                   type=str,
                   help='Subtract feature array by. Default: None')
    p.add_argument('--fdiv',
                   default=None,
                   type=str,
                   help='Divide feature array by. Special: range. Default: None')
    p.add_argument('--tsub',
                   default=None,
                   type=str,
                   help='Subtract target array by. Default: None')
    p.add_argument('--tmult',
                   default=None,
                   type=str,
                   help='Divide target array by, e.g. target_conv. Default: None')
    p.add_argument('--loss',
                   default='mae',
                   type=str,
                   help='Loss function.')
    p.add_argument('--mse_var_ratio',
                   default=10,
                   type=float,
                   help='If mse_var loss function, multiplier for var.')
    p.add_argument('--activation',
                   default='relu',
                   type=str,
                   help='Activation function.')
    p.add_argument('--lr',
                   default=1e-3,
                   type=float,
                   help='Learning rate.')
    p.add_argument('--lr_step',
                   default=5,
                   type=int,
                   help='Step at which to divide learning rate by factor.')
    p.add_argument('--lr_divide',
                   default=5.,
                   type=float,
                   help='Factor to divide learning rate.')
    p.add_argument('--epochs',
                   default=30,
                   type=int,
                   help='Number of epochs')
    p.add_argument('--batch_size',
                   default=1024,
                   type=int,
                   help='Batch size')
    p.add_argument('--kernel_size',
                   default=3,
                   type=int,
                   help='Size of convolution kernel.')
    p.add_argument('--hidden_layers',
                   default=[],
                   nargs='+',
                   type=int,
                   help='List with hidden nodes.')
    p.add_argument('--dr',
                   default=None,
                   type=float,
                   help='Dropout rate.')
    p.add_argument('--noise',
                   default=None,
                   type=float,
                   help='Training noise std')
    p.add_argument('--l2',
                   default=None,
                   type=float,
                   help='L2 regularization for dense layers')
    p.add_argument('--conv_layers',
                   default=[],
                   nargs='+',
                   type=int,
                   help='List with feature maps')
    p.add_argument('--n_workers',
                   default=16,
                   type=int,
                   help='Workers for generator queue')
    p.add_argument('--max_queue_size',
                   default=50,
                   type=int,
                   help='Generator queue size')
    p.add_argument('--convolution',
                   dest='convolution',
                   action='store_true',
                   help='Use convolutional net.')
    p.set_defaults(convolution=False)
    p.add_argument('--batch_norm',
                   dest='batch_norm',
                   action='store_true',
                   help='Use batch_norm.')
    p.set_defaults(batch_norm=False)
    p.add_argument('--valid_after',
                   dest='valid_after',
                   action='store_true',
                   help='Only validate after training.')
    p.set_defaults(valid_after=False)
    p.add_argument('--locally_connected',
                   dest='locally_connected',
                   action='store_true',
                   help='Use locally connected convolutions.')
    p.set_defaults(locally_connected=False)
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='Print additional information.')
    p.set_defaults(verbose=False)
    p.add_argument('--partial_relu',
                   dest='partial_relu',
                   action='store_true',
                   help='...')
    p.set_defaults(partial_relu=False)
    p.add_argument('--eq',
                   dest='eq',
                   action='store_true',
                   help='...')
    p.set_defaults(eq=False)

    args = p.parse_args()

    main(args)
