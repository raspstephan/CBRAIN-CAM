"""Quick script to shuffle samples in a dataset.

Author: Stephan Rasp
"""

from configargparse import ArgParser
import numpy as np
from tqdm import tqdm
from netCDF4 import Dataset
import sys
from datetime import datetime
from subprocess import getoutput





def fast(features_orig, targets_orig, features_shuffle, targets_shuffle,
         chunk_size):
    # Try shuffling in batches that fit into RAM
    n_samples = features_orig.dimensions['sample'].size
    n_chunks = int(np.ceil(n_samples / float(chunk_size)))

    for i in tqdm(range(n_chunks)):
        if verbose >0: print('Chunk %i out of %i' % (i+1, n_chunks))
        start_idx = i * chunk_size
        stop_idx = np.min([(i+1) * chunk_size, n_samples])
        rand_idxs = np.arange(stop_idx - start_idx)
        np.random.shuffle(rand_idxs)
        for orig, shuffle, name in zip([features_orig, targets_orig],
                                       [features_shuffle, targets_shuffle],
                                       ['features', 'targets']):
            if verbose > 0: print('Loading %s chunk into RAM' % name)
            chunk = orig.variables[name][start_idx:stop_idx]

            if verbose > 0: print('Shuffling chunk')
            chunk = chunk[rand_idxs]

            if verbose > 0: print('Writing chunk')
            shuffle.variables[name][start_idx:stop_idx] = chunk


def meticulous(features_orig, targets_orig, features_shuffle, targets_shuffle):

    # Create random indices
    rand_idxs = np.arange(features_orig.dimensions['sample'].size)
    np.random.shuffle(rand_idxs)

    # Write data in random order
    for i in tqdm(range(rand_idxs.shape[0])):
        i_rand = rand_idxs[i]
        features_shuffle.variables['features'][i] = \
            features_orig.variables['features'][i_rand]
        targets_shuffle.variables['targets'][i] = \
            targets_orig.variables['targets'][i_rand]


def main(inargs):
    np.random.seed(inargs.random_seed)

    # Open original files
    print('Reading files:', inargs.pref + '_features.nc',
          inargs.pref + '_targets.nc')
    features_orig = Dataset(inargs.pref + '_features.nc', 'r')
    targets_orig = Dataset(inargs.pref + '_targets.nc', 'r')

    # Create equivalent new files
    print('Creating files:', inargs.pref + '_shuffle_features.nc',
          inargs.pref + '_shuffle_targets.nc')
    features_shuffle = Dataset(inargs.pref + '_shuffle_features.nc', 'w')
    targets_shuffle = Dataset(inargs.pref + '_shuffle_targets.nc', 'w')
    for orig, shuffle in zip([features_orig, targets_orig],
                             [features_shuffle, targets_shuffle]):
        for dim_name, dim in orig.dimensions.items():
            shuffle.createDimension(dim_name, dim.size)
        for var_name, var in orig.variables.items():
            shuffle.createVariable(var_name, var.dtype, var.dimensions)
            if var_name not in ['features', 'targets']:
                shuffle.variables[var_name][:] = var[:]
        shuffle.log = ('Original log:\n' + orig.log + '\nNew log:\n' +
                       create_log_str())

    if inargs.method == 'meticulous':
        meticulous(features_orig, targets_orig, features_shuffle,
                   targets_shuffle)
    elif inargs.method == 'fast':
        fast(features_orig, targets_orig, features_shuffle, targets_shuffle,
             inargs.chunk_size)
    else:
        raise Exception

    features_shuffle.close()
    targets_shuffle.close()
    features_orig.close()
    targets_orig.close()


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--method',
                   type=str,
                   default='fast',
                   help='[Meticulous or fast]')
    p.add_argument('--pref',
                   type=str,
                   help='Prefix. ie without the _features.nc')
    p.add_argument('--random_seed',
                   type=int,
                   default=42,
                   help='Random seed for shuffling of data.')
    p.add_argument('--chunk_size',
                   type=int,
                   default=10_000_000,
                   help='Size of chunks for fast method')
    p.add_argument('--verbose',
                   type=int,
                   default=0,
                   help='Verbosity level')
    args = p.parse_args()

    verbose = args.verbose

    main(args)
