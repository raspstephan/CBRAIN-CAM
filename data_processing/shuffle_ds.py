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


def create_log_str():
    """Create a log string to add to the netcdf file for reproducibility.
    See: https://raspstephan.github.io/2017/08/24/reproducibility-hack.html

    Returns:
        log_str: String with reproducibility information
    """
    time_stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    pwd = getoutput(['pwd']).rstrip()  # Need to remove trailing /n
    try:
        from git import Repo
        git_dir = pwd.rsplit('/', 1)[0]
        git_hash = Repo(git_dir).heads[0].commit
    except ModuleNotFoundError:
        print('GitPython not found. Please install for better reproducibility.')
        git_hash = 'N/A'
    exe_str = ' '.join(sys.argv)

    log_str = ("""
    Time: %s\n
    Executed command:\n
    python %s\n
    In directory: %s\n
    Git hash: %s\n
        """ % (time_stamp, exe_str, pwd, str(git_hash)))
    return log_str


def main(inargs):
    # Open original files
    print('Reading files:', inargs.pref + '_features.nc',
          inargs.pref + '_targets.nc')
    features_orig = Dataset(inargs.pref + '_features.nc', 'r')
    targets_orig = Dataset(inargs.pref + '_targets.nc', 'r')

    # Create random indices
    np.random.seed(inargs.random_seed)
    rand_idxs = np.arange(features_orig.dimensions['sample'].size)
    np.random.shuffle(rand_idxs)

    # Create equivalent new files
    print('Creating files:', inargs.pref + '_features_shuffle.nc',
          inargs.pref + '_targets_shuffle.nc')
    features_shuffle = Dataset(inargs.pref + '_features_shuffle.nc', 'w')
    targets_shuffle = Dataset(inargs.pref + '_targets_shuffle.nc', 'w')
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

    # Write data in random order
    for i in tqdm(range(rand_idxs.shape[0])):
        i_rand = rand_idxs[i]
        features_shuffle.variables['features'][i] = \
            features_orig.variables['features'][i_rand]
        targets_shuffle.variables['targets'][i] = \
            targets_orig.variables['targets'][i_rand]


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--pref',
                   type=str,
                   help='Prefix. ie without the _features.nc')
    p.add_argument('--random_seed',
                   type=int,
                   default=42,
                   help='Random seed for shuffling of data.')
    args = p.parse_args()
    main(args)