"""Function to split dataset into train and validation set.
Currently only works for flattened files.

Author: Stephan Rasp
"""

from argparse import ArgumentParser
from netCDF4 import Dataset
import numpy as np
from datetime import datetime
import sys
from subprocess import getoutput
from preprocess_aqua import create_log_str

# Make train/valid split reproducible
np.random.seed(42)


def main(inargs):
    """Main function. Copied from data_conversion notebook

    Args:
        inargs: ArgParse namespace

    """
    # Open full dataset
    raw_out = Dataset(inargs.raw_fn)
    n_samples = raw_out.dimensions['sample'].size

    # Create indexes for train and valid set
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)
    split_idx = int(n_samples * inargs.train_fraction)
    train_idxs = idxs[:split_idx]
    valid_idxs = idxs[split_idx:]

    # Create train and valid datasets
    train_out = Dataset(inargs.out_dir + inargs.out_pref + '_train.nc', 'w')
    valid_out = Dataset(inargs.out_dir + inargs.out_pref + '_valid.nc', 'w')

    # Loop over train and valid and write data
    for out, idxs in zip([train_out, valid_out], [train_idxs, valid_idxs]):
        out.log = create_log_str()
        # Create dimensions
        out.createDimension('lev', 21)
        out.createDimension('sample', idxs.shape[0])

        # Create variables and write split data
        for name, raw_var in raw_out.variables.items():
            print(name)
            raw_data = raw_var[:]
            var = out.createVariable(name, raw_var.dtype, raw_var.dimensions)
            var.long_name = raw_var.long_name
            var.units = raw_var.units
            if len(raw_var.dimensions) == 1:
                var[:] = raw_data[idxs]
            elif len(raw_var.dimensions) == 2:
                var[:] = raw_data[:, idxs]
            else:
                raise ValueError('Wrong dimensions.')
        # Close rootgroups
        out.close()
    raw_out.close()


if __name__ == '__main__':

    p = ArgumentParser()

    p.add_argument('--raw_fn',
                   type=str,
                   help='Full outputs file.')
    p.add_argument('--out_dir',
                   type=str,
                   help='Output directory.')
    p.add_argument('--out_pref',
                   type=str,
                   help='Prefix for output files.')
    p.add_argument('--train_fraction',
                   type=float,
                   help='Fraction of data in training set.')

    args = p.parse_args()

    main(args)