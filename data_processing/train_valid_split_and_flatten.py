"""Function to split dataset into train and validation set
and flatten them. Random split only works for flattened files. In this case
the file is flattened first and the split.

For split by lon the files are first split and then flattened if requested.
Note that these train and valid files are not shuffled.

If train_fraction is set to 1 and flatten is True, the input file is only
flattened.

This script always leaves the original file intact.

Author: Stephan Rasp
"""

from argparse import ArgumentParser
from netCDF4 import Dataset
import numpy as np
from preprocess_aqua import create_log_str
import os

# Make train/valid split reproducible
np.random.seed(42)


def create_flat(inargs, full_rg, fn_pref):
    """Create flat version of outputs file

    Args:
        inargs: Namespace
        full_rg: Opened full nc file
        fn_pref: Prefix of file. _flat will be appended
    """

    # Create new file
    nc_fn = os.path.join(inargs.out_dir, fn_pref + '_flat.nc')
    print('Flattened file:', nc_fn)
    rg = Dataset(nc_fn, 'w')
    rg.log = create_log_str()

    # Create dimensions
    n_samples = (full_rg.dimensions['date'].size *
                 full_rg.dimensions['time'].size *
                 full_rg.dimensions['lat'].size *
                 full_rg.dimensions['lon'].size)
    rg.createDimension('lev', full_rg.dimensions['lev'].size)
    rg.createDimension('sample', n_samples)

    # Loop through variables and write flattened arrays
    for name, full_var in full_rg.variables.items():
        if full_var.ndim == 4:
            v = rg.createVariable(name, full_var.dtype, 'sample')
            data = full_var[:]  # [date, time, lat, lon]
            v[:] = np.ravel(data)
        elif full_var.ndim == 5:
            data = full_var[:]  # [date, time, lev, lat, lon]
            if inargs.lev_last:
                data = np.rollaxis(data, 2, 5)  # [lev, date, time, lat, lon]
                v = rg.createVariable(name, full_var.dtype, ('sample', 'lev'))
                v[:] = data.reshape(v.shape)
            else:
                data = np.rollaxis(data, 2, 0)  # [lev, date, time, lat, lon]
                v = rg.createVariable(name, full_var.dtype, ('lev', 'sample'))
                v[:] = data.reshape(v.shape)
        else:
            if inargs.verbose: print('Do not create flat:', name)

        # Add additional information
        try:
            v.long_name = full_var.long_name
            v.units = full_var.units
        except AttributeError:
            pass

    # Close nc file
    full_rg.close()
    return rg, nc_fn


def create_train_valid_nc(inargs, full_rg, train_idxs, valid_idxs, fn_pref):
    """Create train and validation netCDF files

    Args:
        inargs: Namespace
        full_rg: Opened nc file of full dataset
        train_idxs: split indices for training set, either sample or lon
        valid_idxs: split indices for validation set, either sample or lon
        fn_pref: prefix for file name. Appends _train.nc/_valid.nc

    Returns:
        (train_rg, valid_rg), (train_fn, valid_fn): nc objects and filenames
    """
    train_fn = inargs.out_dir + fn_pref + '_train'
    valid_fn = inargs.out_dir + fn_pref + '_valid'
    if inargs.split_by_lon:
        train_fn += '_by_lon.nc'
        valid_fn += '_by_lon.nc'
    else:
        train_fn += '_random.nc'
        valid_fn += '_random.nc'
    print('Train/valid files:', train_fn, valid_fn)
    train_rg = Dataset(train_fn, 'w')
    valid_rg = Dataset(valid_fn, 'w')

    # Loop over train and valid and write data
    for rg, idxs in zip([train_rg, valid_rg], [train_idxs, valid_idxs]):
        rg.log = create_log_str()
        # Create dimensions
        for dim_name, dim in full_rg.dimensions.items():
            if dim_name in ['lon', 'sample']:   # Mutually exclusive I hope
                rg.createDimension(dim_name, len(idxs))
            else:
                rg.createDimension(dim_name, dim.size)

        # Create variables and write split data
        for name, raw_var in full_rg.variables.items():
            if inargs.verbose: print(name)
            var = rg.createVariable(name, raw_var.dtype, raw_var.dimensions)
            try:
                var.long_name = raw_var.long_name
                var.units = raw_var.units
            except AttributeError:
                pass
    return (train_rg, valid_rg), (train_fn, valid_fn)


def create_lon_split(inargs, full_rg):
    """Splits data by longitude.

    Args:
        inargs: Namespace
        full_rg: Opened nc file of full dataset

    Returns:
        rgs, fns: Train/valid nc objects and file names
    """

    # Create longitude split indices
    n_lon = full_rg.dimensions['lon'].size
    idxs = np.arange(n_lon)
    split_idx = int(n_lon * inargs.train_fraction)
    train_idxs = idxs[:split_idx]
    valid_idxs = idxs[split_idx:]

    # Create train and valid datasets
    rgs, fns = create_train_valid_nc(inargs, full_rg, train_idxs, valid_idxs,
                                     inargs.out_pref)

    for rg, idxs in zip(rgs, [train_idxs, valid_idxs]):
        for name, full_var in full_rg.variables.items():
            full_data = full_var[:]
            var = rg.variables[name]
            if len(full_var.dimensions) < 4:
                if name == 'lon':
                    var[:] = full_data[idxs]
                else:
                    var[:] = full_data[:]
            elif len(full_var.dimensions) == 4:
                var[:] = full_data[:, :, :, idxs]
            elif len(full_var.dimensions) == 5:
                var[:] = full_data[:, :, :, :, idxs]
            else:
                raise ValueError('Wrong dimensions.')

    # Close full rg
    full_rg.close()
    return rgs, fns


def create_random_split(inargs, flat_rg):
    """Splits data randomly.

    Args:
        inargs: Namespace
        flat_rg: Opened nc file of full dataset
    """
    # Create random split
    n_samples = flat_rg.dimensions['sample'].size
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)
    split_idx = int(n_samples * inargs.train_fraction)
    train_idxs = idxs[:split_idx]
    valid_idxs = idxs[split_idx:]

    # Create train and valid datasets
    rgs, fns = create_train_valid_nc(inargs, flat_rg, train_idxs, valid_idxs,
                                     inargs.out_pref + '_flat')

    for rg, idxs in zip(rgs, [train_idxs, valid_idxs]):
        for name, full_var in flat_rg.variables.items():
            full_data = full_var[:]
            var = rg.variables[name]
            if len(full_var.dimensions) == 1:
                var[:] = full_data[idxs]
            elif len(full_var.dimensions) == 2:
                var[:] = full_data[:, idxs]
            else:
                raise ValueError('Wrong dimensions.')
        rg.close()
    # Close full rg
    flat_rg.close()


def main(inargs):
    """Main function. Copied from data_conversion notebook

    Args:
        inargs: ArgParse namespace
    """

    # Open full dataset
    full_rg = Dataset(inargs.full_fn)

    # Three options: random split, split_by_lon or train_fraction = 1
    if inargs.train_fraction == 1.:
        assert inargs.flatten, 'No split and no flatten. Nothing to do here...'
        print('Train fraction is 1. No split is performed.')
        # Flatten file
        rg, _ = create_flat(inargs, full_rg, inargs.out_pref)
        rg.close()

    elif inargs.split_by_lon:
        print('Split data by longitude.')

        # Split data
        rgs, fns = create_lon_split(inargs, full_rg)

        if inargs.flatten:
            # Flatten data
            for name, rg, fn in zip(['_train', '_valid'], rgs, fns):
                create_flat(inargs, rg, inargs.out_pref + name + '_by_lon')
                # Remove intermediate files
                if inargs.delete_intermediate:
                    print('Removing:', fn)
                    os.remove(fn)
        else:
            # Close files
            [rg.close() for rg in rgs]

    else:
        assert inargs.flatten, 'Random split only for flattened data.'
        print('Randomly split data')
        # Flatten file
        flat_rg, fn = create_flat(inargs, full_rg, inargs.out_pref)

        # Split file
        create_random_split(inargs, flat_rg)

        # Remove intermediate file.
        if inargs.delete_intermediate:
            print('Removing:', fn)
            os.remove(fn)


if __name__ == '__main__':

    p = ArgumentParser()

    p.add_argument('--full_fn',
                   type=str,
                   help='Full outputs file created by preprocess_aqua.py')
    p.add_argument('--out_dir',
                   type=str,
                   help='Output directory.')
    p.add_argument('--out_pref',
                   type=str,
                   default='SPCAM_outputs',
                   help='Prefix for output files (Default = SPCAM_outputs). '
                        '_train/_valid/_flat will be appended.')
    p.add_argument('--train_fraction',
                   type=float,
                   help='Fraction of data in training set. '
                        'If train_fraction = 1, no split is performed.')
    p.add_argument('--split_by_lon',
                   dest='split_by_lon',
                   action='store_true',
                   help='If given, Split the data by longitude. ')
    p.set_defaults(split_by_lon=False)
    p.add_argument('--flatten',
                   dest='flatten',
                   action='store_true',
                   help='If given: flatten time, lat and lon in a separate '
                        'file. NOTE: Twice the memory!')
    p.set_defaults(flatten=False)
    p.add_argument('--lev_last',
                   dest='lev_last',
                   action='store_true',
                   help='If given: Save flat array as [sample, lev]')
    p.set_defaults(lev_last=False)
    p.add_argument('--delete_intermediate',
                   dest='delete_intermediate',
                   action='store_true',
                   help='If given: Delete intermediate files.')
    p.set_defaults(delete_intermediate=False)
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='If given, print debugging information.')
    p.set_defaults(verbose=False)

    args = p.parse_args()

    main(args)
