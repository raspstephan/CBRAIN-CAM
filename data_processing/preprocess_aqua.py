"""Script to read the raw aquaplanet nc files and write one nc file to use
in the neural network scripts.

Author: Stephan Rasp

TODO:
- write mean and std files
- compute adiabatic dt variables
- Create config file for input and output variables
- Create log file
"""
import glob
from argparse import ArgumentParser
from netCDF4 import Dataset
import os
import numpy as np


in_var_list = ['TAP', 'SHFLX', 'LAT']   # Inputs/features
out_var_list = ['SPDT']          # Outputs/targets
var_list = in_var_list + out_var_list


def store_lat_idxs(inargs, sample_rg):
    """Stores latitude indices

    Args:
        inargs: namespace

    Stores:
        inargs.lat_idxs: boolian numpy array with lat dimensions
    """
    inargs.lat_idxs = np.where(
        (sample_rg.variables['lat'][:] >= inargs.lat_range[0]) &
        (sample_rg.variables['lat'][:] <= inargs.lat_range[1])
    )[0]
    if inargs.verbose: print('Latitude indices:', inargs.lat_idxs)


def create_nc(inargs, sample_rg, n_infiles):
    """Create new netcdf file.

    Args:
        inargs: ArgParse namespace
        sample_rg: Example aqua nc file to get dimensions
        n_infiles: Number of aqua files

    Returns:
        rg: NetCDF rootgroup object
    """

    # Create file
    nc_fn = os.path.join(inargs.out_dir, inargs.out_fn)
    if inargs.verbose: print('Preprocessed file:', nc_fn)
    rg = Dataset(nc_fn, 'w')

    # Create dimensions [time, lev, lat, lon]
    # and dimension variables
    # Time (subtract one time step=
    rg.createDimension('time',
                       (sample_rg.dimensions['time'].size - 1) * n_infiles)
    rg.createVariable('time', inargs.dtype, 'time')
    # Level (potentially omit top layers)
    rg.createDimension('lev',
                       sample_rg.variables['lev'][inargs.min_lev:].shape[0])
    v = rg.createVariable('lev', inargs.dtype, 'lev')
    v[:] = sample_rg.variables['lev'][inargs.min_lev:]
    # Latitude (potentially chose latitude range)
    rg.createDimension('lat',
                       sample_rg.variables['lat'][inargs.lat_idxs].shape[0])
    v = rg.createVariable('lat', inargs.dtype, 'lat')
    v[:] = sample_rg.variables['lat'][inargs.lat_idxs]
    # Longitude (nothing special)
    rg.createDimension('lon', sample_rg.variables['lon'][:].shape[0])
    v = rg.createVariable('lon', inargs.dtype, 'lon')
    v[:] = sample_rg.variables['lon'][:]

    # Create all other variables
    for var in var_list:
        if var in sample_rg.variables.keys()
            v = rg.createVariable(var, inargs.dtype,
                                  sample_rg.variables[var].dimensions)
            v.long_name = sample_rg.variables[var].long_name
            v.units = sample_rg.variables[var].units
        elif var == 'LAT':
            add_LAT(inargs, rg)
        else:
            KeyError('Variable %s not implemented.' % var)

    if inargs.verbose: print('Created out_file:', rg)
    return rg


def add_LAT(inargs, rg):
    """Adds latitude array with dimensions [time, lat, lon]

    Args:
        inargs: Namespace
        rg: output file
    """
    # Create variable
    v = rg.createVariable('LAT', inargs.dtype, ('time', 'lat', 'lon'))

    # Create LAT array
    lat_1d = rg.variables['lat'][:]
    lat_3d = lat_1d.reshape((1, lat_1d.shape[0], 1))
    lat_3d = np.repeat(lat_3d, v.shape[0], axis=0)   # Time
    lat_3d = np.repeat(lat_3d, v.shape[2], axis=2)   # Lon

    # Write to file
    v[:] = lat_3d



def write_contents(inargs, rg, aqua_fn, time_idx):
    """Write contents of one aqua file

    Args:
        inargs: argparse namespace
        rg: nc rootgroup to write in
        aqua_fn: filename of aqua file
        time_idx: Current time index

    Returns:
        new_time_idx: Updated time index

    """

    # Open aqua file
    with Dataset(aqua_fn, 'r') as aqua_rg:
        n_time_min_1 = aqua_rg.dimensions['time'].size - 1
        new_time_idx = time_idx + n_time_min_1
        if inargs.verbose: print('Current time_idx:', time_idx,
                                 'New time_idx:', new_time_idx)
        for var in var_list:
            if var in in_var_list:
                var_time_idxs = np.arange(0, 0 + n_time_min_1)
            else:
                var_time_idxs = np.arange(1, 1 + n_time_min_1)
            if inargs.verbose: print('Variable time indices:', var_time_idxs,
                                     'for variable:', var)
            if var in aqua_rg.variables.keys():
                if aqua_rg.variables[var].ndim == 3:
                    # [time, lat, lon]
                    rg.variables[var][time_idx:new_time_idx] = \
                        aqua_rg.variables[var][var_time_idxs][:, inargs.lat_idxs, :]
                elif aqua_rg.variables[var].ndim == 4:
                    # [time, lev, lat, lon]
                    rg.variables[var][time_idx:new_time_idx] = \
                        aqua_rg.variables[var][var_time_idxs][:, inargs.min_lev:,
                                                              inargs.lat_idxs, :]
                else:
                    raise ValueError('Wrong dimensions.')
            elif var == 'LAT':
                pass   # Already dealt with that some other place
            else:
                raise KeyError('Variable %s not implemented.' % var)

    return new_time_idx


def main(inargs):
    """Main function. Takes arguments and executes preprocessing routines.
    """

    # Create list of all input data files
    in_list = sorted(glob.glob(inargs.in_dir + inargs.aqua_pref + '*'))
    if inargs.verbose: print('Input file list:', in_list)

    # Read the first input file to get basic parameters
    sample_rg = Dataset(in_list[0], 'r')
    store_lat_idxs(inargs, sample_rg)

    # Allocate new netCDF file
    rg = create_nc(inargs, sample_rg, len(in_list))
    sample_rg.close()

    # Go through files and write contents
    time_idx = 0
    for aqua_fn in in_list:
        time_idx = write_contents(inargs, rg, aqua_fn, time_idx)

    rg.close()

if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('--in_dir',
                   type=str,
                   help='Directory with input (aqua) files.')
    p.add_argument('--out_dir',
                   type=str,
                   help='Directory to write preprocessed file.')
    p.add_argument('--aqua_pref',
                   type=str,
                   default='AndKua_aqua_',
                   help='Prefix of aqua files. Default = "AndKua_aqua_"')
    p.add_argument('--out_fn',
                   type=str,
                   default='SPCAM_outputs_detailed.nc',
                   help='Filename of preprocessed file. '
                        'Default = "SPCAM_outputs_detailed.nc"')
    p.add_argument('--min_lev',
                   type=int,
                   default=9,
                   help='Minimum level index. Default = 9')
    p.add_argument('--lat_range',
                   type=int,
                   nargs='+',
                   default=[-90, 90],
                   help='Latitude range. Default = [-90, 90]')
    p.add_argument('--dtype',
                   type=str,
                   default='float32',
                   help='Datatype of out variables. Default = "float32"')
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='If given, print debugging information.')
    p.set_defaults(verbose=True)

    args = p.parse_args()

    main(args)
