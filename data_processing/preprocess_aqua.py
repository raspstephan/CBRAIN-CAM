"""Script to read the raw aquaplanet nc files and write one nc file to use
in the neural network scripts.

Author: Stephan Rasp

TODO:
- Create Lat variable
- write mean and std files
- Add select_lat option
"""
import glob
from argparse import ArgumentParser
from netCDF4 import Dataset
import os


var_list = ['TAP', 'SHFLX']


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
    rg.createDimension('time', sample_rg.dimensions['time'].size * n_infiles)
    rg.createVariable('time', inargs.dtype, 'time')
    rg.createDimension('lev',
                       sample_rg.variables['lev'][inargs.min_lev:].shape[0])
    v = rg.createVariable('lev', inargs.dtype, 'lev')
    v[:] = sample_rg.variables['lev'][inargs.min_lev:]
    for name in ['lat', 'lon']:
        rg.createDimension(name, sample_rg.dimensions[name].size)
        v = rg.createVariable(name, inargs.dtype, name)
        v.long_name = sample_rg.variables[name].long_name
        v.units = sample_rg.variables[name].units
        # Write values
        v[:] = sample_rg.variables[name][:]

    # Create all other variables
    for var in var_list:
        v = rg.createVariable(var, inargs.dtype,
                              sample_rg.variables[var].dimensions)
        v.long_name = sample_rg.variables[var].long_name
        v.units = sample_rg.variables[var].units

    if inargs.verbose: print('Created out_file:', rg)
    return rg


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
        new_time_idx = time_idx + aqua_rg.dimensions['time'].size
        if inargs.verbose: print('Current time_idx:', time_idx,
                                 'New time_idx:', new_time_idx)
        for var in var_list:
            if aqua_rg.variables[var].ndim == 3:
                rg.variables[var][time_idx:new_time_idx] = \
                    aqua_rg.variables[var][:, :, :]
            elif aqua_rg.variables[var].ndim == 4:
                rg.variables[var][time_idx:new_time_idx] = \
                    aqua_rg.variables[var][:, inargs.min_lev:, :, :]
            else:
                raise ValueError('Wrong dimensions.')

    return new_time_idx


def main(inargs):
    """Main function. Takes arguments and executes preprocessing routines.
    """

    # Create list of all input data files
    in_list = sorted(glob.glob(inargs.in_dir + inargs.aqua_pref + '*'))
    if inargs.verbose: print('Input file list:', in_list)

    # Read the first input file to get basic parameters
    sample_rg = Dataset(in_list[0], 'r')

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
