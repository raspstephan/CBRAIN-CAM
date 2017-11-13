"""Script to read the raw aquaplanet nc files and write one nc file to use
in the neural network scripts.

Author: Stephan Rasp

TODO:
- Add moisture convergence
"""
import glob
from configargparse import ArgParser
from netCDF4 import Dataset
import os, sys
import numpy as np
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
    rg.log = create_log_str()

    # Create dimensions [date, time, lev, lat, lon]
    # and dimension variables
    rg.createDimension('date', n_infiles)
    v = rg.createVariable('date', inargs.dtype, 'date')
    v[:] = np.arange(1, n_infiles + 1)
    # Time (subtract one time step)
    rg.createDimension('time',(sample_rg.dimensions['time'].size - 1))
    v = rg.createVariable('time', inargs.dtype, 'time')
    v[:] = np.arange(v.shape[0])
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
    for var in inargs.vars:
        if var in sample_rg.variables.keys():
            dims = ['date'] + list(sample_rg.variables[var].dimensions)
            v = rg.createVariable(var, inargs.dtype, dims)
            v.long_name = sample_rg.variables[var].long_name
            v.units = sample_rg.variables[var].units
        elif var == 'LAT':
            add_LAT(inargs, rg)
        elif var in ['dTdt_adiabatic', 'dQdt_adiabatic']:
            tmp_var = 'TAP'
            dims = ['date'] + list(sample_rg.variables[tmp_var].dimensions)
            v = rg.createVariable(var, inargs.dtype, dims)
            if var == 'dTdt_adiabatic':
                v.long_name = 'Adiabatic T tendency'
                v.units = 'K/s'
            else:
                v.long_name = 'Adiabatic Q tendency'
                v.units = 'kg/kg/s'
        else:
            KeyError('Variable %s not implemented.' % var)

    if inargs.verbose: print('Created out_file:', rg)
    return rg


def add_LAT(inargs, rg):
    """Adds latitude array with dimensions [date, time, lat, lon]

    Args:
        inargs: Namespace
        rg: output file
    """
    # Create variable
    v = rg.createVariable('LAT', inargs.dtype, ('date', 'time', 'lat', 'lon'))
    v.long_name = 'Latitude'
    v.units = 'degrees'

    # Create LAT array
    lat_1d = rg.variables['lat'][:]
    lat_4d = lat_1d.reshape((1, 1, lat_1d.shape[0], 1))
    lat_4d = np.repeat(lat_4d, v.shape[0], axis=0)   # Date
    lat_4d = np.repeat(lat_4d, v.shape[1], axis=1)   # Time
    lat_4d = np.repeat(lat_4d, v.shape[3], axis=3)   # Lon

    # Write to file
    v[:] = lat_4d


def compute_adiabatic(inargs, var, aqua_rg):
    """Compute adiabatic tendencies.

    Args:
        inargs: Namespace
        var: Variable to be computed
        aqua_rg: input file

    Returns:
        adiabatic: Numpy array
    """
    # Load relevant files
    if var == 'dTdt_adiabatic':
        base_var = 'TAP'
        phy_var = 'TPHYSTND'
    else:
        base_var = 'QAP'
        phy_var = 'PHQ'
    dt = (
        aqua_rg.variables[base_var][1:][:, inargs.min_lev:, inargs.lat_idxs, :] -
        aqua_rg.variables[base_var][:-1][:, inargs.min_lev:, inargs.lat_idxs, :]
    ) / (0.5 * 60 * 60)   # Convert to s-1
    phy = aqua_rg.variables[phy_var][1:][:, inargs.min_lev:, inargs.lat_idxs, :]
    return dt - phy


def write_contents(inargs, rg, aqua_fn, date_idx):
    """Write contents of one aqua file

    Args:
        inargs: argparse namespace
        rg: nc rootgroup to write in
        aqua_fn: filename of aqua file
        date_idx: date_index

    """

    # Open aqua file
    with Dataset(aqua_fn, 'r') as aqua_rg:
        n_time_min_1 = aqua_rg.dimensions['time'].size - 1
        if inargs.verbose: print('date_idx:', date_idx)
        for var in inargs.vars:
            if var in inargs.current_vars:
                var_time_idxs = np.arange(1, 1 + n_time_min_1)
            else:
                var_time_idxs = np.arange(0, 0 + n_time_min_1)
            if inargs.verbose: print('Variable time indices:', var_time_idxs,
                                     'for variable:', var)
            if var in aqua_rg.variables.keys():
                if aqua_rg.variables[var].ndim == 3:
                    # [time, lat, lon]
                    rg.variables[var][date_idx] = \
                        aqua_rg.variables[var][var_time_idxs][:, inargs.lat_idxs,
                                                              :]
                elif aqua_rg.variables[var].ndim == 4:
                    # [time, lev, lat, lon]
                    rg.variables[var][date_idx] = \
                        aqua_rg.variables[var][var_time_idxs][:, inargs.min_lev:,
                                                              inargs.lat_idxs, :]
                else:
                    raise ValueError('Wrong dimensions.')
            elif var == 'LAT':
                pass   # Already dealt with that some other place
            elif var in ['dTdt_adiabatic', 'dQdt_adiabatic']:
                rg.variables[var][date_idx] = \
                    compute_adiabatic(inargs, var, aqua_rg)
            else:
                raise KeyError('Variable %s not implemented.' % var)


def create_mean_std(inargs):
    """Create files with means and standard deviations.
    These have either dimension z or 1

    Args:
        inargs: Namespace

    """
    # File to read from
    full_rg = Dataset(os.path.join(inargs.out_dir, inargs.out_fn), 'r')

    # Loop over mean and std
    for type, fn in zip(['mean', 'std'], [inargs.mean_fn, inargs.std_fn]):
        # Mean and std files
        nc_fn = os.path.join(inargs.out_dir, fn)
        if inargs.verbose: print(type, 'file:', nc_fn)
        rg = Dataset(nc_fn, 'w')
        rg.log = create_log_str()

        # Create levs dimensions
        rg.createDimension('lev', full_rg.dimensions['lev'].size)

        # Loop through all variables
        for var_name in inargs.vars:
            full_v = full_rg.variables[var_name]
            if full_v.ndim == 4:
                v = rg.createVariable(var_name, inargs.dtype, ())
                if type == 'mean':
                    v[:] = np.mean(full_v[:])
                else:
                    v[:] = np.std(full_v[:], ddof=1)

            elif full_v.ndim == 5:
                # [date, time, lev, lat, lon]
                v = rg.createVariable(var_name, inargs.dtype, 'lev')
                if type == 'mean':
                    v[:] = np.mean(full_v[:], axis=(0, 1, 3, 4))
                else:
                    v[:] = np.std(full_v[:], ddof=1, axis=(0, 1, 3, 4))
            else:
                raise ValueError('Wrong dimensions for variable %s.' % var_name)

        rg.close()
    full_rg.close()


def create_flat(inargs):
    """Create flat version of outputs file

    Args:
        inargs: Namespace
    """
    # File to read from
    full_rg = Dataset(os.path.join(inargs.out_dir, inargs.out_fn), 'r')

    # Create new file
    nc_fn = os.path.join(inargs.out_dir, inargs.flat_fn)
    if inargs.verbose: print('Flattened file:', nc_fn)
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
    for var in inargs.vars:
        full_var = full_rg.variables[var]
        if full_var.ndim == 4:
            v = rg.createVariable(var, inargs.dtype, 'sample')
            data = full_var[:]   # [date, time, lat, lon]
            v[:] = np.ravel(data)
        elif full_var.ndim == 5:
            v = rg.createVariable(var, inargs.dtype, ('lev', 'sample'))
            data = full_var[:]   # [date, time, lev, lat, lon]
            data = np.rollaxis(data, 2, 0)  # [lev, date, time, lat, lon]
            v[:] = data.reshape(v.shape)
        else:
            raise ValueError('Wrong dimensions for variable %s.' % var)

        # Add additional information
        v.long_name = full_var.long_name
        v.units = full_var.units


def main(inargs):
    """Main function. Takes arguments and executes preprocessing routines.

    Args:
        inargs: argument namespace
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
    for date_idx, aqua_fn in enumerate(in_list):
        write_contents(inargs, rg, aqua_fn, date_idx)
    rg.close()

    # Create mean and std files
    create_mean_std(inargs)

    if inargs.flatten:
        create_flat(inargs)


if __name__ == '__main__':

    p = ArgParser()
    p.add('--config_file',
          default='config.yml',
          is_config_file=True,
          help='Name of config file in this directory. '
               'Must contain in and out variable lists.')
    p.add_argument('--vars',
                   type=str,
                   nargs='+',
                   help='All variables. Features and targets')
    p.add_argument('--current_vars',
                   type=str,
                   nargs='+',
                   help='Variables to take ffrom current time step.')
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
    p.add_argument('--mean_fn',
                   type=str,
                   default='SPCAM_mean_detailed.nc',
                   help='Filename of mean file. '
                        'Default = "SPCAM_mean_detailed.nc"')
    p.add_argument('--std_fn',
                   type=str,
                   default='SPCAM_std_detailed.nc',
                   help='Filename of std file. '
                        'Default = "SPCAM_std_detailed.nc"')
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
    p.add_argument('--flatten',
                   dest='flatten',
                   action='store_true',
                   help='If given: flatten time, lat and lon in a separate '
                        'file. NOTE: Twice the memory!')
    p.set_defaults(flatten=False)
    p.add_argument('--flat_fn',
                   type=str,
                   default='SPCAM_outputs_flat.nc',
                   help='Filename of flat file. '
                        'Default = "SPCAM_outputs_flat.nc"')
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='If given, print debugging information.')
    p.set_defaults(verbose=False)

    args = p.parse_args()

    # Perform some input checks
    assert len(args.vars[0]) > 0, 'No vars given.'

    main(args)
