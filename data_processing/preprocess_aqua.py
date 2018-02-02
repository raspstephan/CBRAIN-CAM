"""Script to read the raw aquaplanet nc files and write one nc file to use
in the neural network scripts.

Author: Stephan Rasp

TODO:
- Add moisture convergence
- Add option for LAT
- Read convesion dict from config file
- Add list of variables to log and as variable in arrays
"""
from configargparse import ArgParser
import sys
import numpy as np
from datetime import datetime
from subprocess import getoutput
import xarray as xr
import timeit
import pdb

# Define conversion dict
L_V = 2.5e6   # Latent heat of vaporization is actually 2.26e6
C_P = 1e3   # Specific heat capacity of air at constant pressure
conversion_dict = {
    'SPDT': C_P,
    'SPDQ': L_V,
    'QRL': C_P,
    'QRS': C_P,
    'PRECT': 1e3*24*3600 * 1e-3,
    'FLUT': 1. * 1e-5,
}


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


def crop_ds(inargs, ds):
    """Crops dataset in lat and lev dimensions

    Args:
        inargs: namespace
        ds: Dataset
    Stores:
        ds: Cropped Dataset
    """
    lat_idxs = np.where(
        (ds.coords['lat'].values >= inargs.lat_range[0]) &
        (ds.coords['lat'].values <= inargs.lat_range[1])
    )[0]
    lev_idxs = np.arange(inargs.min_lev, ds.coords['lev'].size)
    if inargs.verbose:
        print('Latitude indices:', lat_idxs)
        print('Level indices:', lev_idxs)
    return ds.isel(lat=lat_idxs, lev=lev_idxs)


def compute_adiabatic(ds, var):
    """Compute adiabatic tendencies.

    Args:
        ds: xarray dataset
        var: Variable to be computed

    Returns:
        adiabatic: xarray dataarray
    """
    # Load relevant files
    if var == 'dTdt_adiabatic':
        base_var = 'TAP'
        phy_var = 'TPHYSTND'
    else:
        base_var = 'QAP'
        phy_var = 'PHQ'
    adiabatic = ds[base_var].diff('time', n=1) / (0.5 * 60 * 60) - ds[phy_var]  # Convert to s-1
    return adiabatic


def create_feature_da(ds, feature_vars, min_lev):
    """Create feature dataArray
    
    Args:
        ds: xarray DataSet
        feature_vars: list of feature variables
        min_lev: min lev

    Returns:
        feature_ds: Dataset with feature variables
    """
    # Get list of all feature DataArrays
    features_list = []
    name_list = []
    for var in feature_vars:
        if var == 'dTdt_adiabatic':
            features_list.append(compute_adiabatic(ds, 'dTdt_adiabatic'))
        elif var == 'dQdt_adiabatic':
            features_list.append(compute_adiabatic(ds, 'dQdt_adiabatic'))
        else:
            features_list.append(ds[var][:-1])
        if 'lev' in features_list[-1].coords:
            name_list += [(var + '_lev%02i') % lev
                          for lev in range(min_lev, 30)]
        else:
            name_list += [var]

    return rename_time_lev_and_cut_times(ds, features_list, name_list,
                                         'feature')


def create_target_da(ds, target_vars, min_lev):
    """Create target DataArray

    Args:
        ds: xarray DataSet
        target_vars: list of target variables
        min_lev: min lev

    Returns:
        target_ds: Dataset with feature variables
    """
    # Get list of all target DataArrays
    targets_list = []
    name_list = []
    for var in target_vars:
        targets_list.append(ds[var][1:] * conversion_dict[var])
        if 'lev' in targets_list[-1].coords:
            name_list += [(var + '_lev%02i') % lev
                          for lev in range(min_lev, 30)]
        else:
            name_list += [var]

    return rename_time_lev_and_cut_times(ds, targets_list, name_list, 'target')


def rename_time_lev_and_cut_times(ds, da_list, name_list, feature_or_target):
    """Create new time and lev coordinates and cut times for non-cont steps

    Args:
        ds: Merged dataset
        da_list: list of dataarrays
        name_list: list with variable names
        feature_or_target: str

    Returns:
        da, name_da: concat da and name da
    """

    ilev = 0
    for da in da_list:
        da.coords['time'] = np.arange(da.coords['time'].size)
        if 'lev' in da.coords:
            da.coords['lev'] = np.arange(ilev, ilev + da.coords['lev'].size)
            ilev += da.coords['lev'].size
        else:
            da.expand_dims('lev')
            da.coords['lev'] = ilev
            ilev += 1

    # Concatenate
    lev_str = feature_or_target + '_lev'
    da = xr.concat(da_list, dim='lev')
    # Cut out time steps
    cut_time_steps = np.where(np.diff(ds.time) > 2.09e-2)[0]
    clean_time_steps = np.array(da.coords['time'])
    print('Cut time steps:', cut_time_steps)
    clean_time_steps = np.delete(clean_time_steps, cut_time_steps)
    da = da.isel(time=clean_time_steps)
    # Rename
    da = da.rename({'lev': lev_str})
    da = da.rename('targets')
    name_da = xr.DataArray(name_list, coords=[da.coords[lev_str]])

    return da, name_da


def reshape_da(da):
    """Reshape from [time, lev, lat, lon] to [sample, lev]

    Args:
        da: xarray DataArray
    Returns:
        da: reshaped dataArray
    """
    da = da.stack(sample=('time', 'lat', 'lon'))
    if 'feature_lev' in da.coords:
        da = da.transpose('sample', 'feature_lev')
    elif 'target_lev' in da.coords:
        da = da.transpose('sample', 'target_lev')
    else:
        raise Exception
    return da


def normalize_da(feature_da, target_da, log_str, norm_fn=None, ext_norm=None,
                 feature_names=None, target_names=None):
    """Normalize feature arrays
    
    Args:
        feature_ds: feature Dataset
        target_ds: target Dataset
        log_str: log string
        norm_fn: Name of normalization file to be saved, only if not ext_norm
        ext_norm: Path to external normalization file

    Returns:
        da: Normalized DataArray
    """
    if ext_norm is None:
        print('Compute means and stds')
        feature_means = feature_da.mean(axis=0)
        feature_stds = feature_da.std(axis=0)
        target_means = target_da.mean(axis=0)
        target_stds = target_da.std(axis=0)
        #target_01 = target_da.quantile(0.01, dim='sample')
        #target_99 = target_da.quantile(0.99, dim='sample')
        feature_names = feature_names
        target_names = target_names

        # Store means and variables
        norm_ds = xr.Dataset({
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'target_means': target_means,
            'target_stds': target_stds,
            #'target_01': target_01,
            #'target_99': target_99,
            'feature_names': feature_names,
            'target_names': target_names,
        })
        norm_ds.attrs['log'] = log_str
        norm_ds.to_netcdf(norm_fn)
        norm_ds.close()
        norm_ds = xr.open_dataset(norm_fn)
    else:
        print('Load external normalization file')
        norm_ds = xr.open_dataset(ext_norm).load()

    feature_da = ((feature_da - norm_ds['feature_means']) /
                  norm_ds['feature_stds'])
    return feature_da


def shuffle_da(feature_da, target_da, seed):
    """Shuffle indices and sort

    Args:
        feature_da: Feature array
        target_da: Target array
        seed: random seed
    Returns:
        feature_da, target_da: Shuffle DataArrays
    """
    print('Shuffling...')
    # Create random coordinate
    np.random.seed(seed)
    assert feature_da.coords['sample'].size == target_da.coords['sample'].size,\
        'Something is wrong...'
    rand_idxs = np.arange(feature_da.coords['sample'].size)
    np.random.shuffle(rand_idxs)

    feature_da.coords['sample'] = rand_idxs
    target_da.coords['sample'] = rand_idxs

    # Sort
    feature_da = feature_da.sortby('sample')
    target_da = target_da.sortby('sample')

    return feature_da, target_da


def rechunk_da(da, sample_chunks=100000):
    """

    Args:
        da: xarray DataArray
        sample_chunks:  Chunk size in sample dimensions
    Returns:
        da: xarray DataArray rechunked
    """
    lev_str = [s for s in list(da.coords) if 'lev' in s][0]
    return da.chunk({'sample': sample_chunks, lev_str: da.coords[lev_str].size})


def main(inargs):
    """Main function. Takes arguments and executes preprocessing routines.

    Args:
        inargs: argument namespace
    """

    t1 = timeit.default_timer()
    # Create log string
    log_str = create_log_str()

    # Load dataset
    merged_ds = xr.open_mfdataset(inargs.in_dir + inargs.aqua_names,
                                  decode_times=False)
    print('Number of time steps:', merged_ds.coords['time'].size)
    # Crop levels and latitude range
    merged_ds = crop_ds(inargs, merged_ds)

    # Create stacked feature and target datasets
    feature_da, feature_names = create_feature_da(merged_ds,
                                                  inargs.feature_vars,
                                                  inargs.min_lev)
    target_da, target_names = create_target_da(merged_ds,
                                               inargs.target_vars,
                                               inargs.min_lev)

    # Reshape
    feature_da = reshape_da(feature_da)
    target_da = reshape_da(target_da)

    # Rechunk 1, not sure if this is good or necessary
    feature_da = rechunk_da(feature_da)
    target_da = rechunk_da(target_da)

    # Normalize features
    norm_fn = inargs.out_dir + inargs.out_pref + '_norm.nc'
    feature_da = normalize_da(feature_da, target_da, log_str, norm_fn,
                              inargs.ext_norm, feature_names, target_names)

    if not inargs.only_norm:
        # Shuffle along sample dimension
        if inargs.shuffle:
            print('WARNING!!! '
                  'For large files this will consume all your memory. '
                  'Use shuffle_ds.py instead!')
            feature_da, target_da = shuffle_da(feature_da, target_da,
                                               inargs.random_seed)
        else:   # Need to reset indices for some reason
            feature_da = feature_da.reset_index('sample')
            target_da = target_da.reset_index('sample')

        # Rechunk 2, not sure if this is good or necessary at all...
        feature_da = rechunk_da(feature_da)
        target_da = rechunk_da(target_da)

        # Convert to Datasets
        feature_ds = xr.Dataset({'features': feature_da},
                                {'feature_names': feature_names})
        target_ds = xr.Dataset({'targets': target_da,
                                'target_names': target_names})

        # Save data arrays
        feature_ds.attrs['log'] = log_str
        target_ds.attrs['log'] = log_str
        feature_fn = inargs.out_dir + inargs.out_pref + '_features.nc'
        target_fn = inargs.out_dir + inargs.out_pref + '_targets.nc'
        print('Save features:', feature_fn)
        feature_ds.to_netcdf(feature_fn)
        print('Save targets:', target_fn)
        target_ds.to_netcdf(target_fn)

    t2 = timeit.default_timer()
    print('Total time: %.2f s' % (t2 - t1))


if __name__ == '__main__':

    p = ArgParser()
    p.add('--config_file',
          default='config.yml',
          is_config_file=True,
          help='Name of config file in this directory. '
               'Must contain feature and target variable lists.')
    p.add_argument('--feature_vars',
                   type=str,
                   nargs='+',
                   help='All variables. Features and targets')
    p.add_argument('--target_vars',
                   type=str,
                   nargs='+',
                   help='Variables to take ffrom current time step.')
    p.add_argument('--in_dir',
                   type=str,
                   help='Directory with input (aqua) files.')
    p.add_argument('--out_dir',
                   type=str,
                   help='Directory to write preprocessed file.')
    p.add_argument('--aqua_names',
                   type=str,
                   default='AndKua_aqua_*',
                   help='String with filenames to be processed. '
                        'Default = "AndKua_aqua_*"')
    p.add_argument('--out_pref',
                   type=str,
                   default='test',
                   help='Prefix for all file names')
    p.add_argument('--ext_norm',
                   type=str,
                   default=None,
                   help='Name of external normalization file')
    p.add_argument('--min_lev',
                   type=int,
                   default=9,
                   help='Minimum level index. Default = 9')
    p.add_argument('--lat_range',
                   type=int,
                   nargs='+',
                   default=[-90, 90],
                   help='Latitude range. Default = [-90, 90]')
    p.add_argument('--target_factor',
                   type=float,
                   default=1.,
                   help='Factor to multiply targets with. For TF comparison '
                        ' set to 1e-3. Default = 1.')
    p.add_argument('--random_seed',
                   type=int,
                   default=42,
                   help='Random seed for shuffling of data.')
    p.add_argument('--shuffle',
                   dest='shuffle',
                   action='store_true',
                   help='If given, shuffle data along sample dimension.')
    p.set_defaults(shuffle=False)
    p.add_argument('--only_norm',
                   dest='only_norm',
                   action='store_true',
                   help='If given, Only compute and save normalization file.')
    p.set_defaults(only_norm=False)
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='If given, print debugging information.')
    p.set_defaults(verbose=False)

    args = p.parse_args()

    main(args)
