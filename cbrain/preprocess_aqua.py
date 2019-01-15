"""Script to read the raw aquaplanet nc files and write one nc file to use
in the neural network scripts.

Author: Stephan Rasp

TODO:
- Add moisture convergence
- Add option for LAT
- Read convesion dict from config file
- Add list of variables to log and as variable in arrays
"""
import sys
import xarray as xr
import numpy as np
from configargparse import ArgParser
from datetime import datetime
from subprocess import getoutput
import timeit
import pdb

DT = 1800.
L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
P0 = 1e5
conversion_dict = {
    'TPHYSTND': C_P,
    'TPHY_NOKE': C_P,
    'TPHYSTND_NORAD': C_P,
    'PHQ': L_S,
    'PHCLDLIQ' : L_S,
    'PHCLDICE' : L_S,
    'SPDT': C_P,
    'SPDQ': L_V,
    'QRL': C_P,
    'QRS': C_P,
    'PRECT': 1e3*24*3600 * 2e-2,
    'TOT_PRECL': 24*3600 * 2e-2,
    'TOT_PRECS': 24*3600 * 2e-2,
    'PRECS': 1e3*24*3600 * 2e-2,
    'FLUT': 1. * 1e-5,
    'FSNT': 1. * 1e-3,
    'FSDS': -1. * 1e-3,
    'FSNS': -1. * 1e-3,
    'FLNT': -1. * 1e-3,
    'FLNS': 1. * 1e-3,
    'QAP': L_S/DT,
    'QCAP': L_S/DT,
    'QIAP': L_S/DT
}

# Dictionary containing the physical tendencies
phy_dict = {
    'TAP': 'TPHYSTND',
    'QAP': 'PHQ',
    'QCAP': 'PHCLDLIQ',
    'QIAP': 'PHCLDICE',
    'VAP': 'VPHYSTND',
    'UAP': 'UPHYSTND'
}
# Define dictionary with vertical diffusion terms
diff_dict = {
    'TAP' : 'DTV',
    'QAP' : 'VD01'
}
# Define time step
dt_sec = (0.5 * 60 * 60)


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
        repo_name = 'CBRAIN-CAM'
        git_dir = pwd.rsplit(repo_name)[0] + repo_name
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


def compute_bp(ds, base_var):
    """GCM state at beginning of time step before physics.
    ?BP = ?AP - physical tendency * dt
    
    Args:
        ds: xarray dataset
        base_var: Base variable to be computed

    Returns:
        bp: xarray dataarray
    """
    return (ds[base_var] - ds[phy_dict[base_var]] * dt_sec).\
            isel(time=slice(1, None, 1))   # Not the first time step


def compute_c(ds, base_var):
    """CRM state at beginning of time step before physics.
    ?_C = ?AP[t-1] - diffusion[t-1] * dt

    Note:
    compute_c() is the only function that returns data from the previous
    time step.

    Args:
        ds: xarray dataset
        base_var: Base variable to be computed

    Returns:
        c: xarray dataarray
    """
    c = ds[base_var].isel(time=slice(0, -1, 1))   # Not the last time step
    if base_var in diff_dict.keys():
        c -= ds[diff_dict[base_var]].isel(time=slice(0, -1, 1)) * dt_sec
    # Change time coordinate. Necessary for later computation of adiabatic
    c['time'] = ds.isel(time=slice(1, None, 1))['time']
    return c


def compute_adiabatic(ds, base_var):
    """Compute adiabatic tendencies.

    Args:
        ds: xarray dataset
        base_var: Base variable to be computed

    Returns:
        adiabatic: xarray dataarray
    """
    return (compute_bp(ds, base_var) - compute_c(ds, base_var)) / dt_sec


def compute_adiabatic_tf(ds, base_var):
    """Compute adiabatic tendencies as in Pierre's TF version

    Args:
        ds: xarray dataset
        base_var: Base variable to be computed

    Returns:
        adiabatic: xarray dataarray
    """
    adiab = (ds[base_var].diff(dim='time', n=1) / dt_sec - ds[phy_dict[base_var]])
    return adiab


def create_feature_or_target_da(ds, vars, min_lev, feature_or_target, 
                                factor=1., flx_same_dt=False):
    """Create feature or target dataset.

    Args:
        ds: xarray DataSet
        feature_vars: list of feature variables
        min_lev: min lev
        feature_or_target: string
        factor: factor to multiply variables with

    Returns:
        ds: Dataset with variables
    """
    features_list = []
    name_list = []
    for var in vars:
        # Compute derived quantities if necessary
        # I should do this more cleverly with regular experssions...
        if 'dt_adiabatic_tf' in var:
            base_var = var[:-15][1:] + 'AP'
            da = compute_adiabatic_tf(ds, base_var)
        elif 'dt_adiabatic' in var:
            base_var = var[:-12][1:] + 'AP'
            da = compute_adiabatic(ds, base_var)
        elif 'BP' in var:
            base_var = var[:-2] + 'AP'
            da = compute_bp(ds, base_var)
        elif '_C' in var:
            base_var = var[:-2] + 'AP'
            da = compute_c(ds, base_var)
        elif var in ['LHFLX', 'SHFLX'] and not flx_same_dt:   # Take from previous time step
            da = ds[var][:-1]
        elif var == 'TPHYSTND_NORAD':
            da = (ds['TPHYSTND'] - ds['QRL'] - ds['QRS'])[1:]
        elif var == 'TPHY_NOKE':
            da = (ds['TPHYSTND'] - ds['DTVKE']/1800.)[1:]
        elif var == 'TOT_PRECL':
            da = (ds['PRECT']*1e3 + ds['PRECTEND'])[1:]
        elif var == 'TOT_PRECS':
            da = ((ds['PRECSC'] + ds['PRECSL'])*1e3 + ds['PRECTEND'])[1:]
        elif var == 'PRECS':
            da = (ds['PRECSC'] + ds['PRECSL'])[1:]
        else:   # Take from current time step
            da = ds[var][1:]
        # if feature_or_target == 'target':  # we are not normalizing anymore!
        #     da *= conversion_dict[var]
        features_list.append(da * factor)

        # Figure out which name to add
        if 'lev' in features_list[-1].coords:
            name_list += [(var + '_lev%02i') % lev
                          for lev in range(min_lev, 30)]
        else:
            name_list += [var]

    return rename_time_lev_and_cut_times(ds, features_list, name_list,
                                         feature_or_target, flx_same_dt)


def rename_time_lev_and_cut_times(ds, da_list, name_list, feature_or_target, flx_same_dt=False):
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
    if flx_same_dt:
        cut_time_steps = []
    else:
        cut_time_steps = np.where(np.abs(np.diff(ds.time)) > 2.09e-2)[0]
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


def get_feature_idxs(feature_names, var):
    return [
        i for i, s in
        enumerate(list(feature_names.data))
        if var in s]


def normalize_da(feature_da, target_da, log_str, norm_fn=None, ext_norm=None,
                 feature_names=None, target_names=None, norm_targets=None,
                 inputs=None, targets = None, norm_features=None):
    """Normalize feature arrays, and optionally target array
    
    Args:
        feature_da: feature Dataset
        target_da: target Dataset
        log_str: log string
        norm_fn: Name of normalization file to be saved, only if not ext_norm
        ext_norm: Path to external normalization file
        feature_names: Feature name strings
        target_names: target name strings
        norm_targets: If 'norm', regular mean-std normalization, if 'scale'
                     scale between -1 and 1 where

    Returns:
        da: Normalized DataArray
    """
    if ext_norm is None:
        print('Compute means and stds')
        feature_means = feature_da.mean(axis=0, skipna=False)
        feature_stds = feature_da.std(axis=0, skipna=False)
        feature_mins = feature_da.min(axis=0, skipna=False)
        feature_maxs = feature_da.max(axis=0, skipna=False)
        target_means = target_da.mean(axis=0, skipna=False)
        target_stds = target_da.std(axis=0, skipna=False)
        target_mins = target_da.min(axis=0, skipna=False)
        target_maxs = target_da.max(axis=0, skipna=False)
        feature_names = feature_names
        target_names = target_names

        # Create feature da by var
        feature_stds.load()
        feature_stds_by_var = feature_stds.copy(True)   # Deep copy
        for inp in inputs:
            var_idxs = get_feature_idxs(feature_names, inp)
            feature_stds_by_var[var_idxs] = feature_stds[var_idxs].mean()
            
        # Create target energy conversion dictionary
        target_stds.load()
        target_conv = target_stds.copy(True)
        for tar in targets:
            var_idxs = get_feature_idxs(target_names, tar)
            target_conv[var_idxs] = conversion_dict[tar]

        # Store means and variables
        norm_ds = xr.Dataset({
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'feature_mins': feature_mins,
            'feature_maxs': feature_maxs,
            'target_means': target_means,
            'target_stds': target_stds,
            'target_mins': target_mins,
            'target_maxs': target_maxs,
            'feature_names': feature_names,
            'target_names': target_names,
            'feature_stds_by_var': feature_stds_by_var,
            'target_conv': target_conv
        })
        norm_ds.attrs['log'] = log_str
        print('Saving normalization file:', norm_fn)
        norm_ds.to_netcdf(norm_fn)
        norm_ds.close()
        norm_ds = xr.open_dataset(norm_fn)
    else:
        print('Load external normalization file')
        if norm_features is not None:
            norm_ds = xr.open_dataset(ext_norm).load()

    if norm_features == 'by_var':
        feature_da = ((feature_da - norm_ds['feature_means']) /
                      norm_ds['feature_stds_by_var'])
    elif norm_features == 'by_lev':
        feature_da = ((feature_da - norm_ds['feature_means']) /
                      norm_ds['feature_stds'])
    elif norm_features is not None:
        raise Exception('Wrong argument for norm_features')
    if norm_targets == 'norm':
        target_da = ((target_da - norm_ds['target_means']) /
                      norm_ds['target_stds'])
    elif norm_targets == 'scale':
        half_range = (norm_ds['target_maxs'] - norm_ds['target_mins']) / 2
        target_da = (target_da - half_range) / (1.1 * half_range)
    elif norm_targets is not None:
        raise Exception('Wrong argument for norm_targets')

    return feature_da, target_da


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


def rechunk_da(da, sample_chunks):
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
    assert len(inargs.in_dir) == len(inargs.aqua_names), 'Number arguments must match!'
    if len(inargs.in_dir) == 1:
        merged_ds = xr.open_mfdataset(inargs.in_dir[0] + inargs.aqua_names[0],
                                      decode_times=False, decode_cf=False, concat_dim='time')
    else:
        dslist = [xr.open_mfdataset(
            inargs.in_dir[i] + inargs.aqua_names[i],decode_times=False, decode_cf=False)
            for i in range(len(inargs.in_dir))]
        # Change time coordinates
        for i, ds in enumerate(dslist[1:]):
            ds['time'] -= (i+1)*100*366
        # Drop variables
        common = list(set.intersection(*map(set,[list(ds.data_vars) for ds in dslist])))
        for i in range(len(dslist)):
            ds = dslist[i]
            todrop = [v for v in list(ds.data_vars) if v not in common]
            dslist[i] = ds.drop(todrop)
        # Concatenate along time axis
        merged_ds = xr.concat(dslist, 'time')
    print('Time checkpoint reading data: %.2f s' % (timeit.default_timer() - t1))
    print('Number of time steps:', merged_ds.coords['time'].size)
    # Crop levels and latitude range
    merged_ds = crop_ds(inargs, merged_ds)

    # Create stacked feature and target datasets
    feature_da, feature_names = create_feature_or_target_da(
        merged_ds,
        inargs.inputs,
        inargs.min_lev,
        'feature',
        flx_same_dt=inargs.flx_same_dt
    )
    target_da, target_names = create_feature_or_target_da(
        merged_ds,
        inargs.outputs,
        inargs.min_lev,
        'target',
        inargs.target_factor,
        flx_same_dt=inargs.flx_same_dt
    )
    print('Time checkpoint create datasets: %.2f s' % (timeit.default_timer() - t1))

    # Reshape
    feature_da = reshape_da(feature_da)
    target_da = reshape_da(target_da)

    # Rechunk 1, not sure if this is good or necessary
    feature_da = rechunk_da(feature_da, inargs.chunk_size)
    target_da = rechunk_da(target_da, inargs.chunk_size)
    print('Time checkpoint reshape and rechunk: %.2f s' % (timeit.default_timer() - t1))

    # Normalize features
    norm_fn = inargs.out_dir + inargs.out_pref + '_norm.nc'
    feature_da, target_da = normalize_da(
        feature_da, target_da, log_str, norm_fn,inargs.ext_norm, feature_names,
        target_names, inargs.norm_targets, inargs.inputs, inargs.outputs, inargs.norm_features)
    print('Time checkpoint normalization arrays: %.2f s' % (timeit.default_timer() - t1))

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
        feature_da = rechunk_da(feature_da, inargs.chunk_size)
        target_da = rechunk_da(target_da, inargs.chunk_size)

        # Convert to Datasets
        feature_ds = xr.Dataset({'features': feature_da},
                                {'feature_names': feature_names})
        target_ds = xr.Dataset({'targets': target_da,
                                'target_names': target_names})
        print('Time checkpoint rechunk and ds: %.2f s' % (timeit.default_timer() - t1))

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
    p.add_argument('--inputs',
                   type=str,
                   nargs='+',
                   help='Feature variables')
    p.add_argument('--outputs',
                   type=str,
                   nargs='+',
                   help='Target variables')
    p.add_argument('--in_dir',
                   type=str,
                   nargs='+',
                   help='Directory with input (aqua) files.')
    p.add_argument('--out_dir',
                   type=str,
                   help='Directory to write preprocessed file.')
    p.add_argument('--aqua_names',
                   type=str,
                   nargs='+',
                   help='String with filenames to be processed.')
    p.add_argument('--out_pref',
                   type=str,
                   default='test',
                   help='Prefix for all file names')
    p.add_argument('--chunk_size',
                   type=int,
                   default=100_000,
                   help='size of chunks')
    p.add_argument('--ext_norm',
                   type=str,
                   default=None,
                   help='Name of external normalization file')
    p.add_argument('--min_lev',
                   type=int,
                   default=0,
                   help='Minimum level index. Default = 0')
    p.add_argument('--lat_range',
                   type=int,
                   nargs='+',
                   default=[-90, 90],
                   help='Latitude range. Default = [-90, 90]')
    p.add_argument('--target_factor',
                   type=float,
                   default=1.,
                   help='Factor to multiply targets with. For TF comparison '
                        'set to 1e-3. Default = 1.')
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
                   help='If given, only compute and save normalization file.')
    p.set_defaults(only_norm=False)
    p.add_argument('--flx_same_dt',
                   dest='flx_same_dt',
                   action='store_true',
                   help='If given, take surface fluxes from same time step.')
    p.set_defaults(flx_same_dt=False)
    p.add_argument('--norm_features',
                   type=str,
                   default=None,
                   help='by_var or by_lev')
    p.add_argument('--norm_targets',
                   type=str,
                   default=None,
                   help='norm or scale')
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='If given, print debugging information.')
    p.set_defaults(verbose=False)

    args = p.parse_args()

    main(args)
