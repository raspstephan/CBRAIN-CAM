"""
This file contains functions used in the main preprocessing script.

Created on 2019-01-23-14-50
Author: Stephan Rasp, raspstephan@gmail.com
"""

from ..imports import *
from ..cam_constants import *

# Set up logging, mainly to get timings easily.
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


# Define dictionary with vertical diffusion terms
diff_dict = {
    'TAP' : 'DTV',
    'QAP' : 'VD01'
}


def compute_bp(ds, var):
    """GCM state at beginning of time step before physics.
    ?BP = ?AP - physical tendency * dt

    Args:
        ds: entire xarray dataset
        var: BP variable name

    Returns:
        bp: xarray dataarray containing just BP variable, with the first time step cut.
    """
    base_var = var[:-2] + 'AP'
    return (ds[base_var] - ds[phy_dict[base_var]] * DT)[1:]  # Not the first time step


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
        c -= ds[diff_dict[base_var]].isel(time=slice(0, -1, 1)) * DT
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
    return (compute_bp(ds, base_var) - compute_c(ds, base_var)) / DT


def create_stacked_da(ds, vars):
    """
    In this function the derived variables are computed and the right time steps are selected.

    Parameters
    ----------
    ds: mf_dataset with dimensions [time, lev, lat, lon]
    vars: list of input and output variables

    Returns
    -------
    da: dataarray with variables [vars, var_names]
    """
    var_list, names_list = [], []
    for var in vars:
        if 'BP' in var:
            da = compute_bp(ds, var)
        elif var in ['LHFLX', 'SHFLX']:
            da = ds[var][:-1]
        elif var == 'PRECST':
            da = (ds['PRECSC'] + ds['PRECSL'])[1:]
        elif 'dt_adiabatic' in var:
            base_var = var[:-12] + 'AP'
            da = compute_adiabatic(ds, base_var)
        else:
            da = ds[var][1:]
        var_list.append(da)
        nlev = da.lev.size if 'lev' in da.coords else 1
        names_list.extend([var] * nlev)

    concat_da = rename_time_lev_and_cut_times(ds, var_list)

    # Delete unused coordinates and set var_names as coordinates
    concat_da['var_names'] = np.array(names_list).astype('object')
    #names_da = xr.DataArray(names_list, coords=[concat_da.coords['stacked']])
    a = 3
    return concat_da


def rename_time_lev_and_cut_times(ds, da_list):
    """Create new time and lev coordinates and cut times for non-cont steps
    This is a bit of a legacy function. Should probably be revised.

    Args:
        ds: Merged dataset
        da_list: list of dataarrays

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
    da = xr.concat(da_list, dim='lev')
    # Cut out time steps
    cut_time_steps = np.where(np.abs(np.diff(ds.time)) > 2.09e-2)[0]
    clean_time_steps = np.array(da.coords['time'])
    print('These time steps are cut:', cut_time_steps)
    clean_time_steps = np.delete(clean_time_steps, cut_time_steps)
    da = da.isel(time=clean_time_steps)
    # Rename
    da = da.rename({'lev': 'var_names'})
    da = da.rename('vars')

    return da


def reshape_da(da):
    """

    Parameters
    ----------
    da: dataarray with [time, stacked, lat, lon]

    Returns
    -------
    da: dataarray with [sample, stacked]
    """
    da = da.stack(sample=('time', 'lat', 'lon'))
    return da.transpose('sample', 'var_names')


def preprocess(in_dir, in_fns, out_dir, out_fn, vars, lev_range=(0, 30)):
    """
    This is the main script that preprocesses one file.

    Returns
    -------

    """
    in_fns = path.join(in_dir, in_fns)
    out_fn = path.join(out_dir, out_fn)
    logging.debug(f'Start preprocessing file {out_fn}')

    logging.info('Reading input files')
    ds = xr.open_mfdataset(in_fns, decode_times=False, decode_cf=False, concat_dim='time')

    logging.info('Crop levels')
    ds = ds.isel(lev=slice(*lev_range, 1))

    logging.info('Create stacked dataarray')
    da = create_stacked_da(ds, vars)

    logging.info('Stack and reshape dataarray')
    da = reshape_da(da).reset_index('sample')

    logging.info(f'Save dataarray as {out_fn}')
    da.to_netcdf(out_fn)

    logging.info('Done!')


if __name__ == '__main__':
    fire.Fire(preprocess)

