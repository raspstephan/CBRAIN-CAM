"""

Created on 2019-01-23-16-20
Author: Stephan Rasp, raspstephan@gmail.com
"""

from ..imports import *

logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


def compute_standard_normalization(data_ds, norm_ds):
    """
    Mean, std, min, max

    Parameters
    ----------
    data_ds
    norm_ds

    Returns
    -------

    """
    # By level
    norm_ds['mean'] = data_ds.vars.mean(dim='sample')
    norm_ds['std'] = data_ds.vars.std(dim='sample')
    norm_ds['min'] = data_ds.vars.min(dim='sample')
    norm_ds['max'] = data_ds.vars.max(dim='sample')

    # Std by variable
    var_list = list(OrderedDict.fromkeys(data_ds.var_names.values))
    std_by_var = []
    for var in var_list:
        std_by_var.append(data_ds.vars[:, data_ds.var_names == var].std().values)
    norm_ds['std_by_var'] = xr.DataArray(
        std_by_var, coords={'var_names_single': np.array(var_list).astype('object')},
        dims=['var_names_single']
    )
    return norm_ds


# def write_output_normalizations(norm_ds, out_dict):
#     """"""
#     pass


def normalize(dir, in_fn, norm_fn, norm_type='standard'):   #, out_dict=None):

    logging.info('Start normalizing by creating dataset.')
    in_fn = path.join(dir, in_fn)
    norm_fn = path.join(dir, norm_fn)
    data_ds = xr.open_dataset(in_fn)
    norm_ds = xr.Dataset()

    if 'standard' in norm_type:
        logging.info('Compute standard normalizations.')
        compute_standard_normalization(data_ds, norm_ds)

    # if out_dict is not None:
    #     logging.info('Write output normalizations from dictionary.')
    #     write_output_normalizations(norm_ds, out_dict)

    logging.info(f'Saving normalization file as {norm_fn}')
    norm_ds.to_netcdf(norm_fn)

    logging.info('Done!')


if __name__ == '__main__':
    fire.Fire(normalize)
