"""
Helper functions that are used throughout cbrain

Created on 2019-01-28-10-33
Author: Stephan Rasp, raspstephan@gmail.com
"""
from .imports import *


def return_var_bool(ds, var_list):
    """
    To be used on stacked variable dimension. Returns bool array.

    Parameters
    ----------
    ds: xarray dataset
    var_list: list of variables

    Returns
    -------
    var_bool: bool array. True where any of var_list is True.

    """
    var_bool = ds.var_names == var_list[0]
    for v in var_list[1:]:
        var_bool = np.bitwise_or(var_bool, ds.var_names == v)
    return var_bool


def return_var_idxs(ds, var_list, var_cut_off=None):
    """
    To be used on stacked variable dimension. Returns indices array

    Parameters
    ----------
    ds: xarray dataset
    var_list: list of variables

    Returns
    -------
    var_idxs: indices array

    """
    if var_cut_off is None:
        var_idxs = np.concatenate([np.where(ds.var_names == v)[0] for v in var_list])
    else:
        idxs_list = []
        for v in var_list:
            i = np.where(ds.var_names == v)[0]
            if v in var_cut_off.keys():
                i = i[var_cut_off[v]:]
            idxs_list.append(i)
        var_idxs = np.concatenate(idxs_list)
    return var_idxs


def save_pickle(fn, obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fn):
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    return obj
