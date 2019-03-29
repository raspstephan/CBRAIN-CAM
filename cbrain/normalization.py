"""
This file contains all the normalization classes.

This is tightly linked with the preprocessing.compute_normalization script.

Created on 2019-01-28-10-31
Author: Stephan Rasp, raspstephan@gmail.com
"""

from .imports import *
from .utils import *
from .cam_constants import *

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


class Normalizer(object):
    """Base normalizer class. All normalization classes must have a transform method."""
    def __init__(self):
        self.transform_arrays = None

    def transform(self, x):
        return x


class StandardNormalizer(object):
    """Standard mean-std normalizer"""
    def __init__(self, norm_ds, var_list):
        var_idxs = return_var_idxs(norm_ds, var_list)
        self.mean = norm_ds['mean'].values[var_idxs]
        self.std = norm_ds['std'].values[var_idxs]
        self.transform_arrays = {
            'mean': self.mean,
            'std': self.std
        }

    def transform(self, x):
        return (x - self.mean) / self.std


class MaxRSNormalizer(object):
    """Normalizer that subtracts the mean and then divides by max(range, std_by_var)"""
    def __init__(self, norm_ds, var_list):
        var_idxs = return_var_idxs(norm_ds, var_list)
        self.mean = norm_ds['mean'].values[var_idxs]
        rang = norm_ds['max'][var_idxs] - norm_ds['min'][var_idxs]
        std_by_var = rang.copy()
        for v in var_list:
            std_by_var[std_by_var.var_names == v] = norm_ds['std_by_var'][
                norm_ds.var_names_single == v]
        self.maxrs = np.maximum(rang, std_by_var).values
        self.transform_arrays = {
            'mean': self.mean,
            'maxrs': self.maxrs
        }

    def transform(self, x):
        return (x - self.mean) / self.maxrs


class DictNormalizer(object):
    """Normalizer that takes a conversion dictionary as input. Simply scales by factors in dict."""
    def __init__(self, norm_ds, var_list, dic=None):
        if dic is None: dic = conversion_dict
        var_idxs = return_var_idxs(norm_ds, var_list)
        var_names = norm_ds.var_names[var_idxs].copy()
        scale = []
        for v in var_list:
            s = np.atleast_1d(dic[v])
            ns = len(s)
            nv = np.sum(var_names == v)
            if ns == nv:
                scale.append(s)
            else:
                scale.append(np.repeat(s, nv))
        self.scale = np.concatenate(scale).astype('float32')
        self.transform_arrays = {
            'scale': self.scale,
        }

    def transform(self, x):
        return x * self.scale

    def inverse_transform(self, x):
        return x / self.scale
