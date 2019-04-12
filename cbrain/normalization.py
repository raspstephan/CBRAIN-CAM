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


class InputNormalizer(object):
    """Normalizer that subtracts and then divides."""
    def __init__(self, norm_ds, var_list, sub='mean', div='std_by_var', var_cut_off=None):
        var_idxs = return_var_idxs(norm_ds, var_list, var_cut_off)
        self.sub = norm_ds[sub].values[var_idxs]
        if div == 'maxrs':
            rang = norm_ds['max'][var_idxs] - norm_ds['min'][var_idxs]
            std_by_var = rang.copy()
            for v in var_list:
                std_by_var[std_by_var.var_names == v] = norm_ds['std_by_var'][
                    norm_ds.var_names_single == v]
            self.div = np.maximum(rang, std_by_var).values
        elif div == 'std_by_var':
            # SR: Total mess. Should be handled better
            tmp_var_names = norm_ds.var_names[var_idxs]
            self.div = np.zeros(len(tmp_var_names))
            for v in var_list:
                std_by_var = norm_ds['std_by_var'][norm_ds.var_names_single == v]
                self.div[tmp_var_names == v] = std_by_var
        else:
            self.div = norm_ds[div].values[var_idxs]
        self.transform_arrays = {
            'sub': self.sub,
            'div': self.div
        }

    def transform(self, x):
        return (x - self.sub) / self.div

    def inverse_transform(self, x):
        return (x * self.div) + self.sub


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

