"""
Contains the ModelDiagnostics class.
TODO:
- Colorbar
- Axes labels
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
import h5py
from tqdm import tqdm
import pandas as pd
import sys, os
sys.path.append('../keras_network/')
sys.path.append('../data_processing/')
from data_generator import DataGenerator
from losses import metrics, all_metrics
from keras.utils.generic_utils import get_custom_objects
metrics_dict = dict([(f.__name__, f) for f in all_metrics])
get_custom_objects().update(metrics_dict)
from preprocess_aqua import L_V, C_P, conversion_dict
import pickle
import pdb

# define global dictionaries and constants
range_dict = {
    'SPDT': [-5e-4, 5e-4],
    'SPDQ': [-5e-7, 5e-7],
    'QRL': [-2e-4, 2e-4],
    'QRS': [-1.2e-4, 1.2e-4],
}
L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_S = L_V + L_I # Sublimation
C_P = 1e3 # Specific heat capacity of air at constant pressure
G = 9.81
P0 = 1e5
with open(os.path.join(os.path.dirname(__file__), 'hyai_hybi.pkl'), 'rb') as f:
    hyai, hybi = pickle.load(f)

class ModelDiagnostics(object):
    """
    Model diagnostics class.
    """
    def __init__(self, model_path, keras_features_fn, keras_targets_fn,
                 keras_norm_fn, nlat=64, nlon=128, nlev=30):
        """
        keras_features [sample, input_z]
        keras_targets [sample, output_z]
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.keras_features_fn = keras_features_fn
        self.keras_targets_fn = keras_targets_fn
        self.keras_norm_fn = keras_norm_fn
        self.keras_features = h5py.File(keras_features_fn, 'r')
        self.keras_targets = h5py.File(keras_targets_fn, 'r')
        self.keras_norm = h5py.File(keras_norm_fn, 'r')
        self.nlat = nlat; self.nlon = nlon; self.nlev = nlev
        self.ngeo = nlat * nlon
        self.feature_vars, self.target_vars = self._get_var_names()
        self.unscale_arr = self._get_unscale_arr()

    def _load_model(self):
        # For keras model
        return keras.models.load_model(self.model_path)

    def _get_var_names(self):
        """
        Return unique variable names for features and targets in correct order.
        """
        return [
            list(dict.fromkeys(
                [f.split('_lev')[0] for f in
                 list(self.keras_norm[f'{a}_names'][:])]
            )) for a in ['feature', 'target']
        ]

    def _get_unscale_arr(self):
        """
        Returns an array of size z_output to unscale the entire output array
        """
        return np.array([
            conversion_dict[v.split('_lev')[0]] for v in
            list(self.keras_norm['target_names'])
        ])

    def plot_double_lat_lev_slice(self, var, itime, ilon, **kwargs):
        # Get predictions and true values. THIS WILL BE DIFFERENT FOR TF
        preds, true = self._get_preds_and_truth(var, itime)
        preds = preds[:, ilon, :]; true = true[:, ilon, :]
        self.plot_double_slice(preds.T, true.T, var, **kwargs)

    def plot_double_lat_lon_slice(self, var, itime, ilev, **kwargs):
        # Get predictions and true values. THIS WILL BE DIFFERENT FOR TF
        preds, true = self._get_preds_and_truth(var, itime)
        preds = preds[:, :, ilev]; true = true[:, :, ilev]
        self.plot_double_slice(preds, true, var, **kwargs)

    @staticmethod
    def plot_double_slice(p, t, var=None, **kwargs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if var is None or var not in range_dict.keys():
            mn = None; mx = None
        else:
            mn = range_dict[var][0]; mx = range_dict[var][1]
        axes[0].imshow(p, **kwargs)
        axes[1].imshow(t, **kwargs)
        axes[0].set_title('CBRAIN Predictions')
        axes[1].set_title('SP-CAM Truth')
        plt.show()

    @staticmethod
    def plot_slice(x, title, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(x, **kwargs)
        ax.set_title(title)
        plt.show()

    def _get_preds_and_truth(self, var, itime):
        f = self.keras_features['features'][
            itime * self.ngeo:(itime + 1) * self.ngeo]
        p = self.model.predict_on_batch(f)
        t = self.keras_targets['targets'][
            itime * self.ngeo:(itime + 1) * self.ngeo]
        return self.reshape_output(p, var), self.reshape_output(t, var)

    def reshape_output(self, x, var=None, unscale=True):
        """
        Assumes [sample, z] for one time step
        """
        x = x.reshape(self.nlat, self.nlon, x.shape[-1])  # [lat, lon, z]
        if var is None: var_idxs = slice(0, None, 1)
        else:
            var_idxs = self._get_var_idxs('target', var)
        x = x[:, :, var_idxs]
        # Unscale
        if unscale: x /= conversion_dict[var]
        return x

    def _get_var_idxs(self, feature_or_target, var):
        return [
            i for i, s in
            enumerate(list(self.keras_norm[f'{feature_or_target}_names'][:]))
            if var in s]

    def _get_dP(self, f):
        PS_idxs = self._get_var_idxs('feature', 'PS')
        PS = (
            f[:, PS_idxs] * self.keras_norm['feature_stds'][PS_idxs] +
            self.keras_norm['feature_means'][PS_idxs]
        )
        return np.diff(P0 * hyai + PS * hybi, axis=1)

    def compute_stats(self, n_iter=None, compute_SPDT_SPDQ=False):
        """
        Compute statistics over entire dataset [lat, lon, lev].
        bias = mean(preds) - mean(true)
        mse = sse(preds, true) / n_samples
        rel_mse = mse / std(true)
        std_error = std(preds) - std(true)
        """
        # Get data generator without shuffling!
        gen_obj = DataGenerator(
            '/',
            self.keras_features_fn,
            self.keras_targets_fn,
            shuffle=False,
            batch_size=self.ngeo,  # time step sized batches
            verbose=True,
        )
        gen = gen_obj.return_generator()
        psum = np.zeros((self.ngeo, gen_obj.target_shape))
        tsum = np.copy(psum); sse = np.copy(psum)
        psqsum = np.copy(psum); tsqsum = np.copy(psum)
        if compute_SPDT_SPDQ:
            SPpred = np.zeros(self.ngeo); SPtrue = np.copy(SPpred)
        n = gen_obj.n_batches if n_iter is None else n_iter
        for t in tqdm(range(n)):  # Every batch is one time step!
            # Load features and targets
            f, t = next(gen)
            # Get predictions
            p = self.model.predict_on_batch(f)  # [ngeo samples, z]
            # Unscale outputs at this level
            p /= self.unscale_arr; t /= self.unscale_arr
            # Compute statistics
            psum += p; tsum += t
            psqsum += p ** 2; tsqsum += t ** 2
            sse += (t - p) ** 2
            if compute_SPDT_SPDQ:
                tmp = self._compute_SPDT_SPDQ(f, t, p)
                SPpred += tmp[0]; SPtrue += tmp[1]

        # Compute average statistics
        self.stats_dict = {}
        pmean = psum / n; tmean = tsum / n
        self.bias = pmean - tmean; self.stats_dict['bias'] = self.bias
        self.mse = sse / n; self.stats_dict['mse'] = self.mse
        self.pred_var = (psqsum / n - pmean ** 2) * n / (n - 1)  # Sample variance
        self.stats_dict['pred_var'] = self.pred_var
        self.true_var = (tsqsum / n - tmean ** 2) * n / (n - 1)
        self.stats_dict['true_var'] = self.true_var
        if compute_SPDT_SPDQ:
            self.en_err_p = SPpred / n
            self.en_err_t = SPtrue / n
            print('Mean squared energy violation. True:',
                  np.mean(self.en_err_t))
            print('Mean squared energy violation. Pred:',
                  np.mean(self.en_err_p))

    def _compute_SPDT_SPDQ(self, f, t, p):
        # Get dP
        dP = self._get_dP(f)
        SPDT_pred = self.vint((p[:, self._get_var_idxs('target', 'SPDT')]),
                              C_P, dP)
        SPDQ_pred = self.vint((p[:, self._get_var_idxs('target', 'SPDQ')]),
                              L_V, dP)
        SPDT_true = self.vint((t[:, self._get_var_idxs('target', 'SPDT')]),
                              C_P, dP)
        SPDQ_true = self.vint((t[:, self._get_var_idxs('target', 'SPDQ')]),
                              L_V, dP)
        return np.square(SPDT_pred + SPDQ_pred), np.square(SPDT_true + SPDQ_true)

    @staticmethod
    def vint(x, factor, dP):
        return np.sum(x * factor * dP / G, -1)

    def mean_stats(self, cutoff_level=9):
        expl_var_str = f'expl_var_cut{cutoff_level}'
        df = pd.DataFrame(
            index=self.target_vars + ['all'],
            columns=list(self.stats_dict.keys()) + [expl_var_str])
        # Compute statistics for each variable
        for var in self.target_vars + ['all']:
            for stat_name, stat in self.stats_dict.items():
                v = None if var == 'all' else var
                df.loc[var, stat_name] = np.mean(self.reshape_output(
                    stat, var=v, unscale=False))

                df.loc[var, expl_var_str] = np.mean((1. - (
                     np.mean(self.reshape_output(self.mse, v, unscale=False), axis=(0, 1)) /
                     np.mean(self.reshape_output(self.true_var, v, unscale=False), axis=(0, 1))
                ).reshape(-1, self.nlev))[:, cutoff_level:])
        return df

    def plot_stat_lat_lev_mean(self, stat_name, var, **kwargs):
        arr = np.mean(self.reshape_output(
            self.stats_dict[stat_name], var=var, unscale=False), axis=1).T
        self.plot_slice(arr, var + ' ' + stat_name, **kwargs)