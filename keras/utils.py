"""
Helper functions for Keras Cbrain implementation.

Author: Stephan Rasp
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
from collections import OrderedDict
from scipy.stats import binned_statistic

# Basic setup
np.random.seed(42)
sns.set_style('dark')
sns.set_palette('deep')
sns.set_context('talk')
plt.rcParams["figure.figsize"] = (10,7)


# Plotting functions
def vis_features_targets_from_nc(outputs, sample_idx, feature_vars, target_vars,
                                 preds=None):
    """Visualize features and targets from the preprocessed dataset.

    Args:
        outputs: nc object
        sample_idx: index of sample to be visualized
        feature_vars: dictionary with feature name and dim number
        target_vars: same for targets
        preds: model predictions

    """
    z = np.arange(20, -1, -1)
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    in_axes = np.ravel(axes[:, :4])
    out_axes = np.ravel(axes[:, 4])

    in_2d = [k for k, v in feature_vars.items() if v == 2]
    in_1d = [k for k, v in feature_vars.items() if v == 1]
    for i, var_name in enumerate(in_2d):
        in_axes[i].plot(outputs.variables[var_name][:, sample_idx], z)
        in_axes[i].set_title(var_name)
    in_bars = [outputs.variables[v][sample_idx] for v in in_1d]
    in_axes[-1].bar(range(len(in_bars)), in_bars, tick_label=in_1d)

    if preds is not None:
        preds = np.reshape(preds, (preds.shape[0], 21, 2))[sample_idx]
        preds[:, 0] /= 1000.
        preds[:, 1] /= 2.5e6

    for i, var_name in enumerate(target_vars.keys()):
        out_axes[i].plot(outputs.variables[var_name][:, sample_idx], z, c='r')
        out_axes[i].set_title(var_name)
        if preds is not None:
            out_axes[i].plot(preds[:, i], z, c='green')

    plt.suptitle('Sample %i' % sample_idx, fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_lat_z_statistic(a, lats, statistic, cmap='inferno', vmin=None, vmax=None):
    b = binned_statistic(lats, a.T, statistic=statistic, bins=20,
                         range=(lats.min(), lats.max()))
    mean_lats = (b[1][1:] + b[1][:-1]) / 2.
    mean_lats = ['%.0f' % l for l in mean_lats]
    plt.imshow(b[0], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(range(len(mean_lats)), mean_lats)
    plt.colorbar()
    plt.show()


def vis_features_targets_from_pred(features, targets,
                                   predictions, sample_idx,
                                   feature_names, target_names):
    """NOTE: FEATURES HARD-CODED!!!

    """
    nz = 21
    z = np.arange(20, -1, -1)
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    in_axes = np.ravel(axes[:, :4])
    out_axes = np.ravel(axes[:, 4])

    for i in range(len(feature_names[:-3])):
        in_axes[i].plot(features[sample_idx, i*nz:(i+1)*nz], z)
        in_axes[i].set_title(feature_names[i])
    in_axes[-1].bar(range(3), features[sample_idx, -3:],
                    tick_label=feature_names[-3:])

    # Split targets
    t = targets.reshape((targets.shape[0], -1, 2))
    p = predictions.reshape((predictions.shape[0], -1, 2))
    for i in range(t.shape[-1]):
        out_axes[i].plot(t[sample_idx, :, i], z, label='True')
        out_axes[i].plot(p[sample_idx, :, i], z, label='Prediction')
        #out_axes[i].set_xlim(-0.5, 0.5)
        out_axes[i].set_title(target_names[i])
        out_axes[i].axvline(0, c='gray', zorder=0.1)
    out_axes[-1].legend(loc=0)
    plt.suptitle('Sample %i' % sample_idx, fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def vis_features_targets_from_pred2(features, targets,
                                    predictions, sample_idx,
                                    feature_names, target_names):
    """NOTE: FEATURES HARD-CODED!!!
    Features are [TAP, QAP, dTdt_adiabatic, dQdt_adiabatic, SHFLX, LHFLX]
    Targets are [SPDT, SPDQ, QRL, QRS, PRECT, FLUT]
    """
    nz = 21
    z = np.arange(20, -1, -1)
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    in_axes = np.ravel(axes[0, :])
    out_axes = np.ravel(axes[1, :])

    for i in range(len(feature_names[:-2])):
        in_axes[i].plot(features[sample_idx, i*nz:(i+1)*nz], z, c='b')
        in_axes[i].set_title(feature_names[i])
    in_axes[-1].bar(range(2), features[sample_idx, -2:],
                    tick_label=feature_names[-2:])

    for i in range(len(target_names[:-2])):
        out_axes[i].plot(targets[sample_idx, i * nz:(i + 1) * nz], z,
                         label='True', c='b')
        out_axes[i].plot(predictions[sample_idx, i * nz:(i + 1) * nz], z,
                         label='Prediction', c='g')
        out_axes[i].set_title(target_names[i])
        #out_axes[i].axvline(0, c='gray', zorder=0.1)
    twin = out_axes[-1].twinx()
    out_axes[-1].bar(1 - 0.2, targets[sample_idx, -2], 0.4,
                     color='b')
    out_axes[-1].bar(1 + 0.2, predictions[sample_idx, -2],
                     0.4, color='g')
    twin.bar(2 - 0.2, targets[sample_idx, -1], 0.4,
                     color='b')
    twin.bar(2 + 0.2, predictions[sample_idx, -1],
                     0.4, color='g')
    out_axes[-1].set_xticks([1, 2])
    out_axes[-1].set_xticklabels(target_names[-2:])
    out_axes[0].legend(loc=0)
    plt.suptitle('Sample %i' % sample_idx, fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

def rmse_stat(x):
    """
    RMSE function for lat_z plots
    Args:
        x: 

    Returns:

    """
    return np.sqrt(np.mean(x**2))