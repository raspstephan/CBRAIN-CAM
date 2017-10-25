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

# Basic setup
np.random.seed(42)
sns.set_style('dark')
sns.set_palette('deep')
sns.set_context('talk')
plt.rcParams["figure.figsize"] = (10,7)


# Plotting functions
def vis_features_targets(outputs, sample_idx, feature_vars, target_vars,
                         preds=None):
    """Visualize features and targets

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