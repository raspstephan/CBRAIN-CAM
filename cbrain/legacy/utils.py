"""
Helper functions for Keras Cbrain implementation.

Author: Stephan Rasp
"""

# Imports
from .imports import *

# Basic setup
np.random.seed(42)


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
                                    feature_names, target_names,
                                    unscale_targets=False):
    """NOTE: FEATURES HARD-CODED!!!
    Features are [TAP, QAP, dTdt_adiabatic, dQdt_adiabatic, SHFLX, LHFLX, SOLIN]
    Targets are [SPDT, SPDQ, QRL, QRS, PRECT, FLUT]
    """
    nz = 21
    z = np.arange(20, -1, -1)
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    in_axes = np.ravel(axes[0, :])
    out_axes = np.ravel(axes[1, :])

    for i in range(len(feature_names[:-3])):
        in_axes[i].plot(features[sample_idx, i*nz:(i+1)*nz], z, c='b')
        in_axes[i].set_title(feature_names[i])
    twin_in = in_axes[-1].twinx()
    in_axes[-1].bar([1, 2], features[sample_idx, -3:-1])
    twin_in.bar([3], features[sample_idx, -1])
    in_axes[-1].set_xticks([1, 2, 3])
    in_axes[-1].set_xticklabels(feature_names[-3:])

    for i in range(len(target_names[:-2])):
        if unscale_targets:
            u = conversion_dict[target_names[i]]
        else:
            u = 1.
        out_axes[i].plot(targets[sample_idx, i * nz:(i + 1) * nz] / u, z,
                         label='True', c='b')
        out_axes[i].plot(predictions[sample_idx, i * nz:(i + 1) * nz] / u, z,
                         label='Prediction', c='g')
        out_axes[i].set_title(target_names[i])
        #out_axes[i].axvline(0, c='gray', zorder=0.1)
    twin_out = out_axes[-1].twinx()
    out_axes[-1].bar(1 - 0.2, targets[sample_idx, -2], 0.4,
                     color='b')
    out_axes[-1].bar(1 + 0.2, predictions[sample_idx, -2],
                     0.4, color='g')
    twin_out.bar(2 - 0.2, targets[sample_idx, -1], 0.4,
                     color='b')
    twin_out.bar(2 + 0.2, predictions[sample_idx, -1],
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


def split_variables(x):
    """Hard-coded variable split for 21 lev"""
    spdt  = x[..., :21]
    spdq  = x[..., 21:42]
    qrl   = x[..., 42:63]
    qrs   = x[..., 63:84]
    prect = x[..., 84]
    flut  = x[..., 85]
    return spdt, spdq, qrl, qrs, prect, flut


def np_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
def np_log_loss(y_true, y_pred):
    return np.mean(np.log(np_rmse(y_true, y_pred) + 1e-20) / np.log(10.))


def run_diagnostics(model_fn, data_dir, valid_pref, norm_fn, verbose=False,
                    convo=False):
    # Load model
    model = keras.models.load_model(model_fn)
    if verbose: print(model.summary())

    # Load normalization file
    norm = h5py.File(norm_fn, 'r')

    # Get data generator without shuffling!
    n_lon = 128
    n_lat = 64
    n_geo = n_lat * n_lon
    gen_obj = DataGenerator(
        data_dir,
        valid_pref + '_features.nc',
        valid_pref + '_targets.nc',
        shuffle=False,
        batch_size=n_geo,
        verbose=False,
    )
    gen = gen_obj.return_generator(convo)
    # Loop over chunks, get predictions compute scores
    sse = np.zeros((gen_obj.target_shape))  # Sum of squared errors [z]
    var_log_loss = np.zeros((6))
    log_loss = 0
    for t in tqdm(range(gen_obj.n_batches)):
        # Load features and targets
        tmp_features, tmp_targets = next(gen)
        # Get predictions
        tmp_preds = model.predict_on_batch(tmp_features)
        # Reshape to [time, lat, lon, lev]
        tmp_targets = tmp_targets.reshape(
            (-1, n_lat, n_lon, tmp_targets.shape[-1]))
        tmp_preds = tmp_preds.reshape((-1, n_lat, n_lon, tmp_preds.shape[-1]))
        # Split by variable
        split_targets = split_variables(tmp_targets)
        split_preds = split_variables(tmp_preds)
        # Compute statistics
        sse += np.sum((tmp_targets - tmp_preds) ** 2, axis=(0, 1, 2))
        log_loss += np_log_loss(tmp_targets, tmp_preds)
        for i in range(6):
            var_log_loss[i] += np_log_loss(split_targets[i], split_preds[i])

    # Get average statistics
    mse = sse / (gen_obj.n_batches * n_geo)
    rel_mse = mse / norm['target_stds'][:] ** 2
    var_log_loss = var_log_loss / (gen_obj.n_batches)
    log_loss = log_loss / (gen_obj.n_batches)

    return rel_mse, mse, var_log_loss, log_loss


def reshape_geo(x):
    return x.reshape((-1, n_lat, n_lon, x.shape[-1]))

def limit_mem():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def global_mean(ds, var):
    return ds[var].mean(dim=('lat', 'lon', 'lev')) 

def basic_debug(debug, ref=False, data_dir='/scratch/05488/tg847872/cloudbrain_ctrl_aquaplanet_03/'):
    if not ref: ds = xr.open_mfdataset(f'{data_dir}*{debug}*', decode_times=False)
    else: ds = xr.open_mfdataset(f'{REF_DIR}AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-01-0[1-5]-00000.nc', decode_times=False)
    plot_global_stats(ds)
    ds['TAP'].max(('lat', 'lon')).T.plot(yincrease=False, robust=True)
    plt.title('Max TAP'); plt.show()
    ds['QAP'].max(('lat', 'lon')).T.plot(yincrease=False, robust=True)
    plt.title('Max QAP'); plt.show()
    return ds

def plot_time_lev(ds, var, func=np.mean, **kwargs):
    fig = plt.figure(figsize=(12, 5))
    plt.imshow(func(ds[var], axis=(2, 3)).T, **kwargs)
    plt.colorbar(shrink=0.3)
    plt.show()

def get2Didxs(a, func): return(np.unravel_index(func(a), a.shape))

def get_cb_inps(ds, t, m, s):
    x = np.concatenate(
        [ds['NNTC'][t], ds['NNQC'][t], ds['NNVC'][t], ds['dTdtadia'][t], ds['dQdtadia'][t],
         np.expand_dims(ds['NNPS'][t], 0), np.expand_dims(ds['NNSOLIN'][t], 0)]
    )
    return normalize(x, m, s)

def normalize(x, m, s):
    return (x - m[:, None, None]) / s[:, None, None]

def stack_outps(ds, t):
    x = np.concatenate(
        [ds['BRAINDQ'].isel(time=t)*L_V, ds['BRAINDT'].isel(time=t)*C_P, 
         ds['QRL'].isel(time=t)*C_P, ds['QRS'].isel(time=t)*C_P])
    return x

def get_P_from_ds(ds):
    return ds.P0 * ds.hyai + ds.PS * ds.hybi

def get_dP_from_ds(ds):
    p = get_P_from_ds(ds)
    p_diff = p.diff(dim='ilev')
    return p_diff.rename({'ilev':'lev'})

def vint(ds, var, factor, lev_sl=slice(0, None)):
    dP = get_dP_from_ds(ds)
    x = ds[var] if type(var) is str else var
    dP['lev'] = x['lev']
    return (dP * x * factor / G).isel(lev=lev_sl).sum(dim='lev')

def vavg(ds, var, factor, lev_sl=slice(0, None)):
    dP = get_dP_from_ds(ds)
    x = ds[var] if type(var) is str else var
    dP['lev'] = x['lev']
    return (dP * x * factor).isel(lev=lev_sl).sum(dim='lev') / dP.isel(lev=lev_sl).sum(dim='lev')

def gw_avg(ds, var=None, da=None):
    da = ds[var] if da is None else da
    return (da * ds['gw']).sum('lat').mean('lon') / 2.

def plot_global_stats(ds):
    gw_avg(ds, da=vint(ds, 'TAP', C_P)[1:]).plot()
    plt.title('Global Dry Static Energy')
    plt.show()
    gw_avg(ds, da=vint(ds, 'QAP', 1.)[1:]).plot()
    plt.title('Global Water Vapor')
    plt.show()