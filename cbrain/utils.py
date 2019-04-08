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


def return_var_idxs(ds, var_list):
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
    # tgb - 2/11/2019 - Added tensorflow for compatibility with eager execution
    tensorflow.keras.backend.set_session(tf.Session(config=config))

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


def save_pickle(fn, obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fn):
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    return obj
