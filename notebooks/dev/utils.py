from cbrain.imports import *

def global_mean(ds, var):
    return ds[var].mean(dim=('lat', 'lon', 'lev')) 

def plot_global_means(ds):
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    axes[0].plot(global_mean(ds, 'TAP'))
    axes[1].plot(global_mean(ds, 'QAP'))
    plt.show()
    
def plot_time_lev(ds, var, func=np.mean, **kwargs):
    fig = plt.figure(figsize=(12, 5))
    plt.imshow(func(ds[var], axis=(2, 3)).T, **kwargs)
    plt.colorbar(shrink=0.3);
    plt.show()

def basics(debug, ref=False):
    if not ref: ds = xr.open_mfdataset(f'{DATA_DIR}*{debug}*', decode_times=False, decode_cf=False)
    else: ds = xr.open_mfdataset(f'{REF_DIR}AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-01-0[1-9]-00000.nc', 
                                 decode_times=False, decode_cf=False)
    ds=ds.isel(time=slice(0,-1))
    plot_global_means(ds)
    plot_time_lev(ds, 'TAP', np.max)
    plot_time_lev(ds, 'QAP', np.max)
    return ds

def normalize(x, m, s):
    return (x - m[:, None, None]) / s[:, None, None]

def gw_avg(ds, var):
    return (ds[var] * ds['gw']).mean(dim=('lat', 'lon'))

def get_cb_inps(ds, t, m, s):
    x = np.concatenate(
        [ds['NNTBP'][t], ds['NNQBP'][t], ds['NNQCBP'][t], ds['NNQIBP'][t], ds['NNVBP'][t],
         np.expand_dims(ds['NNPS'][t], 0),
         np.expand_dims(ds['NNSOLIN'][t], 0), np.expand_dims(ds['NNSHF'][t], 0), np.expand_dims(ds['NNLHF'][t], 0)]
    )
    return normalize(x, m, s)

def stack_outps(ds, t):
    x = np.concatenate(
        [ds['NNDT'].isel(time=t)*C_P, ds['NNDQ'].isel(time=t)*L_V, 
         ds['NNDQC'].isel(time=t)*L_V, ds['NNDQI'].isel(time=t)*L_V, 
         np.expand_dims(ds['NNPRECL'][t], 0)* (24*3600*2e-2), np.expand_dims(ds['NNPRECS'][t], 0)* (24*3600*2e-2),
         np.expand_dims(ds['NNFSNT'][t], 0)* (1e-3), np.expand_dims(ds['NNFSNS'][t], 0)* (-1e-3), 
         np.expand_dims(ds['NNFLNT'][t], 0)* (-1e-3), np.expand_dims(ds['NNFLNS'][t], 0)* (1e-3)])
    return x