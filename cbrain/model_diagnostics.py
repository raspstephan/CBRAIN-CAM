"""
Contains the ModelDiagnostics class.
TODO:
- Colorbar
- Axes labels
"""

# Imports
from .imports import *
from .data_generator import DataGenerator
from .preprocess_aqua import L_V, C_P, conversion_dict
import pickle

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
    Two basic functionalities:
    1. Plotting --> need preds and truth of selected time step in original values for one var
    2. Global statistics --> also from denormalized values

    Differences between TF and Keras:
    1. Data loading: For Keras I will use my data_generator (much faster),
                     for TF I will read and process the raw aqua files
    2. Output normalization
    3. Output shape: 1D for Keras, 2D for TF --> Use TF convention
    NOTE: This cannot handle outputs with one level.
    """
    def __init__(self, model_path, is_tf=False,
                 fpath=None, tpath=None, npath=None, norms=None,
                 tf_filepattern=None, tf_fvars=None, tf_tvars=None, tf_meanpath=None,
                 tf_stdpath=None, nlat=64, nlon=128, nlev=30, ntime=48, raw_nlev=30,
                 watch_mem=False):
        # Basic setup
        self.is_tf = is_tf; self.is_k = not is_tf
        self.model = keras.models.load_model(model_path, custom_objects={"tf": tf})
        self.nlat, self.nlon, self.nlev = (nlat, nlon, nlev)
        self.ngeo = nlat * nlon
        self.ntime = ntime; self.raw_nlev=30
        self.watch_mem = watch_mem
        self.save_str = (model_path.split('/')[-1].split('.')[0] + '_' +
                          fpath.split('/')[-1].split('.')[0].split('_features')[0] + '.pkl')
        # Get variable names and open arrays
        if self.is_k:
            self.k_norm = h5py.File(npath, 'r')
            self._get_k_norm_arrs(*norms)
            self.k_features = h5py.File(fpath, 'r')
            self.k_targets = h5py.File(tpath, 'r')
            self.fvars, self.tvars = self._get_k_vars()
        else:
            self.fvars, self.tvars = (tf_fvars, tf_tvars)
            self.tf_mean, self.tf_std = (nc.Dataset(tf_meanpath), nc.Dataset(tf_stdpath))
            self.tf_files = sorted(glob(tf_filepattern))

    # Init helper functions
    def _get_k_vars(self):
        """
        Return unique variable names for features and targets in correct order.
        """
        return [list(dict.fromkeys(
            [f.split('_lev')[0] for f in list(self.k_norm[f'{a}_names'][:])]
            )) for a in ['feature', 'target']]

    def _get_k_norm_arrs(self, fsub, fdiv, tsub, tmult):
        """
        Allocate normalization arrays for keras.
        """
        self.fsub = 0. if fsub is None else self.k_norm[fsub]
        if fdiv is None: self.fdiv = 1.
        elif fdiv == 'range':
            self.fdiv = self.k_norm['feature_maxs'] - self.k_norm['feature_mins']
        elif fdiv == 'max_rs': self.fdiv = np.maximum(
            self.k_norm['feature_maxs'][:] - self.k_norm['feature_mins'][:],
            self.k_norm['feature_stds_by_var'])
        elif fdiv == 'feature_stds_eps':
            eps = 1e-10
            self.fdiv = np.maximum(self.k_norm['feature_stds'][:], eps)
        else: self.fdiv = self.k_norm[fdiv]
        self.tsub = 0. if tsub is None else self.k_norm[tsub]
        self.tmult = 1. if fsub is None else self.k_norm[tmult]

    def get_pt(self, itime, var=None):
        """
        Returns denormalized predictions and truth for a given time step and var.
        [lat, lon, lev] or [lat, lon, var, lev] if var is None
        """
        if self.is_k: p, t = self._get_k_pt(itime, var)
        else: p, t = self._get_tf_pt(itime, var)
        return p, t

    def _get_k_pt(self, itime, var=None):
        """Keras version"""
        f = (self.k_features['features'][itime*self.ngeo:(itime+1)*self.ngeo] -
             self.fsub) / self.fdiv
        p = self.model.predict_on_batch(f) / self.tmult + self.tsub
        t = self.k_targets['targets'][itime*self.ngeo:(itime+1)*self.ngeo]
        # At this stage they have shape [ngeo, stacked_levs]
        return self._k_reshape(p, var), self._k_reshape(t, var)

    def _get_tf_pt(self, itime=None, var=None, idate=None):
        """Tensorflow version
        If idate is given, instead of itime, return the entire file
        """
        if idate is None:
            idate = itime // self.ntime; itime_tmp = itime % self.ntime
        else: itime_tmp = None
        f = self._get_tf_f_or_t(idate, itime_tmp, 'f')
        if self.watch_mem: p = self.model.predict(f, 1024)
        else: p = self.model.predict_on_batch(f)
        t = self._get_tf_f_or_t(idate, itime_tmp, 't', normalize=False)
        p, t = (self._tf_reshape(p), self._tf_reshape(t))
        if var is None:
            return self._tf_denorm(p), t
        else:
            var_idx = self.tvars.index(var)
            return self._tf_denorm(p)[..., var_idx, :], t[..., var_idx, :]

    def _k_reshape(self, x, var=None):
        """For targets only atm.
        [ngeo, stacked_levs] --> [lat, lon, var, lev]
        Select var if not None.
        """
        x = x.reshape(self.nlat, self.nlon, -1, self.nlev)
        if var is not None: x = x[:, :, self.tvars.index(var), :]
        return x

    def _tf_reshape(self, x):
        """[ngeo, var, nlev] -- > [lat, lon, var, lev]
        or [ngeo*ntime, var, nlev] --> [ntime, lat, lon, var, lev]
        """
        ntar = len(self.tvars)
        if x.shape[0] == self.ngeo:
            return x.reshape(self.nlat, self.nlon, ntar, self.nlev)[:, :, :, ::-1]
        else:
            return x.reshape(self.ntime, self.nlat, self.nlon, ntar, self.nlev)[..., ::-1]

    def _get_tf_f_or_t(self, idate, itime, f_or_t, normalize=True):
        with nc.Dataset(self.tf_files[idate], 'r') as ds:
            arr = []
            vars = self.fvars if f_or_t == 'f' else self.tvars
            for var in vars:
                da = ds[var][:]
                if normalize: da = (da - self.tf_mean[var][:]) / self.tf_std[var][:]
                if da.ndim == 4:   # 3D variables [time, lev, lat, lon] --> [sample, lev]
                    a = np.rollaxis(da, 1, 4).reshape(-1, self.raw_nlev)
                elif da.ndim == 3:   # 2D variables [time, lat, lon]
                    a = np.rollaxis(np.tile(da.reshape(-1), (self.raw_nlev, 1)), 0, 2)
                elif da.ndim == 1:   # lat
                    a = np.rollaxis(np.tile(da, (self.ntime, self.raw_nlev, self.nlon, 1)),
                                    1, 4).reshape(-1, self.raw_nlev)
                else:
                    raise Exception('Incompatible number of dimensions')
                arr.append(a)
            arr = np.expand_dims(np.rollaxis(np.array(arr), 0, 2), 3) # [sample, feature, lev, 1]
        arr =  arr[:, :, -self.nlev:][:, :, ::-1]
        if itime is not None: arr = arr[itime*self.ngeo:(itime+1)*self.ngeo]
        return arr

    def _tf_denorm(self, x, f_or_t='t'):
        for i, var in enumerate(self.fvars if f_or_t == 'f' else self.tvars):
            m, s = [np.rollaxis(ds[var][-self.nlev:], 0, 3)
                    for ds in [self.tf_mean, self.tf_std]]
            x[..., i, :] = x[..., i, :] * s + m
        return x

    # Plotting functions
    def plot_double_xy(self, itime, ilev, var, **kwargs):
        p, t = self.get_pt(itime, var)
        return self.plot_double_slice(p[:, :, ilev], t[:, :, ilev], **kwargs)

    def plot_double_yz(self, itime, ilon, var, **kwargs):
        p, t = self.get_pt(itime, var)
        return self.plot_double_slice(p[:, ilon, :].T, t[:, ilon, :].T, **kwargs)

    def plot_double_slice(self, p, t, title='', unit='', **kwargs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        I1 = axes[0].imshow(p, **kwargs)
        I2 = axes[1].imshow(t, **kwargs)
        cb1 = fig.colorbar(I1, ax=axes[0], orientation='horizontal')
        cb2 = fig.colorbar(I2, ax=axes[1], orientation='horizontal')
        cb1.set_label(unit); cb2.set_label(unit)
        axes[0].set_title('CBRAIN Predictions')
        axes[1].set_title('SP-CAM Truth')
        fig.suptitle(title)
        return fig, axes

    def plot_slice(self, x, title='', unit='', **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        I = ax.imshow(x, **kwargs)
        cb = fig.colorbar(I, ax=ax, orientation='horizontal')
        cb.set_label(unit)
        ax.set_title(title)
        return fig

    # Statistics computation
    def compute_stats(self, niter=None):
        """Compute statistics in for [lat, lon, var, lev]"""
        if self.is_k: nt = self.k_features['features'].shape[0] // self.ngeo
        else: nt = len(self.tf_files) * self.ntime
        if niter is not None: nt = niter
        # Allocate stats arrays
        psum = np.zeros((self.nlat, self.nlon, len(self.tvars), self.nlev))
        tsum = np.copy(psum); sse = np.copy(psum)
        psqsum = np.copy(psum); tsqsum = np.copy(psum)
        for itime in tqdm(range(nt)):
            if self.is_k:
                p, t = self.get_pt(itime)   # [lat, lon, var, lev]
            else:   # For TF load entire aqua file at once!
                itmp = itime % self.ntime; idate = itime // self.ntime
                if itmp == 0:
                    pday, tday = self._get_tf_pt(idate=idate)
                p, t = (pday[itmp], tday[itmp])
            # Compute statistics
            psum += p; tsum += t
            psqsum += p ** 2; tsqsum += t ** 2
            sse += (t - p) ** 2
        # Compute average statistics
        self.stats = {}
        pmean = psum / nt; tmean = tsum / nt
        self.stats['bias'] = pmean - tmean
        self.stats['mse'] = sse / nt
        self.stats['pred_mean'] = psum / nt
        self.stats['true_mean'] = tsum / nt
        self.stats['pred_sqmean'] = psqsum / nt
        self.stats['true_sqmean'] = tsqsum / nt
        self.stats['pred_var'] = psqsum / nt - pmean ** 2
        self.stats['true_var'] = tsqsum / nt - tmean ** 2
        self.stats['r2'] = 1. - (self.stats['mse'] / self.stats['true_var'])
        # Compute horizontal stats [var, lev]
        self.stats['hor_tsqmean'] = np.mean(self.stats['true_sqmean'], axis=(0,1))
        self.stats['hor_tmean'] = np.mean(self.stats['true_mean'], axis=(0, 1))
        self.stats['hor_mse'] = np.mean(self.stats['mse'], axis=(0, 1))
        self.stats['hor_tvar'] = self.stats['hor_tsqmean'] - self.stats['hor_tmean'] ** 2
        self.stats['hor_r2'] = 1 - (self.stats['hor_mse'] / self.stats['hor_tvar'])

    def mean_stats(self, cutoff_level=0):
        """Get average statistics for each variable and returns dataframe"""
        df = pd.DataFrame(index=self.tvars + ['all'],
            columns=list(self.stats.keys()))
        for ivar, var in enumerate(self.tvars):
            for stat_name, stat in self.stats.items():
                # Stats have shape [lat, lon, var, lev]
                df.loc[var, stat_name] = np.mean(stat[..., ivar, cutoff_level:])
        df.loc['all']['hor_r2'] = np.mean(self.stats['hor_r2'][:, cutoff_level:].mean())
        self.stats_df = df
        return df

    def save_stats(self, path=None):
        if path is None:
            os.makedirs('./tmp', exist_ok=True)
            path= './tmp/' + self.save_str
        with open(path, 'wb') as f: pickle.dump(self.stats, f)

    def load_stats(self, path=None):
        if path is None: path= './tmp/' + self.save_str
        with open(path, 'rb') as f: self.stats = pickle.load(f)





    # def _compute_SPDT_SPDQ(self, f, t, p):
    #     # Get dP
    #     dP = self._get_dP(f)
    #     SPDT_pred = self.vint((p[:, self._get_var_idxs('target', 'SPDT')]),
    #                           C_P, dP)
    #     SPDQ_pred = self.vint((p[:, self._get_var_idxs('target', 'SPDQ')]),
    #                           L_V, dP)
    #     SPDT_true = self.vint((t[:, self._get_var_idxs('target', 'SPDT')]),
    #                           C_P, dP)
    #     SPDQ_true = self.vint((t[:, self._get_var_idxs('target', 'SPDQ')]),
    #                           L_V, dP)
    #     return np.square(SPDT_pred + SPDQ_pred), np.square(SPDT_true + SPDQ_true)
    #
    # @staticmethod
    # def vint(x, factor, dP):
    #     return np.sum(x * factor * dP / G, -1)