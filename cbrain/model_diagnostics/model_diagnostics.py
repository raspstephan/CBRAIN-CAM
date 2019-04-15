"""
Contains the ModelDiagnostics class.

"""

# Imports
from ..imports import *
from ..utils import *
from ..data_generator import DataGenerator
from ..cam_constants import *
from ..layers import *
from ..losses import *
import pickle
import yaml

# tgb - 4/10/2019 - Adding a model_path input to load a custom model
class ModelDiagnostics():
    def __init__(self, model, config_fn, data_fn, nlat=64, nlon=128, nlev=30, ntime=48):

        self.nlat, self.nlon = nlat, nlon
        self.ngeo = nlat * nlon

        repo_dir = os.getcwd().split('CBRAIN-CAM')[0] + 'CBRAIN-CAM/'
        with open(config_fn, 'r') as f:
            config = yaml.load(f)

#         self.model = tf.keras.models.load_model(
#             repo_dir + 'saved_models/' + config['exp_name'] + '/model.h5',
#             custom_objects={**layer_dict, **loss_dict})
#         self.model = tf.keras.models.load_model(model_path,
#             custom_objects={**layer_dict, **loss_dict})
        self.model = model

        out_scale_dict = load_pickle(config['output_dict'])

        self.valid_gen = DataGenerator(
            data_fn=config['data_dir'] + config['valid_fn'],
            input_vars=config['inputs'],
            output_vars=config['outputs'],
            norm_fn=config['data_dir'] + config['norm_fn'],
            input_transform=(config['input_sub'], config['input_div']),
            output_transform=out_scale_dict,
            batch_size=self.ngeo,
            shuffle=False,
            xarray=True
        )

    def reshape_ngeo(self, x):
        return x.reshape(self.nlat, self.nlon, -1)

    def get_output_var_idx(self, var):
        var_idxs = self.valid_gen.norm_ds.var_names[self.valid_gen.output_idxs]
        var_idxs = np.where(var_idxs == var)[0]
        return var_idxs

    def get_truth_pred(self, itime, var=None):
        X, truth = self.valid_gen[itime]
        pred = self.model.predict_on_batch(X)
        # Inverse transform
        truth = self.valid_gen.output_transform.inverse_transform(truth.values)
        pred = self.valid_gen.output_transform.inverse_transform(pred)

        if var is not None:
            var_idxs = self.get_output_var_idx(var)
            truth = truth[:, var_idxs]
            pred = pred[:, var_idxs]

        return self.reshape_ngeo(truth), self.reshape_ngeo(pred)

    # Plotting functions
    def plot_double_xy(self, itime, ilev, var, **kwargs):
        t, p = self.get_truth_pred(itime, var)
        if p.ndim == 3: p, t = p[:, :, ilev], t[:, :, ilev]
        return self.plot_double_slice(t, p, **kwargs)

    def plot_double_yz(self, itime, ilon, var, **kwargs):
        t, p = self.get_truth_pred(itime, var)
        return self.plot_double_slice(t[:, ilon, :].T, p[:, ilon, :].T, **kwargs)

    def plot_double_slice(self, t, p, title='', unit='', **kwargs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        I1 = axes[0].imshow(p, **kwargs)
        I2 = axes[1].imshow(t, **kwargs)
        cb1 = fig.colorbar(I1, ax=axes[0], orientation='horizontal')
        cb2 = fig.colorbar(I2, ax=axes[1], orientation='horizontal')
        cb1.set_label(unit);
        cb2.set_label(unit)
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
        nt = self.valid_gen.n_batches
        if niter is not None: nt = niter
        # Allocate stats arrays
        psum = np.zeros((self.nlat, self.nlon, self.valid_gen.n_outputs))
        tsum = np.copy(psum)
        sse = np.copy(psum)
        psqsum = np.copy(psum)
        tsqsum = np.copy(psum)
        for itime in tqdm(range(nt)):
            t, p = self.get_truth_pred(itime)  # [lat, lon, var, lev]
            # Compute statistics
            psum += p
            tsum += t
            psqsum += p ** 2
            tsqsum += t ** 2
            sse += (t - p) ** 2

        # Compute average statistics
        self.stats = {}
        pmean = psum / nt
        tmean = tsum / nt
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
        self.stats['hor_tsqmean'] = np.mean(self.stats['true_sqmean'], axis=(0, 1))
        self.stats['hor_tmean'] = np.mean(self.stats['true_mean'], axis=(0, 1))
        self.stats['hor_mse'] = np.mean(self.stats['mse'], axis=(0, 1))
        self.stats['hor_tvar'] = self.stats['hor_tsqmean'] - self.stats['hor_tmean'] ** 2
        self.stats['hor_r2'] = 1 - (self.stats['hor_mse'] / self.stats['hor_tvar'])

    def mean_stats(self):
        """Get average statistics for each variable and returns dataframe"""
        df = pd.DataFrame(index=self.valid_gen.output_vars + ['all'],
                          columns=list(self.stats.keys()))
        for ivar, var in enumerate(self.valid_gen.output_vars):
            for stat_name, stat in self.stats.items():
                # Stats have shape [lat, lon, var, lev]
                df.loc[var, stat_name] = np.mean(stat[..., self.get_output_var_idx(var)])
        df.loc['all']['hor_r2'] = np.mean(df['hor_r2'].mean())
        self.stats_df = df
        return df
    #
    # def save_stats(self, path=None):
    #     if path is None:
    #         os.makedirs('./tmp', exist_ok=True)
    #         path = './tmp/' + self.save_str
    #     with open(path, 'wb') as f: pickle.dump(self.stats, f)
    #
    # def load_stats(self, path=None):
    #     if path is None: path = './tmp/' + self.save_str
    #     with open(path, 'rb') as f: self.stats = pickle.load(f)