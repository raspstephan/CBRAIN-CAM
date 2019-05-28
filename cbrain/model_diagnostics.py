"""
Contains the ModelDiagnostics class.

"""

# Imports
from cbrain.imports import *
from cbrain.utils import *
from cbrain.data_generator import DataGenerator
from cbrain.cam_constants import *
from cbrain.layers import *
from cbrain.losses import *
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
            # tgb - 4/16/2019 - Changed data_fn to the argument data_fn of ModelDiagnostics
            #data_fn=config['data_dir'] + config['valid_fn'],
            data_fn = data_fn,
            input_vars=config['inputs'],
            output_vars=config['outputs'],
            norm_fn=config['data_dir'] + config['norm_fn'],
            input_transform=(config['input_sub'], config['input_div']),
            output_transform=out_scale_dict,
            batch_size=self.ngeo,
            shuffle=False,
            xarray=True,
            var_cut_off=config['var_cut_off'] if 'var_cut_off' in config.keys() else None
        )

    def reshape_ngeo(self, x):
        return x.reshape(self.nlat, self.nlon, -1)
    
    def get_input_var_idx(self, var):
        var_idxs = self.valid_gen.norm_ds.var_names[self.valid_gen.input_idxs]
        var_idxs = np.where(var_idxs == var)[0]
        return var_idxs

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
    
    # tgb - 4/18/2019 - Gets input and prediction in normalized form
    def get_inp_pred(self,itime):
        """ Gets input and prediction in normalized form """
        X, truth = self.valid_gen[itime]
        pred = self.model.predict_on_batch(X)
        return X.values, pred
    
    # tgb - 5/20/2019 - Get normalized/full pressure coordinate
    def dP_tilde(self,itime):
        X, truth = self.valid_gen[itime]
        return compute_dP_tilde(X.values[:, self.get_input_var_idx('PS')[0]],
                                self.valid_gen.input_transform.div[self.get_input_var_idx('PS')[0]],
                                self.valid_gen.input_transform.div[self.get_input_var_idx('PS')[0]],
                                self.valid_gen.output_transform.scale[self.get_output_var_idx('PHQ')],
                                hyai, hybi)
    
    def dP(self,itime):
        X, truth = self.valid_gen[itime]
        PS = X.values[:, self.get_input_var_idx('PS')[0]]
        PS_div = self.valid_gen.input_transform.div[self.get_input_var_idx('PS')[0]]
        PS_sub = self.valid_gen.input_transform.sub[self.get_input_var_idx('PS')[0]]
        PS = PS * PS_div + PS_sub
        P = P0 * hyai + PS[:, None] * hybi
        return P[:, 1:]-P[:, :-1]

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
    
    # tgb - 4/26/2019 - Calculates precipitation PDF
    def compute_precipPDF(self, niter=None, Nbin=100, Pmin=0, Pmax=350):
        """Compute precipitation [mm/day] PDF with Nbin bins from Pmin to Pmax"""
        # Add attributes
        self.Nbin = Nbin
        self.Pmin = Pmin
        self.Pmax = Pmax
        #TODO: Could calculate bins automatically from first timestep
        nt = self.valid_gen.n_batches
        if niter is not None: nt = niter
        # Constants
        self.CONV = 1e3*24*3600 # Conversion from m/s to mm/day
        # Allocate histogram array
        Phist = np.zeros(self.Nbin)
        Thist = np.copy(Phist)
        for itime in tqdm(range(nt)):
            # Get normalized truth and prediction vectors
            inp, tru = self.valid_gen[itime]  # get normalized
            pred = self.model.predict_on_batch(inp)
            # Calculate true and predicted precipitation
            Pprec = (np.sum(pred[:,-4:-2],axis=1))*self.CONV/(L_V*RHO_L)
            Tprec = (np.sum(tru[:,-4:-2],axis=1))*self.CONV/(L_V*RHO_L)
            # Calculate true and predicted histograms
            hist,edges = np.histogram(Pprec,
                                     range=(self.Pmin,self.Pmax),
                                     bins=Nbin)
            Phist += hist
            hist,edges = np.histogram(Tprec,
                                     range=(self.Pmin,self.Pmax),
                                     bins=Nbin)
            Thist +=hist

        # Saves histograms
        self.precip = {}
        self.precip['predcount'] = Phist
        self.precip['truecount'] = Thist
        self.precip['edges'] = edges
        self.precip['bins'] = 0.5*(edges[1:]+edges[:-1])
    
    # Residual computation
    def compute_res(self, niter=None):
        """Compute budget residuals for [lat, lon, var, lev]"""
        nt = self.valid_gen.n_batches
        if niter is not None: nt = niter
        # Allocate stats arrays
        entres = np.zeros((self.nlat, self.nlon, self.valid_gen.n_outputs))
        masres = np.copy(entres)
        lwres = np.copy(entres)
        swres = np.copy(entres)
        for itime in tqdm(range(nt)):
            inp, p = self.get_inp_pred(itime)  # [lat, lon, var, lev]
            # residuals
            entres += self.reshape_ngeo(self.ent_res(inp, p))
            masres += self.reshape_ngeo(self.mass_res(inp, p))
            lwres += self.reshape_ngeo(self.lw_res(inp, p))
            swres += self.reshape_ngeo(self.sw_res(inp, p))

        # Compute average statistics
        self.res = {}
        self.res['ent'] = entres/nt
        self.res['mass'] = masres/nt
        self.res['lw'] = lwres/nt
        self.res['sw'] = swres/nt
    
    # tgb - 4/18/2019 - mse in W2/m4
    def mse_W2m4(self):
        """Calculate mean-squared-error in W2/m4"""
        return self.stats['mse']*(self.valid_gen.output_transform.scale**2)
    
    # tgb - 4/18/2019 - Loss functions in numpy for model diagnostics purposes
    def mass_res(self, inp, pred):
        inp_div = self.valid_gen.input_transform.div
        inp_sub = self.valid_gen.input_transform.sub
        norm_q = self.valid_gen.output_transform.scale[self.get_output_var_idx('PHQ')]
        
        # Input
        PS_idx = 300
        LHFLX_idx = 303

        # Output
        PHQ_idx = slice(0, 30)
        PHCLDLIQ_idx = slice(30, 60)
        PHCLDICE_idx = slice(60, 90)
        PRECT_idx = 214
        PRECTEND_idx = 215

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

        # 2. Compute water integral
        WATINT = np.sum(dP_tilde *(pred[:, PHQ_idx] + pred[:, PHCLDLIQ_idx] + pred[:, PHCLDICE_idx]), axis=1)

        # 3. Compute latent heat flux and precipitation forcings
        LHFLX = inp[:, LHFLX_idx] * inp_div[LHFLX_idx] + inp_sub[LHFLX_idx]
        PREC = pred[:, PRECT_idx] + pred[:, PRECTEND_idx]

        # 4. Compute water mass residual
        WATRES = LHFLX - PREC - WATINT

        return np.square(WATRES)


    def ent_res(self,inp,pred):
        inp_div = self.valid_gen.input_transform.div
        inp_sub = self.valid_gen.input_transform.sub
        norm_q = self.valid_gen.output_transform.scale[self.get_output_var_idx('PHQ')]
        
        # Input
        PS_idx = 300
        SHFLX_idx = 302
        LHFLX_idx = 303

        # Output
        PHQ_idx = slice(0, 30)
        PHCLDLIQ_idx = slice(30, 60)
        PHCLDICE_idx = slice(60, 90)
        TPHYSTND_idx = slice(90, 120)
        DTVKE_idx = slice(180, 210)
        FSNT_idx = 210
        FSNS_idx = 211
        FLNT_idx = 212
        FLNS_idx = 213
        PRECT_idx = 214
        PRECTEND_idx = 215
        PRECST_idx = 216
        PRECSTEND_idx = 217

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

        # 2. Compute net energy input from phase change and precipitation
        PHAS = L_I / L_V * (
                (pred[:, PRECST_idx] + pred[:, PRECSTEND_idx]) -
                (pred[:, PRECT_idx] + pred[:, PRECTEND_idx])
        )

        # 3. Compute net energy input from radiation, SHFLX and TKE
        RAD = (pred[:, FSNT_idx] - pred[:, FSNS_idx] -
               pred[:, FLNT_idx] + pred[:, FLNS_idx])
        SHFLX = (inp[:, SHFLX_idx] * inp_div[SHFLX_idx] +
                 inp_sub[SHFLX_idx])
        KEDINT = np.sum(dP_tilde * pred[:, DTVKE_idx], 1)

        # 4. Compute tendency of vapor due to phase change
        LHFLX = (inp[:, LHFLX_idx] * inp_div[LHFLX_idx] +
                 inp_sub[LHFLX_idx])
        VAPINT = np.sum(dP_tilde * pred[:, PHQ_idx], 1)
        SPDQINT = (VAPINT - LHFLX) * L_S / L_V

        # 5. Same for cloud liquid water tendency
        SPDQCINT = np.sum(dP_tilde * pred[:, PHCLDLIQ_idx], 1) * L_I / L_V

        # 6. And the same for T but remember residual is still missing
        DTINT = np.sum(dP_tilde * pred[:, TPHYSTND_idx], 1)

        # 7. Compute enthalpy residual
        ENTRES = SPDQINT + SPDQCINT + DTINT - RAD - SHFLX - PHAS - KEDINT

        return np.square(ENTRES)

    # tgb - 4/18/2019 - Add radiation loss
    def lw_res(self,inp,pred):
        inp_div = self.valid_gen.input_transform.div
        inp_sub = self.valid_gen.input_transform.sub
        norm_q = self.valid_gen.output_transform.scale[self.get_output_var_idx('PHQ')]
        
        # Input
        PS_idx = 300

        # Output
        QRL_idx = slice(120, 150)
        FLNS_idx = 213
        FLNT_idx = 212

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

        # 2. Compute longwave integral
        LWINT = np.sum(dP_tilde *pred[:, QRL_idx], axis=1)

        # 3. Compute net longwave flux from lw fluxes at top and bottom
        LWNET = pred[:, FLNS_idx] - pred[:, FLNT_idx]

        # 4. Compute water mass residual
        LWRES = LWINT-LWNET

        return np.square(LWRES)

    def sw_res(self,inp,pred):
        inp_div = self.valid_gen.input_transform.div
        inp_sub = self.valid_gen.input_transform.sub
        norm_q = self.valid_gen.output_transform.scale[self.get_output_var_idx('PHQ')]
        
        # Input
        PS_idx = 300

        # Output
        QRS_idx = slice(150, 180)
        FSNS_idx = 211
        FSNT_idx = 210

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

        # 2. Compute longwave integral
        SWINT = np.sum(dP_tilde *pred[:, QRS_idx], axis=1)

        # 3. Compute net longwave flux from lw fluxes at top and bottom
        SWNET = pred[:, FSNT_idx] - pred[:, FSNS_idx]

        # 4. Compute water mass residual
        SWRES = SWINT-SWNET

        return np.square(SWRES)
    
    # tgb - 4/18/2019 - energy/mass/etc loss functions in W2/m4
    #def
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

def get_mean_sounding(md, lat_slice=slice(30, 33)):
    return md.reshape_ngeo(
        np.array(md.valid_gen[0][0], dtype=np.float32))[lat_slice].mean((0,1))

def get_jacobian(x, model):
    sess = tf.keras.backend.get_session()
    jac = jacobian(model.output, model.input)
    J = sess.run(jac, feed_dict={model.input: x.astype(np.float32)[None]})
    return J.squeeze()


def plot_jacobian(J, gen, inp_var=None, out_var=None, figsize=(15, 15), ax = None, **kwargs):
    inp_vars = gen.norm_ds.var_names[gen.input_idxs].values
    out_vars = gen.norm_ds.var_names[gen.output_idxs].values
    inp_idxs = np.where(inp_vars == inp_var)[0]
    out_idxs = np.where(out_vars == out_var)[0]
    j = J[out_idxs][:, inp_idxs]

    PP = np.meshgrid(P[-j.shape[1]:], P)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    pc = ax.pcolormesh(PP[0], PP[1], j, **kwargs)
    ax.invert_xaxis(); ax.invert_yaxis()
    fig.colorbar(pc, shrink=0.7, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel(inp_var)
    ax.set_ylabel(out_var)