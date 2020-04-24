"""
This file contains functions used in the main preprocessing script.

Created on 2019-01-23-14-50
Author: Stephan Rasp, raspstephan@gmail.com
tgb - 11/13/2019 - Adding RH and deviation from moist adiabat
"""

from ..imports import *
from ..cam_constants import *

# Set up logging, mainly to get timings easily.
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


# Define dictionary with vertical diffusion terms
diff_dict = {
    'TAP' : 'DTV',
    'QAP' : 'VD01'
}


def compute_bp(ds, var):
    """GCM state at beginning of time step before physics.
    ?BP = ?AP - physical tendency * dt

    Args:
        ds: entire xarray dataset
        var: BP variable name

    Returns:
        bp: xarray dataarray containing just BP variable, with the first time step cut.
    """
    base_var = var[:-2] + 'AP'
    return (ds[base_var] - ds[phy_dict[base_var]] * DT)[1:]  # Not the first time step

def compute_TfromNS(ds):
    return compute_bp(ds,'TBP')-compute_bp(ds,'TBP')[:,-1,:,:]

def compute_RH(ds):
    # tgb - 11/13/2019 - Calculates Relative humidity following notebook 027
    def RH(T,qv,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)
        return Rv*p*qv/(R*esat(T))
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))

    TBP = compute_bp(ds,'TBP')
    QBP = compute_bp(ds,'QBP')

    return RH(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])

def compute_dRH_dt(ds):
# tgb - 12/01/2019 - Calculates dRH/dt following 027 and compute_bp
# tgb - 11/13/2019 - Calculates Relative humidity following notebook 027
    def RH(T,qv,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)
        return Rv*p*qv/(R*esat(T))
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))

    TBP = compute_bp(ds,'TBP')
    QBP = compute_bp(ds,'QBP')

    return (RH(ds['TAP'][1:,:,:,:],ds['QAP'][1:,:,:,:],ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])-\
            RH(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm']))/DT

def compute_TfromMA(ds):
    import pickle
    # tgb - 11/13/2019 - Calculate deviations from moist adiabatic profile following notebook 027
    pathPKL = '/local/Tom.Beucler/SPCAM_PHYS'
    hf = open(pathPKL+'20191113_MA.pkl','rb')
    MA = pickle.load(hf)

    T_MAfit = np.zeros((ds['lev'].size,ds['TS'][1:,:,:].values.flatten().size))
    for iTs,Ts in enumerate(ds['TS'][1:,:,:].values.flatten()):
        T_MAfit[:,iTs] = MA['Ts_MA'][:,np.abs(Ts-MA['Ts_range']).argmin()]

    T_MAfit_reshape = np.moveaxis(np.reshape(T_MAfit,(ds['lev'].size,
                                                      ds['TS'][1:,:,:].shape[0],
                                                      ds['TS'][1:,:,:].shape[1],
                                                      ds['TS'][1:,:,:].shape[2])),0,1)

    return compute_bp(ds,'TBP')-T_MAfit_reshape

def compute_TfromNS(ds):
    return compute_bp(ds,'TBP')-compute_bp(ds,'TBP')[:,-1,:,:]

def compute_TfromTS(ds):
    return compute_bp(ds,'TBP')-ds['TS'][1:,:,:]

def compute_Carnotmax(ds):
    # tgb - 11/15/2019 - Calculates local Carnot efficiency from Tmin to Tmax = max(T) over z
    TBP = compute_bp(ds,'TBP')
    return -(TBP-TBP.max(axis=1))/(TBP.max(axis=1)-TBP.min(axis=1))

def compute_CarnotS(ds):
    # tgb - 11/15/2019 - Calculates local Carnot efficiency from Tmin to Tmax=Ts!=max(T) over z
    TBP = compute_bp(ds,'TBP')
    return -(TBP-ds['TS'][1:,:,:])/(ds['TS'][1:,:,:]-TBP.min(axis=1))

def compute_c(ds, base_var):
    """CRM state at beginning of time step before physics.
    ?_C = ?AP[t-1] - diffusion[t-1] * dt
    Note:
    compute_c() is the only function that returns data from the previous
    time step.
    Args:
        ds: xarray dataset
        base_var: Base variable to be computed
    Returns:
        c: xarray dataarray
    """
    c = ds[base_var].isel(time=slice(0, -1, 1))   # Not the last time step
    if base_var in diff_dict.keys():
        c -= ds[diff_dict[base_var]].isel(time=slice(0, -1, 1)) * DT
    # Change time coordinate. Necessary for later computation of adiabatic
    c['time'] = ds.isel(time=slice(1, None, 1))['time']
    return c

def compute_flux(ds,var):

    base_var = var[:-4]
    P = 1e5*(ds['hyai']+ds['hybi']); # Total pressure [Pa]
    dP = P[0,1:].values-P[0,:-1].values; # Differential pressure [Pa]
#     print('dP',dP.shape)

# tgb - 12/3/2019 - Commenting out these lines because SEF go so close to 0 that
#     print('Base variable is ',base_var)
#     SEF = np.moveaxis(np.tile(ds['LHFLX'][1:,:,:]+ds['SHFLX'][1:,:,:],(ds[base_var].shape[1],1,1,1)),0,1)
#     dP = np.moveaxis(np.tile(dP,(SEF.shape[0],SEF.shape[2],SEF.shape[3],1)),3,1)

# #     print('ds[base_var]',ds[base_var].shape)
# #     print('SEF',SEF.shape)
# #     print('dP',dP.shape)
#     if base_var=='PHQ': return L_V*dP/G*ds[base_var][1:,:,:,:]/np.maximum(10,SEF)
#     elif base_var=='TPHYSTND': return C_P*dP/G*ds[base_var][1:,:,:,:]/np.maximum(10,SEF)

# tgb - 12/3/2019 - Divide by SEF fit based on latitude
    pathPKL = '/home/t/Tom.Beucler/SPCAM/CBRAIN-CAM/notebooks/tbeucler_devlog/PKL_DATA/'
    hf = open(pathPKL+'2019_12_03_SEF_fit.pkl','rb')
    SFfit = pickle.load(hf)
    lfit = SFfit['LHFlogfit']
    sfit = SFfit['SHFfit']

    x = np.log10((compute_bp(ds,'TBP')[:,-1,:,:]).values) # Temperature to define eps coordinate
    LHF = 10**(lfit[0]*x**0+lfit[1]*x**1+lfit[2]*x**2)
    LHF = np.moveaxis(np.tile(LHF,(ds[base_var].shape[1],1,1,1)),0,1)
#     print('LHFmean',np.mean(LHF))
#     print('LHFmax',np.max(LHF))
#     print('LHFmin',np.min(LHF))
    SHF = sfit*np.ones(LHF.shape)
    dP = np.moveaxis(np.tile(dP,(LHF.shape[0],LHF.shape[2],LHF.shape[3],1)),3,1)
#     print('PHQav',np.mean(L_V*dP/G*ds[base_var][1:,:,:,:].values))

    if base_var=='PHQ': return L_V*dP/G*ds[base_var][1:,:,:,:]/(LHF+SHF)
    elif base_var=='TPHYSTND': return C_P*dP/G*ds[base_var][1:,:,:,:]/(LHF+SHF)

def compute_eps(ds, var):
    # tgb - 11/26/2019 - Interpolates variable on epsilon grid

    # tgb - 11/26/2019 - Load data to project on eps grid
    pathPKL = '/home/t/Tom.Beucler/SPCAM/CBRAIN-CAM/notebooks/tbeucler_devlog/PKL_DATA/'
    hf = open(pathPKL+'2019_11_22_imin_TNS_logfit.pkl','rb')
    imfit = pickle.load(hf)['logmodel'][0]
    hf = open(pathPKL+'2019_11_22_eps_TNS_linfit.pkl','rb')
    epfit = pickle.load(hf)['linmodel']

    # Pre-process variables
    #eps_res = 100 # For now hardcode epsilon grid resolution
    eps_res = 30 # tgb - 11/29/2019 - For experiment 134
    # Extract variable name (assumes variable+EPS, e.g. 'TPHYSTNDEPS','PHQEPS','TBPEPS')
    base_var = var[:-3]
    print('Base variable is ',base_var)
    if 'BP' in base_var:
        da = compute_bp(ds, base_var)
    elif 'FLUX' in base_var:
        da = compute_flux(ds,base_var)
    elif base_var=='RH':
        da = compute_RH(ds)
    elif base_var=='dRHdt':
        da = compute_dRH_dt(ds)
    elif base_var=='TfromTS':
        da = compute_TfromTS(ds)
    else: da = ds[base_var][1:]
    daT = compute_bp(ds,'TBP') # Temperature to define eps coordinate

    # 1) Generate eps grid for the neural network with vertical resolution eps_res
    # and the interpolated input array
    eps_NN = np.linspace(0,1,eps_res)
    daI = np.reshape(np.moveaxis(da.values,1,3),(da.shape[0]*da.shape[2]*da.shape[3],30)) # Resized dataset
    daTI = np.reshape(np.moveaxis(daT.values,1,3),(daT.shape[0]*daT.shape[2]*daT.shape[3],30)) # Resized temperature
    x_interp = np.zeros((daI.shape[0],int(eps_res)))

    # 2) Calculates vertical interpolation domain [imin_eval:] and eps coordinate as a function of NS T = T[30]
    for isample in range(daI.shape[0]):
        #rint('isample=',isample,'/',daI.shape[0],'                                                          ',end='\r')
        x = daTI[isample,-1]
        #print('x=',x)
        imin_eval = int(np.rint(10**(imfit[0]*np.log10(x)**0+imfit[1]*np.log10(x)**1+\
                                     imfit[2]*np.log10(x)**2+imfit[3]*np.log10(x)**3+\
                                     imfit[4]*np.log10(x)**4)))
        #print('imin_eval=',imin_eval)
        eps_eval = epfit[:,0]*x**0+epfit[:,1]*x**1+epfit[:,2]*x**2+epfit[:,3]*x**3+epfit[:,4]*x**4
        # tgb - 11/23/2019 - Adds dummy 1 at the end because np.where evaluates y output even if condition false and not returning y
        eps_test = np.minimum(1,np.maximum(eps_eval,0))[imin_eval:]
        eps_eval = np.concatenate((np.minimum(1,np.maximum(eps_eval,0))[imin_eval:][::-1],[1]))

    # 3) Interpolate both T and q to the eps grid for the neural network
    # 3.1) Thermodynamic profiles to interpolate
    # tgb - 11/23/2019 - Adds dummy zero at the end because np.where evaluates y output even if condition false and not returning y
        x_input = np.concatenate((daI[isample,imin_eval:][::-1],[0]))
    # 3.2) Interpolation using searchsorted and low-level weighting implementation
    # The goal is to mimic T_interp = np.interp(x=eps_ref,xp=eps_eval,fp=T_input)
    # If left then T_input[0], if right then T_input[-1], else weighted average of T_input[iint-1] and T_input[iint]
        iint = np.searchsorted(eps_eval,eps_NN)
        x_interp[isample,:] = np.where(iint<1,x_input[0],np.where(iint>(30-imin_eval-1),x_input[30-imin_eval-1],\
                                                                  ((eps_eval[iint]-eps_NN)/\
                                                                   (eps_eval[iint]-eps_eval[iint-1]))*x_input[iint-1]+\
                                                                  ((eps_NN-eps_eval[iint-1])/\
                                                                   (eps_eval[iint]-eps_eval[iint-1]))*x_input[iint]))
    # 4) Converts data back to xarray of the right shape
    if eps_res==100:
        x_interp = np.moveaxis(np.reshape(x_interp,(da.shape[0],da.shape[2],da.shape[3],eps_res)),3,1)+\
        0*xr.concat((da,da,da,da[:,:10,:,:]),'lev')
    elif eps_res==30:
        x_interp = np.moveaxis(np.reshape(x_interp,(da.shape[0],da.shape[2],da.shape[3],eps_res)),3,1)+0*da

    x_interp.__setitem__('lev',eps_NN)

    return x_interp

def compute_adiabatic(ds, base_var):
    """Compute adiabatic tendencies.
    Args:
        ds: xarray dataset
        base_var: Base variable to be computed
    Returns:
        adiabatic: xarray dataarray
    """
    return (compute_bp(ds, base_var) - compute_c(ds, base_var)) / DT

def bflx(shf,lhf,tns):
    '''
    Returns buoyancy flux from sensible heat flux (shf),
    latent heaf flux (lhf)
    and near-surface temperature (tns)
    '''
    cpair = 1.00464e3
    latvap = 2.501e6

    return shf/cpair+0.61*tns*lhf/latvap




## functions different from climate_invariant file
class CrhClass:
    def __init__(self):
        pass

    def eliq(self,T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))

    def eice(self,T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*self.eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))

    def esat(self,T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*self.eliq(T)+(T<T00)*self.eice(T)+(T<=T0)*(T>=T00)*(omega*self.eliq(T)+(1-omega)*self.eice(T))

    def RH(self,T,qv,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        S = PS.shape
        p = 1e5 * np.tile(hyam,(S[0],1))+np.transpose(np.tile(PS,(30,1)))*np.tile(hybm,(S[0],1))

        return Rv*p*qv/(R*self.esat(T))

    def qv(self,dP0,qv0,T,RH,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        S = PS.shape
        p = 1e5 * np.tile(hyam,(S[0],1))+np.transpose(np.tile(PS,(30,1)))*np.tile(hybm,(S[0],1))

        qsat0 = R*self.esat(T)*RH/(Rv*p)

        return (np.sum(qv0*dP0,axis=1)/np.sum(qsat0*dP0,axis=1))>=0.8


    def qsat(self,dP0,qv0,T,P0,PS,hyam,hybm):
        return self.qv(dP0,qv0,T,1,P0,PS,hyam,hybm)



    def dP(self,PS):
        S = PS.shape
        P = 1e5 * np.tile(hyai,(S[0],1))+np.transpose(np.tile(PS,(31,1)))*np.tile(hybi,(S[0],1))
        return P[:, 1:]-P[:, :-1]


def CRH(qv0,T,ps,hyam,hybm):
    crh_class = CrhClass()
    dP0 = crh_class.dP(ps)
    return crh_class.qsat(dP0,qv0,T,P0,ps,hyam,hybm)

#     return (np.sum(qv0*dP0,axis=1)/np.sum(qsat0*dP0,axis=1))>=0.8



def create_stacked_da(ds, vars):
    """
    In this function the derived variables are computed and the right time steps are selected.

    Parameters
    ----------
    ds: mf_dataset with dimensions [time, lev, lat, lon]
    vars: list of input and output variables

    Returns
    -------
    da: dataarray with variables [vars, var_names]
    """
    var_list, names_list = [], []
    for var in vars:
        print('var is ',var)
        if 'EPS' in var:
            da = compute_eps(ds, var)
        elif 'FLUX' in var:
            da = compute_flux(ds,var)
        elif 'BP' in var:
            da = compute_bp(ds, var)
        elif var in ['LHFLX', 'SHFLX']:
            da = ds[var][:-1]
        elif var == 'PRECST':
            da = (ds['PRECSC'] + ds['PRECSL'])[1:]
        elif var == 'PRECT':
            da = (ds['PRECC'] + ds['PRECL'])[1:]
        elif var == 'RH':
            da = compute_RH(ds)
        elif var == 'dRHdt':
            da = compute_dRH_dt(ds)
        elif var == 'TfromMA':
            da = compute_TfromMA(ds)
        elif var == 'TfromNS':
            da = compute_TfromNS(ds)
        elif var == 'Carnotmax':
            da = compute_Carnotmax(ds)
        elif var == 'CarnotS':
            da = compute_CarnotS(ds)
        elif var == 'TfromTS':
            da = compute_TfromTS(ds)
        elif var == 'TfromNS':
            da = compute_TfromNS(ds)
        elif 'dt_adiabatic' in var:
            base_var = var[:-12] + 'AP'
            da = compute_adiabatic(ds, base_var)
        else:
            da = ds[var][1:]
        var_list.append(da)
        nlev = da.lev.size if 'lev' in da.coords else 1
        names_list.extend([var] * nlev)

    concat_da = rename_time_lev_and_cut_times(ds, var_list)

    # Delete unused coordinates and set var_names as coordinates
    concat_da['var_names'] = np.array(names_list).astype('object')
    #names_da = xr.DataArray(names_list, coords=[concat_da.coords['stacked']])
    a = 3
    return concat_da


def rename_time_lev_and_cut_times(ds, da_list):
    """Create new time and lev coordinates and cut times for non-cont steps
    This is a bit of a legacy function. Should probably be revised.

    Args:
        ds: Merged dataset
        da_list: list of dataarrays

    Returns:
        da, name_da: concat da and name da
    """

    ilev = 0
    for da in da_list:
        da.coords['time'] = np.arange(da.coords['time'].size)
        if 'lev' in da.coords:
            da.coords['lev'] = np.arange(ilev, ilev + da.coords['lev'].size)
            ilev += da.coords['lev'].size
        else:
            da.expand_dims('lev')
            da.coords['lev'] = ilev
            ilev += 1

    # Concatenate
    da = xr.concat(da_list, dim='lev')
    # Cut out time steps
    cut_time_steps = np.where(np.abs(np.diff(ds.time)) > 2.09e-2)[0]
    clean_time_steps = np.array(da.coords['time'])
    print('These time steps are cut:', cut_time_steps)
    clean_time_steps = np.delete(clean_time_steps, cut_time_steps)
    da = da.isel(time=clean_time_steps)
    # Rename
    da = da.rename({'lev': 'var_names'})
    da = da.rename('vars')

    return da


def reshape_da(da):
    """

    Parameters
    ----------
    da: dataarray with [time, stacked, lat, lon]

    Returns
    -------
    da: dataarray with [sample, stacked]
    """
    da = da.stack(sample=('time', 'lat', 'lon'))
    return da.transpose('sample', 'var_names')


def preprocess(in_dir, in_fns, out_dir, out_fn, vars, lev_range=(0, 30),split_bflx=False):
    """
    This is the main script that preprocesses one file.

    Returns
    -------

    """
    from os import path
    if in_dir=='None': logging.debug(f'No in_dir so in_fns is set to in_fns')
    else: in_fns = path.join(in_dir, in_fns)
    out_fn_pos = "PosCRH_"+out_fn
    out_fn_neg = "NegCRH_"+out_fn
    out_fn = path.join(out_dir, out_fn)
    out_fn_pos = path.join(out_dir, out_fn_pos)
    out_fn_neg = path.join(out_dir, out_fn_neg)
    logging.debug(f'Start preprocessing file {out_fn}')

    logging.info('Reading input files')
    logging.debug(f'Reading input file {in_fns}')
    ds = xr.open_mfdataset(in_fns, decode_times=False, decode_cf=False, concat_dim='time')

    logging.info('Crop levels')
    ds = ds.isel(lev=slice(*lev_range, 1))

    logging.info('Create stacked dataarray')
    da = create_stacked_da(ds, vars)

    logging.info('Stack and reshape dataarray')
    da = reshape_da(da).reset_index('sample')
    print(split_bflx)
    if(split_bflx):
        logging.info('Splitting the dataset with respect to crh value')
        path = '/home1/07064/tg863631/CBrain_project/CBRAIN-CAM/cbrain/'
        path_hyam = 'hyam_hybm.pkl'

        hf = open(path+path_hyam,'rb')
        hyam,hybm = pickle.load(hf)
        qv0 = da[:,:30]
        T = da[:,30:60]
        ps = da[:,60]
        ## for rh
        if "RH" in vars or "TfromNS" in vars:
            vars_q = ['QBP', 'TBP', 'PS', 'SOLIN', 'SHFLX', 'LHFLX', 'PHQ', 'TPHYSTND', 'FSNT', 'FSNS', 'FLNT', 'FLNS']
            da_q = create_stacked_da(ds, vars_q)
            da_q = reshape_da(da_q).reset_index('sample')
            qv0 = da_q[:,:30]
            T = da_q[:,30:60]
            ps = da_q[:,60]

        mask = CRH(qv0,T,ps,hyam,hybm)
        da_pos = da.where(mask,drop=True)
        logging.info(str(da_pos.shape[0])+' data points found with postive threshold')
        logging.info(f'Save postive CRH dataarray as {out_fn_pos}')
        da_pos.to_netcdf(out_fn_pos)
        da_neg = da.where(np.logical_not(mask),drop=True)
        logging.info(str(da_neg.shape[0])+' data points found with negative threshold')
        logging.info(f'Save negative CRH dataarray as {out_fn_neg}')
        da_neg.to_netcdf(out_fn_neg)


    else:
        logging.info(f'Save dataarray as {out_fn}')
        da.to_netcdf(out_fn)

    logging.info('Done!')


if __name__ == '__main__':
    fire.Fire(preprocess)
