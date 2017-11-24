"""Script to stack features and targets and normalize them.

Author: Stephan Rasp
"""

from argparse import ArgumentParser
from netCDF4 import Dataset
import numpy as np
from preprocess_aqua import create_log_str
import os
import pdb


# Define conversion dict
L_V = 2.5e6   # Latent heat of vaporization is actually 2.26e6
C_P = 1e3 # Specific heat capacity of air at constant pressure
conversion_dict = {
    'SPDT': C_P,
    'SPDQ': L_V,
    'QRL': C_P,
    'QRS': C_P,
    'PRECT': 1e3*24*3600 * 1e-3,
    'FLUT': 1. * 1e-5,
}

feature_names = ['TAP', 'QAP', 'dTdt_adiabatic', 'dQdt_adiabatic', 'SHFLX',
                 'LHFLX', 'SOLIN']
target_names = ['SPDT', 'SPDQ', 'QRL', 'QRS', 'PRECT', 'FLUT']

def main(inargs):
    """TODO
    """
    #pdb.set_trace()
    # Open file
    in_file = Dataset(inargs.in_fn)
    mean_file = Dataset(inargs.mean_fn)
    std_file = Dataset(inargs.std_fn)


    # Create new netCDF
    nc_fn = os.path.join(inargs.out_fn)
    print('Out file:', nc_fn)
    rg = Dataset(nc_fn, 'w')
    rg.log = create_log_str()

    n_samples = in_file.dimensions['sample'].size
    rg.createDimension('sample', n_samples)
    rg.createDimension('f_dim', 87)
    rg.createDimension('t_dim', 86)

    rg.createVariable('features', 'float32', ('sample', 'f_dim'))
    rg.createVariable('targets', 'float32', ('sample', 't_dim'))

    # Load and norm files
    ifeat = 0
    itarg = 0

    for v in feature_names:
        print(v)
        if in_file.variables[v].ndim == 2:
            f = in_file.variables[v][:, :]
            plusz = f.shape[1]
        elif in_file.variables[v].ndim == 1:
            f = np.atleast_2d(
                in_file.variables[v][:]
            ).T
            plusz = f.shape[1]

        # normalize
        f = (f - mean_file.variables[v][:]) / std_file.variables[v][:]

        rg.variables['features'][:, ifeat:ifeat + plusz] = f
        ifeat += plusz

    for v in target_names:
        print(v)
        if in_file.variables[v].ndim == 2:
            t = in_file.variables[v][:, :]
            plusz = t.shape[1]
        elif in_file.variables[v].ndim == 1:
            t = np.atleast_2d(
                in_file.variables[v][:]
            ).T
            plusz = t.shape[1]

        # normalize
        t = t * conversion_dict[v]

        rg.variables['targets'][:, itarg:itarg + plusz] = t
        itarg += plusz

    in_file.close()
    mean_file.close()
    std_file.close()
    rg.close()






if __name__ == '__main__':

    p = ArgumentParser()

    p.add_argument('--in_fn',
                   type=str,
                   help='')
    p.add_argument('--out_fn',
                   type=str,
                   help='')
    p.add_argument('--mean_fn',
                   type=str,
                   help='')
    p.add_argument('--std_fn',
                   type=str,
                   help='')
    p.add_argument('--config_file',
                   type=str,
                   help='This is the same config file as for the keras runs.')


    args = p.parse_args()

    main(args)