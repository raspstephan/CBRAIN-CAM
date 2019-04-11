"""
All CAM specific (and other) constants are defined here.

Created on 2019-01-23-14-51
Author: Stephan Rasp, raspstephan@gmail.com
"""

DT = 1800.
L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_F = L_I
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
P0 = 1e5
RHO_L = 1e3

from .imports import hyai, hybi
P = P0*hyai+P0*hybi # Total pressure [Pa]
P = (P[1:] + P[:-1]) / 2 / 100

phy_dict = {
    'TAP': 'TPHYSTND',
    'QAP': 'PHQ',
    'QCAP': 'PHCLDLIQ',
    'QIAP': 'PHCLDICE',
    'VAP': 'VPHYSTND',
    'UAP': 'UPHYSTND'
}