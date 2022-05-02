# -*- coding: utf-8 -*-
"""
Time-domain fitting code to fit CO2 cell.

Using old lab-comb DCS setup.

Created on Tue Jan 21 13:49:15 2020

@author: Nate the Average
"""

# Not strictly needed but nice to have
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time

# Load pldspectrapy from elsewhere on computer
from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy as pld
import td_support as td

# Load transmission spectrum and frequency-axis
file_name = os.path.join('dcs_spectra','FFT_CO2_best_2avg.txt')
trans = np.loadtxt(file_name)
# Frequency axis, calculated using old 2-CW laser technique
#  self-referenced combs can use cooler functions, including pull from log file
nu_start = 198.9332e12 # spreadsheet entry vLow_Actual (computed with Orbits)
points_per_ig = 240000 # kk2 = nMagic = k
frep_clk = 200.044687332e6 # whichever counter readout you sent to the DAQ for clocking
C = pld.CombTools()
C.wvn_start = 6800
C.wvn_stop = 7000
x_hz = C.freq_axis_2CWlasers(nu_start, frep_clk, points_per_ig)
x_wvn = x_hz / pld.SPEED_OF_LIGHT / pld.M_2_CM

tic = time()

# Convert interesting portion of spectrum to time-domain
C.start_pnt, C.stop_pnt = td.bandwidth_select_td(x_wvn, [C.wvn_start, C.wvn_stop])
x_wvn = x_wvn[C.start_pnt:C.stop_pnt]
y_fd = -np.log(trans[C.start_pnt:C.stop_pnt])
y_td = np.fft.irfft(y_fd)


# Initialize thermodynamic parameters
mod, pars = td.spectra_single_lmfit()
# lmfit object set up for room-temperature H2O fitting. Customize fit params
pars['mol_id'].value = 2 # fit for CO2 (Hitran code 2)
pars['pressure'].vary = False
pars['temperature'].vary = False
pars['molefraction'].set(value = 1, vary = False)
# fit for frequency-shift and pathlength

pld.db_begin('linelists') # point HAPI (nested function in fit) to linelist file

# produce arbitrary baseline/etalon time-domain weighting function
# can change these values later for cleaner residual spectrum, 
#  but not critical for molefraction fitting.
bl = 50
weight = td.weight_func(len(y_fd), bl)

# Now make fit and plot results
Fit = mod.fit(y_td, xx = x_wvn, params = pars, weights = weight, name = 'CO2')
y_lessbl = td.plot_fit(x_wvn, Fit)
print('Time-domain fit in %d seconds' % (time() - tic))

etalons = [[60,70],[125,137]] # etalons from 
weight2 = td.weight_func(len(y_fd), bl, etalons)
#plt.plot(weight2)