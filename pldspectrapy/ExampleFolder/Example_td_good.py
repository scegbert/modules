# -*- coding: utf-8 -*-
"""
Demonstrate time-domain (cepstral) analysis on laboratory DCS measurement.

Working with previously phase-corrected spectrum from the vc707 "moose" DAQ,
 which has a log file with the frequency counter information to calculate x-axis.

Created on Sat Jan 25 15:56:24 2020

@author: ForAmericanEyesOnly
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from time import time

# lab group codes
from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy as pld
import td_support as td

plt.rcParams.update({'figure.autolayout':True,'lines.linewidth':0.8}) # plot formatting
def rfft(x):
    return np.real(np.fft.rfft(x))

# Fit parameters
band_fit = [6884, 7180] # wavenumber fit region

## load spectrum and calculate frequency axis
file_name = '20200124170754'
trans = np.loadtxt(os.path.join('dcs_spectra',file_name + '.txt'), skiprows=1)
# Get frequency axis from DAQ/counter log file
Log = pld.DAQFilesVC707(os.path.join('dcs_spectra', file_name))
x_wvn_full = pld.mobile_axis(Log)

tic = time() # time-domain-specific fitting starts here

# Convert interesting portion of spectrum to time-domain
start_pnt, stop_pnt = td.bandwidth_select_td(x_wvn_full, band_fit)
if start_pnt < stop_pnt:
    # Normal setup
    x_wvn = x_wvn_full[start_pnt:stop_pnt]
    y_td = np.fft.irfft(-np.log(trans[start_pnt:stop_pnt]))
else:
    # DCS in 0.5-1.0 portion of Nyquist window, need to flip x-axis to fit
    x_wvn = x_wvn_full[start_pnt:stop_pnt:-1]
    y_td = np.fft.irfft(-np.log(trans[start_pnt:stop_pnt:-1]))

## Initialize thermodynamic parameters
mod, pars = td.spectra_single_lmfit()
# lmfit object set up for room-temperature H2O fitting. Customize fit params
pars['temperature'].vary = False
pars['molefraction'].set(value = 1, vary = False)
pars['pressure'].set(value = 0.05, vary = True)
# fit for pathlength, pressure, and Wenzel clock drift (shift)
# can type "pars" into command line to see initial values and which are floating.

pld.db_begin('linelists') # point HAPI (nested function in fit) to linelist file

# produce arbitrary baseline/etalon time-domain weighting function
# can change these values later for cleaner residual spectrum, 
#  but not critical for molefraction fitting.
bl = 50
weight = td.weight_func(len(x_wvn), bl)

## Now make fit and plot results
linelist = 'H2O_PaulLF' # use Paul Schroeder 2018 JQSRT water lines
Fit = mod.fit(y_td, xx = x_wvn, params = pars, weights = weight, name = linelist)
y_lessbl = td.plot_fit(x_wvn, Fit)
print('Time-domain fit in %d seconds' % (time() - tic))

# from time-domain fit, see an etalon at 50
# Add the etalon-removal without changing the fit.
bl = 40
etalons = [[45,55],[90,103]]
weight = td.weight_func(len(x_wvn), bl, etalons)
Fit.weights = weight
y_lessbl = td.plot_fit(x_wvn, Fit)