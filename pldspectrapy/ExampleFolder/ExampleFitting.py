# -*- coding: utf-8 -*-
"""
Example of fitting room-temperature CO2 cell data with frequency-domain code
Solve for baseline and X_CO2 molefraction.

Starting point for fitting
- 1 phase-averaged FFT in text file
- Cmob lock frequencies from spreadsheet
- Initial guess thermodynamic parameters
- Desired subset of Nyquist window for fitting 

Careful with frequency units on your measurement files.
I try to assume you are working in cm-1, Hz, or nm in the IR. If you are in THz units, expect a bug.

Created on Thu Oct 18 13:38:36 2018

@author: Nate the Average
"""

# Not strictly needed but nice to have
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
# And line for nice figure plots
from matplotlib import rcParams; rcParams.update({'figure.autolayout': True})

# Load pldspectrapy from elsewhere on computer
from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy as pld

#blOrder  = 15     # number baseline coefficients


###################################################
# Example 2: getting phase-corrected FFT and spreadsheet numbers
###################################################

## Input comb spectra
file_name = os.path.join('dcs_spectra','FFT_CO2_best_2avg.txt')
transmission = np.loadtxt(file_name)
# locking spreadsheet parameters, from "shift red" 2CW lasers spreadsheet
nu_start = 198.9332e12 # spreadsheet entry vLow_Actual (computed with Orbits)
points_per_ig = 240000 # kk2 = nMagic = k
frep_clk = 200.044687332e6 # whichever counter readout you sent to the DAQ for clocking

## Thermodynamic properties
# fit room-temperature CO2 band
wvn_start = 6880 # cm-1
wvn_stop = 7000 #cm-1
Temp = 23 + 273 # Kelvin
Lpath = 35.875 * 2.54 # cm
Pressure = 638 / 760 # atm
X_CO2 = 1 # 

tic = time()
##
Ex2 = pld.Fitting()
Ex2.def_environment(Temp,Pressure,Lpath,1e7/wvn_stop,1e7/wvn_start)
Ex2.def_molecules(['CO2'],[X_CO2])
# Alternate transmission data loading
Ex2.data_spectra = transmission
Ex2.num_spectra = 1
Ex2.freq_axis_2CWlasers(nu_start, frep_clk, points_per_ig)
# Slower fitting 
Ex2.baseline_remove() # Makes Ex2.absorption_meas
Ex2.fit()
print('Frequency-domain fitting in %d seconds' % (time() - tic))

plt.figure()
plt.plot(Ex2.x_data, Ex2.absorption_meas[0], label = 'Data')
plt.plot(Ex2.x_data, Ex2.result.best_fit, label = 'Model')
plt.legend(); plt.xlabel('Wavenumber (cm-1)'); plt.ylabel('Absorbance')


## Now repeat fit with higher baseline order
#for ii in range(30):
#    Ex2.fit_params.add('Cheb'+str(ii), value = 0,vary=True)
#Ex2.fit_params['shift'].vary = True
#Ex2.fit_repeat()
#plt.figure()
#plt.plot(Ex2.x_data, Ex2.absorption_meas, label = 'Data')
#plt.plot(Ex2.x_data, Ex2.result2.best_fit, label = 'Model')
#plt.legend(); plt.xlabel('Wavenumber (cm-1)'); plt.ylabel('Absorbance')
#print(f'{Ex2.result2.best_values["X_CO2"]*100:.2f}' + '% expected CO2')