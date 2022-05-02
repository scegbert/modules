# -*- coding: utf-8 -*-
"""
Testing pldfitting.py object with some gasifier data.

Careful with frequency units on your measurement files.
I try to assume you are working in cm-1, Hz, or nm in the IR. If you are in THz units, expect a bug.

Created on Thu Oct 18 13:38:36 2018

@author: Nate the Average
"""

# Not strictly needed but nice to have
import matplotlib.pyplot as plt
import numpy as np
import os
# And line for nice figure plots
from matplotlib import rcParams; rcParams.update({'figure.autolayout': True})

# Load pldspectrapy from elsewhere on computer
from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy as pld

# Setup parameters

fileName = os.path.join('dcs_spectra','at1000_01_20Perc.txt') # relative path from repository home folder to measurement
                                         # 2-column text file from Schroeder 2018 gasifier data
                                         # 1st column is wavenumber, 2nd column is transmission
# Thermodynamic properties
Temp = 1000 + 273 # Kelvin
Lpath = 4.8       # cm
Pressure = 1      # atm
X_H2O    = 20/100 # 20%, molefraction
# Spectral fitting window
wvn_start = 6900   # cm-1
wvn_stop  = 6920   # cm-1
blOrder  = 15     # number baseline coefficients

Example = pld.Fitting() # initialize object with all the data you'll use

custLineList = False

# Must take two steps to define fitting variables
Example.def_environment(Temp,Pressure,Lpath,1e7/wvn_stop,1e7/wvn_start)
if custLineList:
    # Try 20% H2O from Jinyu Yang custom linelist in file data/Ar_H2O_Labfit.data
    # Can find this hapi-compatible non-Voigt linelist in RiekerLabs Google Drive under SpectralDatabases/MinesArgonData
    Example.def_molecules(['H2O'],[X_H2O],['ArH2O_Labfit']) # try 20% H2O from custom linelist ArH2O_Labfit.data
    Example.molec.md['H2O'][2] = 'SDVoigt'
else:
    # Or try 20% H2O from HitranOnline
    Example.def_molecules(['H2O'],[X_H2O])

# Now load in the transmission spectrum from text file into object
Example.load_spectra(fileName, getFreq = True)

# May want to customize fitting parameters here
Example.bl_order = blOrder # fit 40th order Chebyshev
Example.fit_params['X_H2O'].vary = True
Example.fit_params['Temp'].vary = False # float water molefraction but not temperature

# Now baseline-subtract
Example.baseline_remove()


# Finally fit for H2O
Example.fit()

plt.figure()
plt.plot(Example.x_data, Example.absorption_meas[0], label = 'Data')
plt.plot(Example.x_data, Example.result.best_fit, label = 'Model')
plt.legend(); plt.xlabel('Wavenumber (cm-1)'); plt.ylabel('Absorbance')

# What if I wanted to change some of the source parameters of the H2O model?
# For instance, run the linelist from a custom list called ArH2O_Labfit.data instead of H2O.data from hapi.fetch (HitranOnline)?
Example.molec.md['H2O'][3] = 'ArH2O_Labfit'
# Because this dictionary value is a 4-item list with [molID, iso, lineshape, linelistFile]
#
## I can also repeat the fit now
#Example.result2 = Example.gmodel(Example.y_data, Example.fit_params, xx = Example.x_data, molDict = Example.molec)
## And plot the resulting fit
#plt.plot(Example.x_data, Example.y_data,label = "Measurement")
#plt.plot(Example.x_data, Example.result2.best_fit, label = "Fit result")
#plt.legend(); plt.xlabel("Wavenumber (cm-1)"); plt.ylabel("Absorbance")

print("DONE")
