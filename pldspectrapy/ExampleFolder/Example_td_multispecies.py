# -*- coding: utf-8 -*-
"""
Fit spectrum for 2 species.

Uses synthetic data.

Created on Tue Jun 30 10:02:51 2020

@author: Nate the Average
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

band_fit = [6900,6920]
x_wvn = np.arange(band_fit[0], band_fit[1], .01)

mod, pars = td.spectra_single_lmfit('co2') # co2 fit setup
pars['co2mol_id'].value = 2
pars['co2pathlength'].vary = False
pars['co2pressure'].vary = False
pars['co2molefraction'].set(value = .2, vary = True)
pars['co2shift'].vary = False
pars['co2temperature'].vary = True
# second spectral model for h2o
mod2, pars2 = td.spectra_single_lmfit('h2o')
pars2['h2oshift'].vary = False
pars2['h2opathlength'].vary = False
pars2['h2otemperature'].expr = 'co2temperature' # will not fit separately from other temperature
pars2['h2opressure'].vary = False

# and combine spectral models
modfull = mod + mod2
pars.update(pars2)

# simulate the absorption spectrum
pld.db_begin('linelists')
meas = modfull.eval(xx = x_wvn, params = pars, co2name = 'CO2', h2oname = 'H2O_PaulLF')
meas += np.random.randn(len(meas)) * 3e-5

data = modfull.eval_components(xx = x_wvn, params = pars, co2name='CO2', h2oname = 'H2O_PaulLF')

# and fit the synthetic data
weight = td.weight_func(len(x_wvn), 50)
Fit = modfull.fit(meas, xx = x_wvn, params = pars, weights = weight,
                  co2name = 'CO2', h2oname = 'H2O_PaulLF')
td.plot_fit_multispecies(x_wvn, Fit);

# record fitted variables
x_co2 = Fit.best_values['co2molefraction']
x_co2_uc = td.lmfit_uc(Fit, 'co2molefraction')
x_h2o = Fit.best_values['h2omolefraction']
x_h2o_uc = td.lmfit_uc(Fit, 'h2omolefraction')
t_kelvin = Fit.best_values['co2temperature']

print('%.2f+-%.2f%% H2O' % (100*x_h2o, 100*x_h2o_uc))
print('%.2f+-%.2f%% CO2' % (100*x_co2, 100*x_co2_uc))