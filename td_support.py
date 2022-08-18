 # -*- coding: utf-8 -*-
"""
Universal time-domain codes.

TODO:
    working example of how to use these functions
    plotting function for multispecies fit

For implementation into pldspectrapy.
Can handle multispecies fitting, each with their own full path characteristics.
You can apply a constraint to match, say, pathlength, pressure, temperature of each.

Created on Tue Nov  5 13:30:40 2019

@author: Nate Malarich
significant modifications by: Scott
"""
# built-in modules
import numpy as np
import matplotlib.pyplot as plt

# Modules from the internet
from lmfit import Model

# In-house modules (in this case a lab-custom version of hapi for accurate high-T)
from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy.pldhapi as hapi

plt.rcParams.update({'font.size':11,'figure.autolayout': True})
'''
First codes for setting up x-axis

'''
def largest_prime_factor(n):
    '''
    Want 2 * (x_stop - x_start - 1) to have small largest_prime_factor
    '''
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

def bandwidth_select_td(x_array, band_fit, max_prime_factor=False, print_value=True):
    '''
    Tweak bandwidth selection for swift time-domain fitting.
    
    Time-domain fit does inverse FFT for each nonlinear least-squares iteration,
    and speed of FFT goes with maximum prime factor.
    
    INPUTS:
        x_array = x-axis for measurement transmission spectrum
        band_fit = [start_frequency, stop_frequency]
        max_prime_factor
    '''
    x_start = np.argmin(np.abs(x_array - band_fit[0]))
    x_stop = np.argmin(np.abs(x_array - band_fit[1]))
       
    len_td = 2 * (np.abs(x_stop - x_start) - 1) # np.fft.irfft operation
    prime_factor = largest_prime_factor(len_td)
    counter = 0 
    
    if max_prime_factor: 
        while prime_factor > max_prime_factor:
            x_stop -= 1
            len_td = 2 * (np.abs(x_stop - x_start) - 1)
            prime_factor = largest_prime_factor(len_td)
            counter += 1
        if print_value: print(str(counter) + ' data points were removed for a greatest prime factor of ' + str(prime_factor))
    
    return x_start, x_stop

def apodize_ig(IG, npts, pts_axis=False, zero_pad=True):
    '''
    symetrically, boxcar apodize the interferogram(s) passed to function, can be 1D or 2D array
    INPUTS:
    IG: inteferograms, 1D or 2D
    npts: number of points in final IG(s) +/- 1 to keep things symmetric
    pts_axis: which axis has the inteferogram data? (pts_axis=1 when IG[which_IG,which_pt_in_IG])
    zero_pad: add zeros in place of data you removed (this is especially useful to keep frequency axis the same)
    '''
       
    if zero_pad: # zero pad (keep the output the same length as the imput)
        IG_apodized = np.zeros(np.shape(IG), dtype=IG.dtype)
        if pts_axis: 
            i_trim = int((np.shape(IG)[pts_axis]-npts)/2)
            if i_trim == 0: IG_apodized = IG
            else: 
                if pts_axis==0: IG_apodized[i_trim:-i_trim,:] = IG[i_trim:-i_trim,:] # data in first element
                elif pts_axis==1: IG_apodized[:,i_trim:-i_trim] = IG[:,i_trim:-i_trim] # data in second element
                else: IG_apodized = None # can't handle 3D IG (yet? why is your IG array so big?)
        else: 
            i_trim = int((len(IG)-npts)/2)
            IG_apodized[i_trim:-i_trim] = IG[i_trim:-i_trim] # only single IG (1D)
    
    else: # don't zero pad (output will be shorter than input
        if pts_axis: 
            i_trim = int((np.shape(IG)[pts_axis]-npts)/2)
            if i_trim == 0: IG_apodized = IG
            else:
                if pts_axis==0: IG_apodized = IG[i_trim:-i_trim,:] # data in first element
                elif pts_axis==1: IG_apodized = IG[:,i_trim:-i_trim] # data in second element
                else: IG_apodized = None # can't handle 3D IG (yet? why is your IG array so big?)
        else: 
            i_trim = int((len(IG)-npts)/2)
            if i_trim == 0: IG_apodized = IG
            else: IG_apodized = IG[i_trim:-i_trim] # only single IG (1D)

    return IG_apodized

def weight_func(len_fd, bl_start, bl_stop=0, etalons = []):
    '''
    Time-domain weighting function, set to 0 over selected baseline, etalon range
    INPUTS:
        len_fd = length of frequency-domain spectrum
        bl_start = number of points at beginning to attribute to baseline
        bl_stop = number of points at the end to attribute to baseline
        etalons = list of [start_point, stop_point] time-domain points for etalon spikes
    '''
    weight = np.ones(2*(len_fd-1))
    
    if bl_start != 0: weight[:bl_start] = 0; weight[-bl_start:] = 0
    if bl_stop != 0: 
        if bl_stop < len_fd-bl_start-2: weight[len_fd-1-bl_stop:len_fd-1+bl_stop]=0 # min of 2 datapoints to fit conditions
        else: please=stop # bl_stop is too big (will remove all useful data in the array)
    
    
    for et in etalons:
        weight[et[0]:et[1]] = 0
        weight[-et[1]:-et[0]] = 0
    
    return weight

'''
Wrapper codes for producing absorption models in time-domain.
To be called using lmfit nonlinear least-squares
'''


def spectra_single(xx, mol_id, iso, molefraction, pressure, 
                   temperature, pathlength, shift, name = 'H2O', flip_spectrum=False):
    '''
    Spectrum calculation for adding multiple models with composite model.
    
    See lmfit model page on prefix, parameter hints, composite models.
    
    INPUTS:
        xx -> wavenumber array (cm-1)
        name -> name of file (no extension) to pull linelist
        mol_id -> Hitran integer for molecule
        iso -> Hitran integer for isotope
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
        flip_spectrum -> set to True if Nyquist window is 0.5-1.0
    
    '''

    nu, coef = hapi.absorptionCoefficient_Voigt(((int(mol_id), int(iso), molefraction),),
            name, HITRAN_units=False,
            OmegaGrid = xx + shift,
            Environment={'p':pressure,'T':temperature},
            Diluent={'self':molefraction,'air':(1-molefraction)})
    if flip_spectrum:
        TD_fit = np.fft.irfft(coef[::-1] * pathlength)
    else:
        TD_fit = np.fft.irfft(coef * pathlength)
        
    # print('y='+str(np.round(molefraction,6)*100)+'%     shift='+str(np.round(shift,6))
    #                +'     T='+str(np.round(temperature,4))+'     P='+str(np.round(pressure,6)))
    
    return TD_fit
    
    
def spectra_single_apodized(xx_orig, mol_id, iso, molefraction, pressure, 
                   temperature, pathlength, shift, 
                   i_fit_start, i_fit_stop, ppig_apod, zero_pad=False,
                   name = 'H2O'):
    '''
    Spectrum calculation for adding multiple models with composite model.
    
    See lmfit model page on prefix, parameter hints, composite models.
    
    INPUTS:
        xx -> wavenumber array (cm-1)
        xx_orig -> wavenumber axis corresponding to the original IG (cm-1)
        name -> name of file (no extension) to pull linelist
        mol_id -> Hitran integer for molecule
        iso -> Hitran integer for isotope
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
        flip_spectrum -> set to True if Nyquist window is 0.5-1.0
        
        i_fit_start -> wavenumber region the spectrum should be trimmed to (before apodization)
        i_fit_stop
        ppig_apod -> how many points to keep in the apodized IG (see apodize_ig)
        baseline -> nominal baseline shape, needs to end at 0s or fft is mad
        
    '''
        
    # generate pre-apodized array of absorption coefficients
    coef = np.zeros(np.shape(xx_orig))

    nu, coef[i_fit_start:i_fit_stop] = hapi.absorptionCoefficient_Voigt(((int(mol_id), int(iso), molefraction),),
            name, HITRAN_units=False,
            OmegaGrid = xx_orig[i_fit_start:i_fit_stop] + shift,
            Environment={'p':pressure,'T':temperature},
            Diluent={'self':molefraction,'air':(1-molefraction)})
            # note that nu is unused (shift gets dropped in reported data, so that it matches measurement)

    # fftshift center burst to center
    trans_psuedo = np.exp(-coef*pathlength)
    IG_coef = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift((np.append(trans_psuedo[::-1], trans_psuedo)))))
    IG_apod = apodize_ig(IG_coef, ppig_apod, zero_pad=zero_pad)
    trans_apod = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(IG_apod))).__abs__()
    
    if zero_pad: 
        trans_trim = trans_apod[int(np.floor(len(trans_apod)/2)):][i_fit_start:i_fit_stop+1] # trim extra half, trim to region of interest

    else: # if we didn't zero_pad, the wvn axis from before doesn't line up.  
        apod_frac = 2*len(xx_orig) / ppig_apod # len(xx_orig) is ppig_orig / 2
        i_apodfit_start = int(np.round(i_fit_start / apod_frac))
        i_apodfit_stop = int(np.round(i_fit_stop / apod_frac))
        trans_trim = trans_apod[int(np.floor(len(trans_apod)/2)):][i_apodfit_start:i_apodfit_stop+1] # trim extra half, trim to region of interest
     
    TD_fit = np.fft.irfft(-np.log(trans_trim))
        
    return TD_fit
    
def spectra_single_lmfit(prefix='', sd = False, apod = False):
    '''
    Set up lmfit model with function hints for single absorption species
    '''
    if sd:
        mod = Model(spectra_sd, prefix = prefix)
    elif apod:
        mod = Model(spectra_single_apodized, prefix = prefix)
    else:
        mod = Model(spectra_single, prefix = prefix)
        
    mod.set_param_hint('mol_id',vary = False)
    mod.set_param_hint('iso', vary = False)
    mod.set_param_hint('pressure',min=0)
    mod.set_param_hint('temperature',min=0)
    mod.set_param_hint('pathlength',min=0)
    mod.set_param_hint('molefraction',min=0,max=1)
    mod.set_param_hint('shift',value=0,min=-.2,max=.2)
    
    if apod: 
        mod.set_param_hint('i_fit_start',min=0,vary = False)
        mod.set_param_hint('i_fit_stop',min=0,vary = False)
        mod.set_param_hint('ppig_apod',min=0,vary = False)
        mod.set_param_hint('zero_pad',value=False,vary = False) # untested
    
    pars = mod.make_params()
    
    # let's set up some default thermodynamics
    pars[prefix + 'mol_id'].value = 1
    pars[prefix + 'iso'].value = 1
    pars[prefix + 'pressure'].value = 640/760
    pars[prefix + 'temperature'].value = 296
    pars[prefix + 'pathlength'].value = 100
    pars[prefix + 'molefraction'].value = 0.01
        
    return mod, pars
    
def spectra_cross_section_lmfit(prefix=''):
    '''
    Set up lmfit model with function hints for cross sectional database
    '''

    mod = Model(spectra_cross_section, independent_vars=['xx', 'xx_HITRAN', 'coef_HITRAN'], prefix = prefix)
        
    mod.set_param_hint('pressure', value=640/760, min=0)
    mod.set_param_hint('temperature', value=296, min=0)
    mod.set_param_hint('pathlength', value = 100, min=0)
    mod.set_param_hint('molefraction', value = 0.01, min=0,max=1)
    mod.set_param_hint('shift', value=0, min=-.2, max=.2)
        
    pars = mod.make_params()
            
    return mod, pars
    

def spectra_cross_section(xx, xx_HITRAN, coef_HITRAN, molefraction, pressure, temperature, pathlength, shift):
    '''
    Spectrum calculation for adding multiple models with composite model.
    
    See lmfit model page on prefix, parameter hints, composite models.
    
    INPUTS:
        xx -> wavenumber array (cm-1)
        ceoffs -> pass in the absorption coefficients from the sigma or xsc HITRAN file
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
    
    TODO - scale by pressure per Amanda's mFID stuff
    
    '''
    
    coef_HITRAN_interp = np.interp(xx, xx_HITRAN + shift, coef_HITRAN)

    coef = coef_HITRAN_interp * hapi.volumeConcentration(molefraction*pressure, temperature)

    TD_fit = np.fft.irfft(coef * pathlength)
    
    # progress report if you would like it: 
    # print('y='+str(np.round(molefraction,2))+'     shift='+str(np.round(shift,4))+'     T='+str(np.round(temperature,0)))
    
    return TD_fit
    
def spectra_sd(xx, mol_id, iso, molefraction, pressure, 
                   temperature, pathlength, shift, name = 'H2O', flip_spectrum=False):
    '''
    Spectrum calculation for adding multiple models with composite model.
    
    See lmfit model page on prefix, parameter hints, composite models.
    
    INPUTS:
        xx -> wavenumber array (cm-1)
        name -> name of file (no extension) to pull linelist
        mol_id -> Hitran integer for molecule
        iso -> Hitran integer for isotope
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
        flip_spectrum -> set to True if Nyquist window is 0.5-1.0
    
    '''

    nu, coef = hapi.absorptionCoefficient_SDVoigt(((int(mol_id), int(iso), molefraction),),
            name, HITRAN_units=False,
            OmegaGrid = xx + shift,
            Environment={'p':pressure,'T':temperature},
            Diluent={'self':molefraction,'air':(1-molefraction)})
    if flip_spectrum:
        TD_fit = np.fft.irfft(coef[::-1] * pathlength)
    else:
        TD_fit = np.fft.irfft(coef * pathlength)
    return TD_fit


'''
Tools for plotting results and baseline removal.
'''
def lmfit_uc(Fit, str_param):
    '''
    Get statistical fitting uncertainty of some fit parameter named str_param
    INPUTS:
        Fit = lmfit Model result object (Fit = mod.fit(...))
        str_param = name of parameter to extract
    warning: some fits are unstable and cannot calculate statistical uncertainties
    '''
    fit_report = Fit.fit_report()
    for line in fit_report.split('\n'):
        if (str_param + ':') in line:
            foo = line.split()
            fit_value = (float(foo[1]))
            fit_uc = (float(foo[3]))
    
    return fit_uc

def plot_fit(x_data, Fit, tx_title = True, plot_td = True, wvn_range=False):
    '''
    Plot lmfit time-domain result.
    INPUTS:
        x_data: x-axis (wavenumber) for fit
        Fit: lmfit object result from model.fit()
    
    '''
    y_datai = Fit.data
    fit_datai = Fit.best_fit
    weight = Fit.weights
    # plot frequency-domain fit
    data_lessbl = np.real(np.fft.rfft(y_datai - (1-weight) * (y_datai - fit_datai)))
    model = np.real(np.fft.rfft(fit_datai))
    # plot with residual
    fig, axs = plt.subplots(2,1, sharex = 'col', gridspec_kw={'height_ratios': [3,1]})
    
    if wvn_range: # trim things down if desired (noisy edges keep messing up my plots)
        i_range = bandwidth_select_td(x_data, wvn_range)
        x_data = x_data[i_range[0]:i_range[1]]
        data_lessbl = data_lessbl[i_range[0]:i_range[1]]
        model = model[i_range[0]:i_range[1]]
        
    axs[0].plot(x_data, data_lessbl, x_data, model)
    axs[1].plot(x_data, data_lessbl - model)
    axs[0].set_ylabel('Absorbance'); #axs[0].legend(['data','fit'])
    axs[1].set_ylabel('Residual'); axs[1].set_xlabel('Wavenumber ($cm^{-1}$)')
    if tx_title:
        t_fit = Fit.best_values['temperature']
        x_fit = Fit.best_values['molefraction']
        axs[0].set_title('Combustor fit T = ' + f'{t_fit:.0f}' + 'K, ' + 
                  f'{100*x_fit:.1f}' + '% H2O')
    if plot_td: 
        # and time-domain fit
        plt.figure()
        plt.plot(fit_datai)
        plt.plot(y_datai - fit_datai)
        plt.plot(weight)
    #    plt.legend(['model','residual','weighting function'])
    
    return data_lessbl

def plot_fit_multispecies(x_data, Fit, plot_td = True, wvn_range=False):
    '''
    Plot baseline-subtracted fit and time-domain fit for multispecies fit.
    
    INPUTS:
        pars_multispecies = lmfit parameters object set up by 
                    "full_pars = ...
                    
    TODO:
        Is multipath_mod in the Fit result?
        What is .data linelist name of each multispecies molecule?
          lmfit seems clunky for automated fit-plotting of multiple species
          from multiple linelists
    '''
    y_datai = Fit.data
    fit_datai = Fit.best_fit
    weight = Fit.weights
    # plot frequency-domain fit
    data_lessbl = np.real(np.fft.rfft(y_datai - (1-weight) * (y_datai - fit_datai)))
    model = np.real(np.fft.rfft(fit_datai))
    # plot with residual
    fig, axs = plt.subplots(2,1, sharex = 'col', gridspec_kw={'height_ratios': [3,1]})
    axs[0].plot(x_data, data_lessbl, x_data, model)
    axs[1].plot(x_data, data_lessbl - model)
    axs[0].set_ylabel('Absorbance'); #axs[0].legend(['data','fit'])
    axs[1].set_ylabel('Residual'); axs[1].set_xlabel('Wavenumber ($cm^{-1}$)')

    # and time-domain fit
    if plot_td:
        plt.figure()
        plt.plot(fit_datai)
        plt.plot(y_datai - fit_datai)
        plt.plot(weight)
    
    
#    # calculate each multispecies frequency-domain fit
#    pars_out = pars_multispecies.copy()
#    for key, val in Fit.best_values.items():
#        pars_out[key].value = val
#    # Actually want to initialize the H2O of the next segment at the 1st segment value
#    full_pars = pars_out.copy()
#    comps = multipath_mod.eval_components(xx=x_data,params=pars_out,CH4name = 'CH4')
    
    return data_lessbl
    
    