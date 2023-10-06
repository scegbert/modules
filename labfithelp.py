r'''

these are some funtions that I have made to make working with labfit a little easier 
the goal is to use the labfit engine (exe) while controlling things from Python (not using the GUI)
there is still a long ways to go, but hopefully this is a first step that can be built on
have fun with labfit

scott

TODO
- remove trim function
- streamline folder things
- add datatype to hold information? (wvn, trans, etc.)
    - could this datatype be put in a dictionary?
- updated "floated_line_moved" to open DF of inp and search for the right index (or something better suited for new features)
- nself_quantumJpp update to change any value as a function of another value
- new feature creation (improved method for specifying values)
- update the doublet stuff (would also be wise to make it apparent that it is water specific)

r'''

import os
import subprocess
import shutil

import numpy as np
import matplotlib.pyplot as plt

# from sys import path
# path.append(r'C:\Users\scott\Documents\1-WorkStuff\code\scottcode\modules') # hopefully this occurs in the calling function

import linelist_conversions as db
from hapi import partitionSum # hapi has Q(T) built into the script, with this function to call it

import pandas as pd
import pickle

from time import sleep, time

#%% labfit specific parameters

lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)

lines_header_lwa = 18 # number of lines per header in lwa file

#%% measurement specific parameters

meas_T_round = 50 # round meaurement temperature to the nearest value (allows 905 and 895 to be grouped together if desired)
meas_P_round = 0.5 # round measurement pressure to the nearest value (0.5 groups pressure as 0.5, 1.0, 1.5, 16.5, etc.)

q0offset = 0 # how many 0's come before the actual quantum number pairs (I have seen 0 and 2)

feature_new = 1000000 # base number for new features, not actually linked in the code for now (oops)

#%% 
def trim(data_in, wvn_range):
    r'''
    Overview:
        spectraly trims input dataset
        planning to remove this function (not used enough to be that helpful)
    Returns: 
        data_out = trimmed dataset (same format as input)
    Inputs:
        data_in = DF or list of data to be trimmed, if list, ensure that wvn is in 0th position
        wvn_range = list of range [start, stop]
    r'''
    
    try: 
        data_out = data_in[data_in.nu > wvn_range[0]]  # lower wavenumber limit
        data_out = data_out[data_out.nu < wvn_range[1]]  # upper wavenumber limit    
    
    except: 
        
        try: 
            
            istart = np.argmin(abs(data_in[0,:] - wvn_range[0])) # won't work if you didn't put wvn in the first position
            istop = np.argmin(abs(data_in[0,:] - wvn_range[1]))

            data_out = data_in[:,istart:istop+1] # +1 to get the feature on the edge

        except: 
        
            istart = np.argmin(abs(data_in - wvn_range[0])) # won't work if you didn't put wvn in the first position
            istop = np.argmin(abs(data_in - wvn_range[1]))

            data_out = data_in[istart:istop+1] # +1 to get the feature on the edge
    
    return data_out
#%%
def information_df(d_labfit, bin_name, bins, cutoff_s296, T, d_old=None, df_external_load=False, d_load=False):
    r'''
    Overview:
        pull data out of labfit DTL file (fit parameters and uncertainties)
        add some other information to it (quantum assignments, largely specific to assymetric tops - water)
    Returns: 
        df_load = DF with parameters and uncertainties
    Inputs:
        
    r'''
    
    if df_external_load is False: 

        if d_load is False: d_load = os.path.join(d_labfit, bin_name) # location of measurement file (folder)
        else: d_load = d_load   
        
        wvn_range = bins[bin_name][1:3] # wavenumber range of relevant bin - trimming off any chebyshev buffer that may exist
        
        df_load = trim(db.labfit_to_df(d_load, htp=False), wvn_range) # open and trim old database
        
        if d_old is not None: 
            df_old = trim(db.labfit_to_df(d_old, htp=False), wvn_range) # open and trim old database
            df_load['nu_og'] = df_old.nu
        
    else: df_load = df_external_load
        
    if cutoff_s296 is not None: 
        cutoff_strength_atT = np.zeros((len(df_load.elower), 1))
        cutoff_strength = np.zeros((len(df_load.elower), 1))
        
        df_ratio = pd.DataFrame()
        
        for T_i in sorted(set(T)):
            
            cutoff_strength_atT = strength_T(T_i, df_load.elower, df_load.nu) * cutoff_s296
        
            cutoff_strength = 10**(2*np.log10(cutoff_s296) - np.log10(cutoff_strength_atT)) # reflect across S296
           
            df_ratio['ratio_'+str(T_i)] = np.log10((df_load.sw / cutoff_strength) * (296 / T_i)) # ratio of strength and cuttoff and ideal gas estimate for # molecules (at fixed P and V)
        
        df_load = pd.concat([df_load, df_ratio], axis=1)
        df_load['ratio_max'] = df_ratio.values.max(axis=1)

        if d_old is not None: df_load['ratio_max_og'] = df_load['ratio_max'] + np.log10(df_old.sw / df_load.sw) # log(a*b) = log(a) + log(b) 
    
    else: df_load['ratio_max'] = 0 # check everything if no cuttoff is given
            
    return quantum_assignments(df_load)
#%%
def quantum_assignments(df_load): 
    # improve organization of the quantum information 
    
    quanta = df_load.quanta.str.replace('-',' -').replace('q','').str.split(expand=True) # looking for doublets (watch out for negative quanta without spaces, ie -2-2-2)
    
    try: quanta = quanta.drop(columns=[12]) # if present, the 13th column is a row of NANs from the trailing space. remove if needed
    except: pass
    
    if quanta.isnull().to_numpy().any(): please = stop # if anything got input as None, that means we're missing something. Let's do a quick check for that  
    
    df_load['vp'] = ''
    df_load['vpp'] = ''
    
    which = (quanta[0].astype('int32')>=0)&(quanta[1].astype('int32')>=0)&(quanta[2].astype('int32')>=0)&(
             quanta[3].astype('int32')>=0)&(quanta[4].astype('int32')>=0)&(quanta[5].astype('int32')>=0) # let's ignore negative values
            
    df_load.vp[which] = (quanta[0+q0offset]+quanta[1+q0offset]+quanta[2+q0offset])[which] # extract v' values (add strings, ie 000 or 101s)
    df_load.vpp[which] = (quanta[3+q0offset]+quanta[4+q0offset]+quanta[5+q0offset])[which] # extract v' values (add strings, ie 000 or 101s)
    
    df_load['Jp'] = quanta[6+q0offset].astype(int) # extract J' value
    df_load['Kap'] = quanta[7+q0offset].astype(int) # extract Ka' value
    df_load['Kcp'] = quanta[8+q0offset].astype(int) # extract Kc' value
    
    df_load['Jpp'] = quanta[9+q0offset].astype(int) # extract J" value
    df_load['Kapp'] = quanta[10+q0offset].astype(int) # extract Ka" value
    df_load['Kcpp'] = quanta[11+q0offset].astype(int) # extract Kc" value
    
    Jdelta = df_load['Jp'] - df_load['Jpp']
    
    df_load['m'] = 999
    df_load.m[Jdelta == 1]  =  df_load.Jpp[Jdelta == 1] # m = J''+1 = J' for delta J = 1 (R branch)
    df_load.m[Jdelta == 0]  =  df_load.Jpp[Jdelta == 0] # m = J'' when delta J = 0 (Q branch)
    df_load.m[Jdelta == -1] = -df_load.Jpp[Jdelta == -1] # m = -J'' for delta J = -1 (P branch)

    taup =  df_load['Kap'] +  df_load['Kcp']  - df_load['Jp'] # tau' family (0 or 1)
    taupp = df_load['Kapp'] + df_load['Kcpp'] - df_load['Jpp'] # tau'' family (0 or 1)
        
    df_load['tau'] = taup.astype(str) + taupp.astype(str)
    df_load.tau = df_load.tau.where(df_load.tau.isin(['00','01','10','11']), other='') # only keep posible transitions (possible for us)
    
    df_load['Jdelta'] = Jdelta
    df_load['Kadelta'] = df_load['Kap'] - df_load['Kapp']
    df_load['Kcdelta'] = df_load['Kcp'] - df_load['Kcpp']
    
    # df_load['Jm'] = df_load[['Jp', 'Jpp']].max(axis=1) # max of J
    # df_load['Km'] = df_load[['Kap', 'Kapp']].max(axis=1) # max of Ka (see toth self-broadened widths)
        
    # Wagner Collisional Parameters (Temp) pg 217 bottom
    # Bernath book (spectra of atoms) pg 357 bottom
    
    
    # identify doublets
    
    df_load['doublets'] = np.empty((len(df_load), 0)).tolist()
    df_load['reversed'] = np.empty((len(df_load), 0)).tolist()

    if 'ratio_max' not in df_load: df_load['ratio_max'] = 0 # check everything if no cuttoff is given

    for i_feature in df_load[df_load.ratio_max>-1].index.tolist():
        
        which_doub = (((df_load.vp == df_load.vp[i_feature]) & (df_load.vpp == df_load.vpp[i_feature]) &
                     (df_load.Jp == df_load.Jp[i_feature]) & (df_load.Jpp == df_load.Jpp[i_feature]) & 
                     (df_load.index != i_feature) & (df_load.index < feature_new) &
                     (df_load.local_iso_id == df_load.local_iso_id[i_feature])) 
                     &
                     (((df_load.Kap == df_load.Kap[i_feature]) & (df_load.Kapp == df_load.Kapp[i_feature])) | 
                     ((df_load.Kcp == df_load.Kcp[i_feature]) & (df_load.Kcpp == df_load.Kcpp[i_feature])))
                     & 
                     (abs(df_load.nu - df_load.nu[i_feature]) < 0.2))
        
        # if you get an error like 'Can only compare identically-labeled Series objects', 
        # you probably have two features with the same index (search for i_feature in the REI file)
        
        which_rev = ((df_load.vp == df_load.vp[i_feature]) & (df_load.vpp == df_load.vpp[i_feature]) & # reverse rotation
                     (df_load.Jpp == df_load.Jp[i_feature]) & (df_load.Jp == df_load.Jpp[i_feature]) & 
                     (df_load.Kapp == df_load.Kap[i_feature]) & (df_load.Kap == df_load.Kapp[i_feature]) & 
                     (df_load.Kcpp == df_load.Kcp[i_feature]) & (df_load.Kcp == df_load.Kcpp[i_feature]) & 
                     (df_load.index < feature_new) & (df_load.local_iso_id == df_load.local_iso_id[i_feature])
                     & 
                     (abs(df_load.nu - df_load.nu[i_feature]) < 0.2)) 
        
        df_load['doublets'][i_feature] = df_load[which_doub].index.tolist()
        df_load['reversed'][i_feature] = df_load[which_rev].index.tolist()
            
    return df_load
#%%
def compare_dfs(d_labfit, bins, bin_name, props_which, prop, prop2=False, prop3=False, d_old=False, plots=True):
    r'''
    Overview:
        reads in DTL file with uncertainties
        only keeps parameters that have been floated for a given property
        compares this to another DTL file of your choosing (typically the original file)
        
        d_old = either a filepath to an old (reference) file or a dataframe to compare against
        
    Returns: 
        df_compare = DF where the two DTL file values are compared (side-by-side and differences) where main file prop was floated
        df_props = DF of the main file where main file prop was floated
    Inputs:
        
    r'''
    
    d_new = os.path.join(d_labfit, bin_name)
    wvn_range = bins[bin_name][1:3]
    
    df_new = trim(db.labfit_to_df(d_new, htp=False), wvn_range) # open newest version of the database
    
    if d_old is not False: 
        if type(d_old) == str: # if you gave me a file path, open that file(typically og file)
            
            df_props_old = trim(db.labfit_to_df(d_old, htp=False), wvn_range) # if you dont give it a df, it will use the og one
            df_props_old.insert(7, 'uc_sw_perc_old', df_props_old['uc_sw'] / df_props_old['sw'])
            df_props_old.uc_sw_perc_old.mask(df_props_old.uc_sw < 0, -1, inplace=True)
            
        elif type(d_old) == pd.core.frame.DataFrame: # if you give me an "old" dataframe, use that
            
            df_props_old = d_old.copy()
                
        df_all = df_new.join(df_props_old, how='inner', rsuffix='_old') # join dataframes, add suffix _old to old data
    
    else: 
        
        df_all = df_new.copy()
    
    df_all.insert(7, 'uc_sw_perc', df_all['uc_sw'] / df_all['sw'])
    df_all.uc_sw_perc.mask(df_all.uc_sw < 0, -1, inplace=True)
    
    
    for prop_compare in props_which: 
        
        if prop_compare == props_which[0]: # make the df for the first round
            df_props = df_all[prop_compare].copy().to_frame()

            if d_old is not False: df_compare = df_all.nu.copy().to_frame()
            else: df_compare = None
        
        else: 
            df_props[prop_compare] = df_all[prop_compare]
            if d_old is not False: df_compare[prop_compare] = df_all[prop_compare]
        
        df_props['uc_'+prop_compare] = df_all['uc_'+prop_compare]
        
        if d_old is not False: 
            df_compare[prop_compare + '_old'] = df_props_old[prop_compare]
            df_compare[prop_compare+'_delta'] = df_all[prop_compare] - df_props_old[prop_compare] 
        
            if prop_compare == 'sw': 
            
                df_compare[prop_compare+'_delta_perc'] = (df_all[prop_compare] - df_props_old[prop_compare]) / df_props_old[prop_compare]
            
            df_compare['uc_'+prop_compare] = df_all['uc_'+prop_compare]
            df_compare['uc_'+prop_compare + '_old'] = df_props_old['uc_'+prop_compare]
            
        if prop_compare == 'sw':
        
            df_props['uc_sw_perc'] = df_all['uc_sw_perc']
            
            if d_old is not False: 
                
                df_compare['uc_sw_perc'] = df_all['uc_sw_perc']
                df_compare['uc_sw_perc_old'] = df_props_old['uc_sw_perc_old']

    if prop is not False and prop2 is False: 

        if d_old is not False: 
            df_compare = df_compare[df_compare['uc_'+prop[0]]>-1] # features where prop[0] has changed
        df_props = df_props[df_props['uc_'+prop[0]]>-1] # features where prop[0] has changed
        
    if prop2 is not False and prop3 is False: 
    
        print('you are looking at ' + prop[1] + ' and ' + prop2[1])
        if d_old is not False: 
            df_compare = df_compare[(df_compare['uc_'+prop[0]]>-1) | (df_compare['uc_'+prop2[0]]>0)] # features where prop[0] has changed
        df_props = df_props[(df_props['uc_'+prop[0]]>-1) | (df_props['uc_'+prop2[0]]>0)] # features where prop[0] has changed
        
    if prop3 is not False:
        
        print('you are looking at ' + prop[1] + ' and ' + prop2[1] + ' and ' + prop3[1])
        if d_old is not False: 
            df_compare = df_compare[(df_compare['uc_'+prop[0]]>-1) | (df_compare['uc_'+prop2[0]]>-1) | (df_compare['uc_'+prop3[0]]>-1)] # features where prop[0] has changed
        df_props = df_props[(df_props['uc_'+prop[0]]>-1) | (df_props['uc_'+prop2[0]]>-1) | (df_compare['uc_'+prop3[0]]>-1)] # features where prop[0] has changed
    
    
    if plots: 
        for propi in [prop, prop2, prop3]: 
            if propi is not False: 
                
                df_changed = df_all[df_all['uc_'+propi[0]]>0] # features where propi[0] has changed (comparing back to original, always)

                prop_old = df_changed[propi[0]+'_old']
                prop_new = df_changed[propi[0]]
                prop_delta = prop_new - prop_old

                prop_err = df_changed['uc_'+propi[0]]
                prop_err = prop_err.replace(-1,0)


                plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
                try: plt.vlines(0, min(prop_new), max(prop_new), 'k', '--')
                except: pass

                if propi[0] == 'sw': 
                    prop_delta = prop_delta / prop_new # relative errors
                    plt.errorbar(prop_delta, prop_new, xerr=prop_err / prop_new, ls='none')
                    plt.xlabel('(Updated - Original) / Updated ' + propi[1])
                    plt.yscale('log')
                
                else: 
                    plt.errorbar(prop_delta, prop_new, xerr=prop_err, ls='none')
                    plt.xlabel('Updated - Original ' + propi[1])
                
                for j in df_changed.index:
                    j = int(j)
                    plt.annotate(str(j),(prop_delta[j], prop_new[j]))

                plt.plot(prop_delta, prop_new, 'x')

                plt.ylabel('Updated ' + propi[1])
                
                plt.title(bin_name)
        
    return [df_props, df_compare]
#%%
def labfit_to_spectra(d_labfit, bins, bin_name, og = False, d_load=False):
    r'''
    Overview:
        load LWA file to plot transmission (normalized by Chebyshev and/or zero offset) and residual as functions of wavenumber
        reads in T and P to help with grouping measurments during plotting
    Returns: 
        
    Inputs:
        d_load gives the option to directly input which folder to look at (can still use og if desired)
    r'''
    
    
    if d_load is False: d_load = os.path.join(d_labfit, bin_name)

    if og == True: d_load = os.path.join(d_load, bin_name + '-000-og') # grab the og LWA file from the bin folder
    elif og != False: d_load = og
    
    lwa_all = open(d_load+'.lwa', "r").readlines()
    
    index_meas_all = list(np.where(np.array(lwa_all,copy=False) == lwa_all[0])[0]) # indices where measurement files start
    num_meas = len(index_meas_all) # how many measurements are there?
    
    for i in range(num_meas): 
        
        index_meas = index_meas_all[i]
        try: index_meas_next = index_meas_all[i+1]
        except: index_meas_next = -1 # this is the last measurement (go to the end)
                
        T_i = round((float(lwa_all[index_meas+3].split()[1]) + 273)/meas_T_round)*meas_T_round # round to the nearest meas_T_round
        P_i = round(float(lwa_all[index_meas+3].split()[2])/meas_P_round)*meas_P_round # round to the nearest meas_P_round
        
        lwa_meas = np.genfromtxt(lwa_all[index_meas:index_meas_next], skip_header=lines_header_lwa)
        
        wvn_raw = lwa_meas[:,0]
        trans_raw = lwa_meas[:,1]
        res_raw = lwa_meas[:,2]
        
        wvn_range_cheby = [min(wvn_raw),max(wvn_raw)]
        wvn_range = [bins[bin_name][1], bins[bin_name][2]]
        
        rei_all = open(d_load+'.rei', "r").readlines()
        lines_before_cheby = 2
        
        cheby_num = int(rei_all[1].split()[1])
        cheby_rows = int(np.ceil(cheby_num/5))
        cheby_rows_floats = int(np.ceil(cheby_num/25))
        
        rei_per_spectra = lines_before_cheby + cheby_rows + cheby_rows_floats + 127
        
        cheby_coeffs = rei_all[lines_main_header+i*rei_per_spectra+lines_before_cheby].split()[0:5]
        for j in range(cheby_rows-1): cheby_coeffs.extend(rei_all[lines_main_header+i*rei_per_spectra+lines_before_cheby+(j+1)].split())
        cheby_coeffs = np.array(cheby_coeffs,dtype=float)
        
        cheby_raw = np.polynomial.chebyshev.Chebyshev(cheby_coeffs,domain=wvn_range_cheby)(wvn_raw)

        zero_offseti = float(rei_all[lines_main_header+i*rei_per_spectra+lines_before_cheby+5].split()[0])
         
        trans_raw_cheby = (trans_raw-zero_offseti*100) / cheby_raw 
        
        [wvni, transi, resi, chebyi] = trim(np.vstack((wvn_raw, trans_raw_cheby, res_raw, cheby_raw)), wvn_range)
    
        if i == 0: 

            T = [T_i]
            P = [P_i]
            wvn = [wvni]
            trans = [transi]
            res = [resi]
            cheby = [chebyi]
            zero_offset = [zero_offseti]
            
        else: 
             
            T.append(T_i)
            P.append(P_i)
            wvn.append(wvni)
            trans.append(transi)
            res.append(resi)
            cheby.append(chebyi)
            zero_offset.append(zero_offseti)
       
    return [T, P, wvn, trans, res, wvn_range, cheby, zero_offset]
#%%
def strength_T(T, elower, nu, Tref=296, molec_id = 1, iso = 1):
    r'''
    Overview: 
        strength of feature S(T) (used to compare whether or not strength is above noise floor for a given measurement)
    Returns: 
        linestrengths at S(T) given T, E", and nu (line center)
            inputs can be single value or list
    Inputs:
        T = temperature to be evaluated (Kelvin)
        Tref = reference temperature, 296 for HITRAN (Kelvin)
        elower = lower state energy at which to calculate S(T) (cm-1)
        molec_id, iso = HITRAN molecule reference codes for molecule (default = H2O) and isotopologue (default = most abundent)
        
    r'''
    
    h = 6.62607015E-34 # J/s
    c = 29979245800 # cm/s
    kb = 1.380649E-23 # J/K
    c2 = h*c/kb # second radiation constant (cm K) ~1.44
      
    if type(T) is np.array: T = list(T)# weird hapi preference
    
    Qref = partitionSum(molec_id, iso, Tref) # Total Internal Partition Sum (TIPS) at Tref from HAPI
    Q = np.array(partitionSum(molec_id, iso, T)) # TIPS at T
    
    if type(T) is list: T = np.array(T) # and back again for the division part of things 
    
    boltzman = np.exp(-c2*elower*((1/T)-(1/Tref))) # boltzman populations (fraction of molecules at given lower state energy)
    stimulated = (1-np.exp(-c2*nu/T))/(1-np.exp(-c2*nu/Tref)) # stimulated emission, usually negligible in NIR (nu >> T)
    # see Simeckova, Einstein A-coefficients and statistical weights... and/or https://hitran.org/docs/definitions-and-units/
    
    return Qref/Q * boltzman * stimulated # S(T) = S(Tref) * Qref/Q * boltzman * stimulated 
#%%
def plot_spectra(T,wvn,trans,res,res_og, df=False, offset=2, prop=False, prop2=False, features=False, axis_labels=True, all_names=True):
    r'''
    Overview:
        plots output of labfit_to_spectra grouping things by temperature (or whatever variable you feed in as T)
    Returns: 
        
    Inputs:
        offset = separation between the two plots
        
    r'''
        
    plt.figure(figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
    
    colors = ['gold','coral','violet','royalblue','deepskyblue','cyan']
    
    for T_i in sorted(set(T)): # iterate through all of the temperatures that were measured
        color_i = colors[sorted(set(T)).index(T_i)] # which color for this temperature? 
        if df is not False: plt.plot(df.nu, df['ratio_'+str(T_i)]+100, 'X', markersize=5, label=str(T_i), color=color_i)
        for meas_i in np.where(np.array(T,copy=False) == T_i)[0]: # all the measurements at this temperature
            plt.plot(wvn[meas_i],trans[meas_i], color=color_i)
            plt.plot(wvn[meas_i],res[meas_i]+100+offset, color=color_i)
            if res_og is not False: 
                plt.plot(wvn[meas_i],res_og[meas_i]+100+offset*2, color=color_i)
                plt.plot(wvn[meas_i],res[meas_i] - res_og[meas_i]+100+offset*3, color=color_i)
        
    if df is not False: 
        if 'nu_og' in df.columns: 
            changed = (df.nu != df.nu_og) | (df.ratio_max != df.ratio_max_og)        
            plt.plot(df[changed].nu_og, df[changed].ratio_max_og+100, 'X', markersize=3, label='orig.', color='k')
        
            for index, value in changed.items(): 
                if value: 
                    plt.plot([df.loc[index].nu_og, df.loc[index].nu], [df.loc[index].ratio_max_og+100, df.loc[index].ratio_max+100], color='k')
        
        # overlay the doublets and reversals
        plt.plot(df.nu[df.doublets.astype(bool)], df[df.doublets.astype(bool)].ratio_max+100, 'g+', markersize=10, label='doublets') 
        plt.plot(df.nu[df.reversed.astype(bool)], df[df.reversed.astype(bool)].ratio_max+100, 'bx', markersize=10, label='reversed') 
        
        if prop is not False:
            plt.plot(df[df['uc_'+prop[0]] > -1].nu, 
                     df[df['uc_'+prop[0]] > -1].ratio_max+100, 
                         'mx', markersize=25, label='floated '+prop[1]) # overlay floated features
            
        if prop2 is not False:
            plt.plot(df[df['uc_'+prop2[0]] > -1].nu, 
                     df[df['uc_'+prop2[0]] > -1].ratio_max+100, 
                         'c1', markersize=30, label='floated '+prop2[1]) # overlay floated features
        
        if features is not False:
            plt.plot(df[df.index.isin(features)].nu, df[df.index.isin(features)].ratio_max+100, 'y3', markersize=30, color='m', label='verify') # overlay other features
        
        for j in df.index:
            j = int(j)
            
            if np.round(df.mass[j],2) == 18.01: # watch out for isotopes
                colorj = 'k'
            else: 
                colorj = 'r' # usually below noise floor
            
            doub_rev_names = ''
            if df.doublets.astype(bool)[j]: 
                
                for doublet in df.doublets[j]: 
                    if doublet == df.doublets[j][0]: doub_rev_names += ' d' + str(int(doublet))
                    else: doub_rev_names += ', d' + str(int(doublet))
                
            if df.reversed.astype(bool)[j]: doub_rev_names += ' r' + str(int(df.reversed[j][0]))
                                
            if all_names is False: 
                if features is not False: 
                    if j in features: 
                        plt.annotate('  '+str(j) + doub_rev_names,(df.nu[j], df.ratio_max[j]+100),color=colorj) # add feature identifier for doublets
                
            else: 
                plt.annotate('  '+str(j) + doub_rev_names,(df.nu[j], df.ratio_max[j]+100),color=colorj) # add feature identifier for non-doublets
         
    wvn_min = min(map(min, wvn))
    wvn_max = max(map(max, wvn))
    
    plt.hlines(100, wvn_min, wvn_max, colors='k', linestyles='dashed', linewidth=5)
    plt.hlines(100+offset, wvn_min, wvn_max, colors='k', linestyles='dashed', linewidth=5)
    if res_og is not False: plt.hlines(100+offset*2, wvn_min, wvn_max, colors='k', linestyles='dashed', linewidth=5)
    
    plt.legend(loc='lower right')
    if axis_labels: 
        plt.xlabel('wavenumber $(cm^{-1})$')
        plt.ylabel('Trans and Residual+' + str(100+offset))  
        if res_og is not False: plt.ylabel('Trans, Res +' + str(100+offset) + ', OG Res+' + str(100+offset*2))  

    return
#%% 
def newest_rei(d_folder, bin_name): 
    r'''
    Overview:
        identifies the most recently saved REI file in the saved file folder (the one to use if you are reverting to an older REI)
    Returns: 
        
    Inputs:
        
    r'''
    
    try: # find file with highest number, assumes format B12a-003 or B3-019 and we need to get the -xxx part
        if int(bin_name[1:]) < 9.5: i_start = 3; i_stop = 6 # (try taking off only the first character , ie 'B2')
        else: i_start = 4; i_stop = 7 # index by 1 if double digit bin (ie B12)
    
    except: # if there was a letter at the end of the bin name (ie B2a for air)
        if int(bin_name[1:-1]) < 9.5: i_start = 4; i_stop = 7 # try taking off the first and last character
        else: i_start = 5; i_stop = 8 # index by 1 if double digit bin (ie B12a)

    i = 0
    file_extension = ''
    while file_extension != '.rei': # find the highest numbered rei file (skip over other files like folders and notes)
        i-=1
        file_name = os.listdir(d_folder)[i]
        file_extension = file_name[-4:]
    
    num_file = int(file_name[i_start:i_stop]) # find the file with the highest number in the "save" folder
        
    return [num_file, file_name]

def save_file(d_folder_output, bin_name, d_save_name='', d_folder_input=None, d_og=False): 
    r'''
    Overview:
        save the REI file from the most recent iteration of labfit
        if saving "og" it will save all Labfit files (so you can open and plot them if desired)
    Returns: 
        
    Inputs:
        d_folder_input = where to get the file you want to save
        
        
    r'''   
    
    if d_folder_input is None: d_folder_input = d_folder_output
    
    d_input = os.path.join(d_folder_input, bin_name)
    d_output = os.path.join(d_folder_output, bin_name)

    if d_og is False: 

        [num_file, _] = newest_rei(d_output, bin_name)

        d_output = os.path.join(d_folder_output, bin_name, bin_name + '-' + str(num_file+1).zfill(3) + '-' + d_save_name) # avoid over writing existing files by adding 1 to file name
        
        # LWA first because we delete it before running labfit (if error exists, don't save)
        shutil.copy2(d_input + '.lwa', d_output + '.lwa') # save a copy of the LWA file (for transission and residual info)       
        shutil.copy2(d_input + '.rei', d_output + '.rei') # save a copy of the REI file (with final results
        shutil.copy2(d_input + '.dtl', d_output + '.dtl') # save a copy of the DTL file (for pandas df of uncertainties)
        shutil.copy2(d_input + '.inp', d_output + '.inp') # save a copy of the INP file (to run again if needed)
        
        print('file saved as: ' + bin_name + '-' + str(num_file+1).zfill(3) + '-' + d_save_name)
    
    else: 
        
        if not os.path.exists(d_output): # make a new folder if needed
            os.makedirs(d_output)
        
        # d_file = r'\{}-000-og'.format(bin_name)
        d_file = r'\{}-000-HITRAN'.format(bin_name)
        
        shutil.copy2(d_input + '.lwa', d_output + d_file + '.lwa') 
        shutil.copy2(d_input + '.sho', d_output + d_file + '.sho') # save a copy of all files as og files
        shutil.copy2(d_input + '.rei', d_output + d_file + '.rei') 
        shutil.copy2(d_input + '.plt', d_output + d_file + '.plt') 
        shutil.copy2(d_input + '.inp', d_output + d_file + '.inp') 
        shutil.copy2(d_input + '.dtl', d_output + d_file + '.dtl') 
   
        print('\n\nfile saved as: {}\n\n'.format(d_file))
    
    return
    
#%% 
def float_lines(d_folder, bin_name, features, prop, use_which='rei_new', features_constrain=[], 
                nudge_sd = False, nudge_delta_air = False, d_folder_input=None, features_new=None):
    r'''
    Overview: 
        float given feature for given feature (features can be list, prop needs to be single value)
    Returns: 
        
    Inputs:
        features = list of features to float
        prop = which property to float
        use_which = which file to float the lines (will be saved as the INP)
        features_constrain = groupings of features to constrain [[1,2], [4,5], etc.] prop_feat1 = prop_feat2, prop_feat4 = prop_feat5, etc.
        nudge_sd = SD does best when bumped off of 0 (we expect a value of 0.1, so that's where we put it)
            watch out, sometimes SD floats back to 0 after iterating a few times
        d_folder_input = which folder we want to get the saved REI from
            
    r'''
    
    if prop == False: prop = notgiven # prop is False (pick a feature to float)
    
    d_file = os.path.join(d_folder, bin_name)
    
    if d_folder_input is None: d_folder_input = d_folder
    
    if use_which == 'rei_new': rei_all = open(d_file+'.rei', "r").readlines() # newest REI file (being used by labfit)
    elif use_which == 'inp_new': rei_all = open(d_file+'.inp', "r").readlines() # newest INP file (being used by labfit)
    elif use_which == 'rei_saved': # revert to most recent saved REI file
       
        [_, use_which] = newest_rei(os.path.join(d_folder_input, bin_name), bin_name)
        rei_all = open(os.path.join(d_folder_input, bin_name, use_which), "r").readlines()
   
    elif use_which == 'inp_saved': # revert to most recent saved INP file
   
       [_, use_which] = newest_rei(os.path.join(d_folder_input, bin_name), bin_name)
       rei_all = open(os.path.join(d_folder_input, bin_name, use_which)[:-3] + 'inp', "r").readlines()
   
    else: rei_all = open(use_which+'.rei', "r").readlines() # grab whatever the string tells you to get
    
    print('use_which = ' + use_which)
       
    lines_until_features = lines_main_header + int(rei_all[0].split()[2]) * lines_per_asc # all of the header down to the spectra
       
    for i in features: 
        
        if i > feature_new: 
            i_closest = features_new.loc[i].closest # find a close feature and start your search there
            line = int(lines_until_features + lines_per_feature*i_closest - 2)
            
        else: 
            line = lines_until_features + lines_per_feature*i - 2
        
        if int(rei_all[line-2].split()[0]) != i: # check if the feature changed places with a neighbor (nu)
            line = floated_line_moved(line, i, rei_all, lines_per_feature)
        
        rei_all[line] = rei_all[line][0:prop[2]*3] + '0' + rei_all[line][prop[2]*3+1:] # 0 means to float this parameter
        
        if prop[0]=='sd_self' and nudge_sd: 
            
            rei_all[line-1] = rei_all[line-1][0:-6] + '1' + rei_all[line-1][-5:] # change speed dependence from 0 to  0.10000 (recommendation of Brian)
            
        if prop[0]=='delta_air' and nudge_delta_air: 
        
            # set all air shifts to 0 (linear -> exp.), shift exponenets to 1.0
            rei_all[line-2] = rei_all[line-2][:72] + ' -0.0001000 ' + rei_all[line-2][84:]
            
    if prop[0]=='sd_self' and nudge_sd: print('Speed dependence was nudged') # only print 1 time
    if prop[0]=='delta_air' and nudge_delta_air: print('Delta air was nudged') # only print 1 time
        
    notfloated = []
    if len(features_constrain) > 0:
        print('constraint are being used (use_which = ' + use_which + ')')

        aux_total = int(rei_all[1].split()[5]) # number of aux parameters
        constraint_total = int(rei_all[1].split()[6]) # number of constraints

        if prop[0] == 'nu' or prop[0] == 'sw': # these features will require creating more auxiliary parameters

            if constraint_total == 0: rei_aux = rei_all.copy() # aux come before constraints and indexing from 0 does weird stuff
            else: rei_aux = rei_all[:-constraint_total*2]

            for doublet in features_constrain: 
                  
                if prop[0] == 'nu': aux_name = ' ' + str(doublet[0]) + '_' + str(doublet[1]) + '_LineCenterOffset'
                elif prop[0] == 'sw': aux_name = ' ' + str(doublet[0]) + '_' + str(doublet[1]) + '_LineStrengthRatio'
          
                aux_total += 1 # reference number for this auxiliary parameter     
                rei_aux.append(aux_name.ljust(46 - int(np.floor(np.log10(aux_total)))) + str(aux_total) + '\n') # reference number with spacing
            
                doublet_value = np.zeros(2)
                j=0
                
                for doubleti in doublet: 
                                        
                    line = lines_until_features + lines_per_feature*doubleti - 2

                    if int(rei_all[line-2].split()[0]) != doubleti: # if the feature changed places with a neighbor (change in nu)
                        line = floated_line_moved(line, doubleti, rei_all, lines_per_feature)
                                        
                    if prop[0] == 'nu': doublet_value[j] = rei_all[line-2].split()[3] # line-2 = feature value, line = float 1's and 0's
                    elif prop[0] == 'sw': doublet_value[j] = rei_all[line-2].split()[4]

                    # print('     ' + str(line) + '     ' + str(doubleti) + '     ' + str(doublet_value[j]))
                    j += 1
                
                if prop[0] == 'nu': aux_value = '     ' + ('%.10e' % (doublet_value[1] - doublet_value[0])).replace('e','D') # offset in weird fortran notation with 5 spaces
                elif prop[0] == 'sw': aux_value = '     ' + ('%.10e' % (doublet_value[1] / doublet_value[0])).replace('e','D') # line strength ratio
                
                rei_aux.append(aux_value.ljust(28) + '1\n')
            
            if constraint_total > 0: rei_aux.extend(rei_all[-constraint_total*2:]) # tack the constraints back on the end and reset back to rei_all
            rei_all = rei_aux.copy()
            
        
        for doublet in features_constrain: 
            if doublet[0] not in features or doublet[1] not in features: 
                print(doublet)
                notfloated.append(doublet) 
            
            if prop[0] == 'nu' or prop[0] == 'sw': # these features will require creating more auxiliary parameters
                aux_which = aux_total - len(features_constrain) + features_constrain.index(doublet) + 1 # which aux would we reference (overwritten if not nu or sw)
                constraint_str = '       50'.ljust(17 - int(np.floor(np.log10(aux_which)))) + str(aux_which) + '        1\n'
            
            constraint_total += 1 # reference number for this constraint
            if prop[0] == 'nu': rei_all.append('       1'.ljust(32 - int(np.floor(np.log10(constraint_total)))) + str(constraint_total) + '\n') # 1 for addition
            elif prop[0] == 'sw': rei_all.append('       3'.ljust(32 - int(np.floor(np.log10(constraint_total)))) + str(constraint_total) + '\n') # 3 for multiplication
            else: 
                rei_all.append('       8'.ljust(32 - int(np.floor(np.log10(constraint_total)))) + str(constraint_total) + '\n') # 8 for equality
                constraint_str = '\n' # no auxiliary parameter to refer to
            
            rei_all.append('      ' + str(prop[3]).ljust(10 - int(np.floor(np.log10(doublet[1])))) + str(doublet[1]) + '        1       ' + 
                                      str(prop[3]).ljust(10 - int(np.floor(np.log10(doublet[0])))) + str(doublet[0]) + '        1' + constraint_str)
        
               
        rei_all[1] = rei_all[1][:26]+ str(aux_total).rjust(7) + str(constraint_total).rjust(7) + rei_all[1][40:] # update number of aux and constraints at top of file
    
    if len(notfloated) > 0: 
        print('\n\n     you forgot to float features that you are constraining\n')
        print(notfloated)
        print('\n')
        please = stophere # add this feature to your floated list and try again
    
    open(d_file + '.inp', 'w').writelines(rei_all)
    
    return
# %% 
def floated_line_moved(line, i, rei_all, lines_per_feature):
    r'''
    Overview: 
        if a line has changed places with it's neighbor (they are listed by center position not index)
        go find the line with the feature
    Returns: 
        
    Inputs:    
        line = line in text file where the feature should have been (good starting point)
        i = the number of that feature
        rei_all = the rei file in list of lines from text file
        lines_features_total = lines per feature (4)
    
    r'''
    
    # print('previously listed feature swapped with neighbor')
    line_missing = True 
    j = 0

    while line_missing: # check the closest features (usually they're just the neighbor, consider double checking if you're concerned)
        
        # print('     looking at feature index +/- ' + str(j+1))
        line_up = line + j*lines_per_feature + 2 # look at higher wavenumbers
        if int(rei_all[line_up].split()[0]) == i: 
            line = line_up + 2
            line_missing = False
        
        j += 1        
        
        line_down = line - j*lines_per_feature - 2 # at lower wavenumbers (alternating)
        if int(rei_all[line_down].split()[0]) == i:                 
            line = line_down + 2
            line_missing = False
        
        # print(str(int(rei_all[line_up].split()[0])) + '     ' + str(int(rei_all[line_down].split()[0]))) # who did you look at?
        
        if j > 750: # we added 750 features, it could be pretty far away for air water :(
            print('   *****   I think we missed feature ' + str(i) + ' (we tried to check the nearest ' + str(j) + ' features)   *****   ')
            please = stophere # feature moved too far, lets stop and regroup
        
    return line

#%%

def unfloat_lines(d_folder, bin_name, features_keep, features_keep_doublets, d_folder_input=None):
    
    d_file = os.path.join(d_folder, bin_name)
    
    if d_folder_input is None: d_folder_input = d_folder
    
    [_, use_which] = newest_rei(os.path.join(d_folder_input, bin_name), bin_name)
    rei_latest = open(os.path.join(d_folder_input, bin_name, use_which), "r").readlines()
    lines_until_features_latest = lines_main_header + int(rei_latest[0].split()[2]) * lines_per_asc # all of the header down to the spectra
    
    rei_og = open(os.path.join(d_folder_input, bin_name, bin_name + '-000-og')+'.rei', "r").readlines()
    lines_until_features_og = lines_main_header + int(rei_og[0].split()[2]) * lines_per_asc # all of the header down to the spectra
    
    print('\n\n\n **********     dont forget to include doublet relationships manually     ********** ')

    for i in features_keep: 
            
        line_latest = lines_until_features_latest + lines_per_feature*i - 2
        if int(rei_latest[line_latest-2].split()[0]) != i: # check if the feature changed places with a neighbor (nu)
            line_latest = floated_line_moved(line_latest, i, rei_latest, lines_per_feature)
            
        line_og = lines_until_features_og + lines_per_feature*i - 2
    
        if int(rei_og[line_og-2].split()[0]) != i: # check if the feature changed places with a neighbor (nu)
            line_og = floated_line_moved(line_og, i, rei_og, lines_per_feature)    
           
        rei_og[line_og-2:line_og+1] = rei_latest[line_latest-2:line_latest+1]
        
        for doublet in features_keep_doublets: 
            if i in doublet:
                print(i)
                print(rei_latest[line_latest])
        
    open(d_file + '.inp', 'w').writelines(rei_og)
        
    return

#%%

def add_features(d_folder, bin_name, features_new, use_which='rei_new', d_folder_input=None, new_type='big'):
    r'''
    Overview: 
        add a new feature to the INP file, currently numbers that feature starting at 1,XX0,001 with the bin number as XX
        
        if you're getting weird errors on features that should be plenty big for fitting, you might need to increase S296 to snag them 
        if they're below noise floor in first iteration, it gets mad as it varies parameters it doesn't think it can see
        
    Returns: 
        
    Inputs:
        
    r'''
    
    d_file = os.path.join(d_folder, bin_name)

    if use_which == 'rei_new': rei_all = open(d_file+'.rei', "r").readlines() # newest REI file (being used by labfit)
    elif use_which == 'inp_new': rei_all = open(d_file+'.inp', "r").readlines() # newest INP file (being used by labfit)
    elif use_which == 'rei_saved': # revert to most recent saved REI file

        if d_folder_input is None: d_folder_input = d_folder
        
        num_file = int(os.listdir(os.path.join(d_folder_input, bin_name))[-1][3:6]) # find the file with the highest number in the "save" folder  
        use_which = os.listdir(os.path.join(d_folder_input, bin_name))[-1][:-3]+'rei' # make sure file ends in REI (issues with first file)
        rei_all = open(os.path.join(d_folder_input, bin_name, use_which), "r").readlines()

    else: rei_all = open(use_which+'.rei', "r").readlines() # grab whatever the string tells you to get
    
    print('use_which = ' + use_which)
    
    lines_until_features = lines_main_header + int(rei_all[0].split()[2]) * lines_per_asc # all of the header down to the spectra
    features_total = int(rei_all[0].split()[3])
    lines_features_total = 4 * features_total

    rei_updated = rei_all[:lines_until_features + lines_features_total] # all lines with features
    rei_constraints = rei_all[lines_until_features + lines_features_total:] # lines at the bottom with constraints (could be  - and probably is - nothing)
        

    for i in range(len(features_new)): 
        
        feature_id = new_feature_number(bin_name, new_type, i)
        
        if new_type == 'big_all': 
            
            feature_add = [' ' + str(feature_id) + '  1 1  ' + str(features_new[i]) + # guesses at values for water (will update later)
                           '00000  0.50000E-29   0.07000  5000.0000000  0.7000  -0.0100000  0.00000000   18.0106\n',
                           '   0.20000  0.2000  0.0000000  0.00000000    0.00000    0.00000    0.12000\n',
                           '   0  0  1  0  1  1  1  1  0  1  1  1  1  1  1\n',
                           '//  0  0      0 0 0          0 0 0  0  0  0        0  0  0  \n']
        
        if new_type == 'big_nsg': 
            
            feature_add = [' ' + str(feature_id) + '  1 1  ' + str(features_new[i]) + # guesses at values for water (will update later)
                           '00000  0.50000E-29   0.07000  5000.0000000  0.7000  -0.0100000  0.00000000   18.0106\n',
                           '   0.20000  0.2000  0.0000000  0.00000000    0.00000    0.00000    0.12000\n',
                           '   0  0  1  1  1  1  1  1  0  1  1  1  1  1  1\n',
                           '//  0  0      0 0 0          0 0 0  0  0  0        0  0  0  \n']
        
        if new_type == 'big_ns': 
            
            feature_add = [' ' + str(feature_id) + '  1 1  ' + str(features_new[i]) + # guesses at values for water (will update later)
                           '00000  0.50000E-29   0.07000  5000.0000000  0.7000  -0.0100000  0.00000000   18.0106\n',
                           '   0.20000  0.2000  0.0000000  0.00000000    0.00000    0.00000    0.12000\n',
                           '   0  0  1  1  1  1  1  1  1  1  1  1  1  1  1\n',
                           '//  0  0      0 0 0          0 0 0  0  0  0        0  0  0  \n']

        elif new_type == 'small_all':  # don't float lower state energy for features that are too small (ie only seen at one temperature)
            
            feature_add = [' ' + str(feature_id) + '  1 1  ' + str(features_new[i]) + # guesses at values for water (will update later)
                           '00000  0.10000E-30   0.07000  5000.0000000  0.7000  -0.0100000  0.00000000   18.0106\n',
                           '   0.20000  0.2000  0.0000000  0.00000000    0.00000    0.00000    0.12000\n',
                           '   0  0  1  0  1  1  1  1  0  1  1  1  1  1  1\n',
                           '//  0  0      0 0 0          0 0 0  0  0  0        0  0  0  \n']

        elif new_type == 'small_nsg':  # don't float lower state energy for features that are too small (ie only seen at one temperature)
            
            feature_add = [' ' + str(feature_id) + '  1 1  ' + str(features_new[i]) + # guesses at values for water (will update later)
                           '00000  0.10000E-30   0.07000  5000.0000000  0.7000  -0.0100000  0.00000000   18.0106\n',
                           '   0.20000  0.2000  0.0000000  0.00000000    0.00000    0.00000    0.12000\n',
                           '   0  0  1  1  1  1  1  1  0  1  1  1  1  1  1\n',
                           '//  0  0      0 0 0          0 0 0  0  0  0        0  0  0  \n']
                           
        elif new_type == 'small_ns':  # don't float lower state energy for features that are too small (ie only seen at one temperature)
            
            feature_add = [' ' + str(feature_id) + '  1 1  ' + str(features_new[i]) + # guesses at values for water (will update later)
                           '00000  0.10000E-30   0.07000  5000.0000000  0.7000  -0.0100000  0.00000000   18.0106\n',
                           '   0.20000  0.2000  0.0000000  0.00000000    0.00000    0.00000    0.12000\n',
                           '   0  0  1  1  1  1  1  1  1  1  1  1  1  1  1\n',
                           '//  0  0      0 0 0          0 0 0  0  0  0        0  0  0  \n']
            # E" = 3000, sw 0.50000E-27
        
        rei_updated.extend(feature_add.copy())
        
    rei_updated.extend(rei_constraints)
        
    features_total_updated = features_total + len(features_new)
    rei_updated[0] = rei_updated[0][:33] + str(features_total_updated) + rei_updated[0][38:]
    
    open(os.path.join(d_folder, bin_name + '.inp'), 'w').writelines(rei_updated)

    return 
#%%
def new_feature_number(bin_name, new_type, i): 
    
    feature_id = feature_new + int(bin_name[1:]) * 1e4 + i + 1
    if new_type == 'big_all': feature_id += 0
    elif new_type == 'big_nsg': feature_id += 100
    elif new_type == 'big_ns': feature_id += 200
    elif new_type == 'small_all': feature_id += 1000
    elif new_type == 'small_nsg': feature_id += 1100
    elif new_type == 'small_ns': feature_id += 1200  
    else: throw_an =error # you are trying to make a new kind of new feature. stop it. 

    return int(feature_id)
#%%
def shrink_feature(df_shrink, cutoff_s296, T): 

    df_sw = pd.DataFrame()

    for T_i in sorted(set(T)):
        
        cutoff_strength_atT = strength_T(T_i, df_shrink.elower, df_shrink.nu) * cutoff_s296
        cutoff_strength = 10**(2*np.log10(cutoff_s296) - np.log10(cutoff_strength_atT)) # reflect across S296
       
        df_sw['sw_'+str(T_i)] = 10 * cutoff_strength * T_i / 296 # ratio of strength and cuttoff and ideal gas estimate for # molecules (at fixed P and V)

    return df_sw.values.min(axis=1)

#%%
def bin_ASC(d_load, base_name, d_save, bins, bin_name):
    r'''
    Overview: 
        update a generic INP file (with the correct ASC references) to only process a given spectral region (bin)
        good for automatically breaking things up into bins
    Returns: 
        
    Inputs:
        
    r'''
       
    bin_start = '%.4f' % (bins[bin_name][1] + bins[bin_name][0])
    bin_stop = '%.4f' % (bins[bin_name][2] + bins[bin_name][3])
    
    inp_all = open(os.path.join(d_load, base_name + '.inp'), "r").readlines()
    
    inp_all[0] = '    ' + bin_start + '   ' + bin_stop + inp_all[0][25:]  # only replacing the range
    
    open(os.path.join(d_save,bin_name) + '.inp', 'w').writelines(inp_all)

    return

def bin_ASC_cutoff(d_load, base_name, d_save, bins, bin_name, d_cutoff_locations, d_conditions, ASC_sniffer=False):
    r'''
    Overview: 
        A more complicated version of bin_asc that can identify asc files needed and create fancy inp files
        where a single measurement has been broken up into multiple ASC's (esp. for a low-transmission cutoff)
        Will also work when measurement not broken into multiple ASC (but I'm still keeping the other function due to it's readability)
    Returns: 
        
    Inputs:
        
    r'''
    
    which_ASC_files = [] # how many ASC files is this INP file using? (for help with sniffer test)
       
    cheby_min = 2 # min cheby coeffs for small regions (const. + slope)
    width_at_min = 1 # 4 cm-1 and less = cheby_min

    cheby_max = 7 # most cheby coeffs we want to use
    width_at_max = 14 # 10 cm-1 and more = cheby_max
    
   
    bin_start = '%.4f' % (bins[bin_name][1] + bins[bin_name][0])
    bin_stop = '%.4f' % (bins[bin_name][2] + bins[bin_name][3])

    inp_all = open(os.path.join(d_load, base_name + '.inp'), "r").readlines()

    inp_all[0] = '    ' + bin_start + '   ' + bin_stop + inp_all[0][25:]  # update the range
    
    f = open(d_cutoff_locations, 'rb')
    cutoff_locations = pickle.load(f)
    f.close() 
    
    if ASC_sniffer is False or ASC_sniffer is True: 
        
        for meas_key in reversed(d_conditions):  # keeps things in reverse order so indexing doesn't change for unprocessed files (working up from bottom)

            print(meas_key)
            
            index_meas_file = (d_conditions.index(meas_key))*lines_per_asc + lines_main_header # where is the .asc line?

            if len(cutoff_locations[meas_key]) == 1: # if the measurement file is never broken into parts

                inp_all[index_meas_file] = cutoff_locations[meas_key][0][0] + '\n' # insert the updated name in that location
                
            else: # we need to choose which subfiles to use, possibly mutiple of them
                          
                asc_limits = np.array(cutoff_locations[meas_key])[:,1:3].astype(float) # all wvn bounds for this measurement file
                
                # if you're getting an error here and are working with the edge bins, you need to make sure their boundaries are within the ASC limits
                asc_start = np.where((asc_limits[:,0] < bins[bin_name][1]) & (asc_limits[:,1] > bins[bin_name][1]))[0][0] # which asc file has the starting wavenumber?
                asc_stop = np.where((asc_limits[:,0] < bins[bin_name][2]) & (asc_limits[:,1] > bins[bin_name][2]))[0][0] # which asc file has the ending wavenumber?
                                        
                inp_all[index_meas_file] = cutoff_locations[meas_key][asc_start][0] + '\n' # insert the first (maybe only) updated name in that location
                            
                width = min([cutoff_locations[meas_key][asc_start][2] - bins[bin_name][1], # either from start of bin to end of first asc
                             bins[bin_name][2] - bins[bin_name][1]])   # or the width of the whole bin
                
                cheby_num = (cheby_max - cheby_min) / (width_at_max - width_at_min) * (width - width_at_min) + cheby_min
                cheby_num_int = 7 - int(max([min([cheby_num, cheby_max]), cheby_min]))
                cheby_floats_og = inp_all[index_meas_file + 6]
                inp_all[index_meas_file + 6] = cheby_floats_og[::-1].replace('0','1',cheby_num_int)[::-1] # change number of floated parameters in cheby            
                
                print('          {} - {} cm-1 wide ({} cheby coeffs)'.format(cutoff_locations[meas_key][asc_start][0], np.round(width,1), 7-cheby_num_int))

                if ASC_sniffer is True: which_ASC_files.append(cutoff_locations[meas_key][asc_start][0])           

                for asc_iter in range(asc_start+1,asc_stop+1): # there is a bin break in the middle of this file. let's add a few instances of the asc input and save it
               
                    if asc_iter == asc_start+1: # for the first iteration, grab the info you need to copy
                        inp_meas = inp_all[index_meas_file+1:index_meas_file+lines_per_asc] # snag the entire segment of .inp dedicated to this .asc (except file part)
                        inp_below_asc = inp_all[index_meas_file+lines_per_asc:] # snag all lines below this segment (will append later)
                        inp_all = inp_all[0:index_meas_file+lines_per_asc] # remove the end lines (now that we've saved them elsewhere)
                        num_asc = int(inp_all[0].split()[2])
                
                    num_asc += 1 # add one asc file every time we loop, we'll put this into the final inp file
                    
                    width = min([cutoff_locations[meas_key][asc_iter][2] - cutoff_locations[meas_key][asc_iter][1], # either width of the asc file 
                                 (bins[bin_name][2] + bins[bin_name][3]) - cutoff_locations[meas_key][asc_iter][1], # or from start of asc file to the end of the bin
                                 cutoff_locations[meas_key][asc_iter][2] - (bins[bin_name][1] + bins[bin_name][0])]) # or from the start of the bin to the end of the ASC
                    
                    cheby_num = (cheby_max - cheby_min) / (width_at_max - width_at_min) * (width - width_at_min) + cheby_min
                    cheby_num_int = 7 - int(max([min([cheby_num, cheby_max]), cheby_min]))
                    inp_meas[5] = cheby_floats_og[::-1].replace('0','1',cheby_num_int)[::-1] # change number of floated parameters in cheby    
                    
                    inp_all.append(cutoff_locations[meas_key][asc_iter][0] + '\n') # insert the asc name
                    inp_all.extend(inp_meas) # add all of the asc measurement conditions information for that file
                                                 
                    print('          {} - {} cm-1 wide ({} cheby coeffs)'.format(cutoff_locations[meas_key][asc_iter][0], np.round(width,1), 7-cheby_num_int))

                    if ASC_sniffer is True: which_ASC_files.append(cutoff_locations[meas_key][asc_iter][0])           
                           
                    if asc_iter == asc_stop: 
                        inp_all.extend(inp_below_asc) # add back everything that was below the asc file we were working on
                        inp_all[0] = inp_all[0][0:27] + str(num_asc).rjust(3) + inp_all[0][30:]
    
    else: 
        
        meas_key = ASC_sniffer[5:].split('.')[0].replace('_', ' ').rsplit(' ',1)[0].replace('-','_') # remove .ASC and other number info to get the measurement file name
        
        for i, val in enumerate(cutoff_locations[meas_key]):
            if val[0] == ASC_sniffer: asc_iter = i
        
        index_meas_file = lines_main_header # where is the .asc line? (for the sniffer, there is only one at the top)
    
        inp_all = open(os.path.join(d_load, base_name + '_sniffer.inp'), "r").readlines()
    
        inp_all[0] = '    ' + bin_start + '   ' + bin_stop + inp_all[0][25:]  # update the range
    
        
        inp_all[index_meas_file] = cutoff_locations[meas_key][asc_iter][0] + '\n' # insert the first (maybe only) updated name in that location
                    
        width = min([cutoff_locations[meas_key][asc_iter][2] - cutoff_locations[meas_key][asc_iter][1], # either width of the asc file 
                     (bins[bin_name][2] + bins[bin_name][3]) - cutoff_locations[meas_key][asc_iter][1], # or from start of asc file to the end of the bin
                     cutoff_locations[meas_key][asc_iter][2] - (bins[bin_name][1] + bins[bin_name][0])]) # or from the start of the bin to the end of the ASC
        
        cheby_num = (cheby_max - cheby_min) / (width_at_max - width_at_min) * (width - width_at_min) + cheby_min
        cheby_num_int = 7 - int(max([min([cheby_num, cheby_max]), cheby_min]))
        cheby_floats_og = inp_all[index_meas_file + 6]
        inp_all[index_meas_file + 6] = cheby_floats_og[::-1].replace('0','1',cheby_num_int)[::-1] # change number of floated parameters in cheby            
        
        print('          {} - {} cm-1 wide ({} cheby coeffs)'.format(cutoff_locations[meas_key][asc_iter][0], np.round(width,1), 7-cheby_num_int))
        
        
    open(os.path.join(d_save,bin_name) + '.inp', 'w').writelines(inp_all)

    return which_ASC_files

#%%
def run_labfit(d_labfit, bin_name, use_rei=False, time_limit=120):
    r'''
    Overview: 
        run labfit, try to resolve common feature errors
        delete the lwa and check if a new one is generated to ensure that labfit really ran successfully
    Returns: 
        feature_error = feature labfit thinks is causing the error
    Inputs:
        
    r'''
    
    os.chdir(d_labfit) # change directory to labfit folder
    
    if use_rei: 
        result = os.system('copy '+bin_name+'.rei '+bin_name+'.inp') # copy file (success returns 0)
        if result != 0: print('\n\nrun_labfit wasnt able to run REI file (failed to copy)\n\n')
    
    outdat = open('OUTDAT.TXT', "r").readlines()
    
    outdat_start = outdat[14].find(':') 
    outdat_stop = outdat[14].find('.') 
    outdat[14] = outdat[14][:outdat_start+2] + bin_name + outdat[14][outdat_stop:] # change the name of the INP file you want to run
    
    open('OUTDAT.TXT', 'w').writelines(outdat)
    
    if os.path.isfile(bin_name+'.lwa'): os.remove(bin_name+'.lwa') # remove old file
    
    sp = subprocess.Popen(['labfit.exe'], stdin=subprocess.PIPE, shell=False, 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    poll_code = None # None = not done running, 0 = done running
    i = 0
    while poll_code == None: 
        i+=1
        sleep(1) # check every second
        poll_code = sp.poll() # is the file done running? (None = no)

        if time_limit == i: 
            
            sp.kill()
            print('   labfit timed out after {} seconds'.format(time_limit))
            
    dtl_all = open(bin_name+'.dtl', "r").readlines()
    
    feature_error = None # we'll check for a bad feature
    labfit_errors = [' Line   ',  # feedback that labfit might give in the DTL file if it didn't run
                     ' The problem MAY have to do with Line ', 
                     ' On the line after line number: ', 
                     ' This is forbidden by the constraint rules', 
                     ' Covariance matrix is not positive definite'] 

    

    if i >= time_limit: 
        feature_error = 'timeout'
                
    else: 

        for line in dtl_all:        
            if line.startswith(labfit_errors[0]): 
                print(line)
                feature_error = int(line.split()[1])
            if line.startswith(labfit_errors[1]):
                print(line)
                feature_error = int(line.split()[8])
            if line.startswith(labfit_errors[2]):
                print(line)
                inp_all = open(bin_name+'.inp', "r").readlines()
                feature_error_neighbor = int(line.split()[6]) # snag the feature listed
                for linei in range(len(inp_all)): 
                    if inp_all[linei].startswith(str(feature_error_neighbor).rjust(8)): # # when you find the right feature
                        feature_error = int(inp_all[linei+4].split()[0]) # grab the next feature
            if line.startswith(labfit_errors[3]):
                print(line)
                contraint = duplicated # you have a duplicated constraint. consider double checking who is constrained twice and which can be eliminated
            if line.startswith(labfit_errors[4]):
                print(line)
                labfit = notconverging # Labfit is upset. You'll need to change something. 
                # things that have worked in the past: 
                    # try floating lines one by one to see which errors out
                    # shortening an ASC file by one datapoint (usually one of the shorter ones, ~100 points)
        
    if os.path.isfile(bin_name+'.lwa') is False and feature_error == None: 
    
        feature_error = 'no LWA'
        
    os.chdir("..") # change directory back to where the python script lives

    return feature_error
#%%
def nself_initilize(d_load, base_name, update_name):
    r'''
    Overview: 
        change values for n_gamma_self as a function of quantum number J"
        could be updated to change any value as a function of another
        
        currently, this value is not in HITRAN but is set to 0.75 by labfit
    Returns: 
        None
    Inputs:
        
    r'''
    
    inp_all = open(os.path.join(d_load, base_name + '.inp'), "r").readlines()

    lines_until_features = lines_main_header + int(inp_all[0].split()[2]) * lines_per_asc # how many lines before the first line with features
    
    features_total = int(len(inp_all[lines_until_features:])/lines_per_feature)
    if int(inp_all[0].split()[3]) != features_total: please = stop # you missed something (both methods should both tell you how many features there are)
    
    # values taken from initial exploriation of the pure water data by scott 11/2022

    for i in range(features_total):
        
        if i < feature_new: 
            
            inp_index = lines_until_features+i*lines_per_feature # locate the feature
            
            # Jpp = int(inp_all[inp_index+3].replace('-',' -').split()[-3]) # snag the J" quantum number (should be the third to last value J", Ka", Kc")
            gamma_self = float(inp_all[inp_index+1].split()[0])
            
            if i < 10: 
                print('                                                       {}'.format(gamma_self))
                print(inp_all[inp_index+1])
            
            # nJpp = 0.886955 * np.exp(-0.045500 * Jpp)
            n_gamma = -0.20932 + 2.1923 * gamma_self
            
            if n_gamma < 0.2: n_gamma = 0.2
            elif n_gamma > 0.9: n_gamma = 0.9
            n = ('%.4f' % n_gamma).rjust(8) # give a little space between neighbor
            
            inp_all[inp_index+1] = inp_all[inp_index+1][0:12] + n + inp_all[inp_index+1][18:]  # replace the self width with the new value 

    # os.mkdir(d_save) # use this if you need to make a new folder for saving your files
    open(os.path.join(d_load,base_name) + update_name + '.inp', 'w').writelines(inp_all)
        
    return
#%%
def compile_results(d_labfit, base_name_input, base_name_output, bins, d_save = None): 
    '''
    Overview: 
        compile various inp files into a single inp file
    Returns: 
        
    Inputs:
        
    r'''

    d_done = os.path.join(d_labfit,'done')
    bins_done = os.listdir(d_done)
    df_load = None

    inp_all = open(os.path.join(d_labfit, base_name_input + '.inp'), "r").readlines()
    lines_until_features_inp = lines_main_header + int(inp_all[0].split()[2]) * lines_per_asc # all of the header down to the spectra
    features_total_og = int(inp_all[0].split()[3])
    features_total_updated = features_total_og # filler for now

    for bin_name in list(bins.keys())[::-1]: # move backwards to avoid disturbing earlier bins
        if bin_name in bins_done: # sorts them according to bin_names (lazy but effective)
            print(bin_name)
            if df_load is None: # first iteration

                d_load = os.path.join(d_done, bin_name, bin_name + '-000-og') # go find the base file (need a dtl -> df)
                df_load = db.labfit_to_df(d_load, htp=False) # open database

            wvn_range = bins[bin_name][1:3] # what wavenumber do we want?
            feature_start = df_load['nu'].sub(wvn_range[0]).abs().values.argmin()# what are the feature indexes of those wavenumbers? 
            feature_stop = df_load['nu'].sub(wvn_range[1]).abs().values.argmin()

            use_which = os.path.join(d_done, bin_name, os.listdir(os.path.join(d_done, bin_name))[-1])[0:-4] # snag the last file in the directory (TODO - air-water vs pure water)
            rei_bin = open(use_which + '.rei', "r").readlines()
            lines_until_features_bin = lines_main_header + int(rei_bin[0].split()[2]) * lines_per_asc # all of the header down to the spectra
            
            features_added = int(rei_bin[0].split()[3]) - features_total_og
            features_total_updated = features_total_updated + features_added
            
            # check that we're starting on the right feature
            line_feature_inp = lines_until_features_inp + lines_per_feature*(feature_start-1)
            line_feature_bin = lines_until_features_bin + lines_per_feature*(feature_start-1)
            if inp_all[line_feature_inp].split()[0] != rei_bin[line_feature_bin].split()[0]: 
                please = stophere # the first features aren't lining up (something is wrong)
            
            # check that we're ending on the right feature
            line_feature_inp = lines_until_features_inp + lines_per_feature*(feature_stop-1)
            line_feature_bin = lines_until_features_bin + lines_per_feature*(feature_stop-1) + lines_per_feature*features_added
            if inp_all[line_feature_inp].split()[0] != rei_bin[line_feature_bin].split()[0]: 
                please = stophere # the last features aren't lining up (something is wrong)
            
            for i in range(feature_start, feature_stop+1): # for all of the features in this bin (added features will be hidden in here and will throw off the count)
            
                line_feature_inp = lines_until_features_inp + lines_per_feature*(i-1)
                line_feature_bin = lines_until_features_bin + lines_per_feature*(i-1)
                
                inp_all[line_feature_inp] = rei_bin[line_feature_bin] # copy first line
                inp_all[line_feature_inp+1] = rei_bin[line_feature_bin+1][:-8] + '0.0\n' # copy second line, reset SD = 0
                inp_all[line_feature_inp+2] = '   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n' # change all floats to fixed
                inp_all[line_feature_inp+3] = rei_bin[line_feature_bin+3] # copy last line (quantum stuff)   
                
            for j in range(features_added): # add additional lines for each new feature (note that the actual new feature was probably already added, we're just cleaning things up)
                
                line_feature_inp = lines_until_features_inp + lines_per_feature*(i+j) # j indexed from 0 (removed -1), i left where it was
                line_feature_bin = lines_until_features_bin + lines_per_feature*(i+j)
                
                inp_all[line_feature_inp:line_feature_inp] = rei_bin[line_feature_bin:line_feature_bin+4] # add 4 new lines for all feature parameters
                inp_all[line_feature_inp+1] = rei_bin[line_feature_bin+1][:-8] + '0.0\n' # reset SD = 0
                inp_all[line_feature_inp+2] = '   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n' # change any floats to fixed

    inp_all[0] = inp_all[0][:33] + str(features_total_updated) + inp_all[0][38:]
    
    if d_save is None: d_save = d_labfit
    open(os.path.join(d_save,base_name_output) + '.inp', 'w').writelines(inp_all)        

    return

#%% feature sniffer (find which feature(s) is the problem)

def feature_sniffer(features_test, d_labfit, bin_name, bins, prop_which, props, props_which, 
                    prop_which2=False, prop_which3=False, iter_sniff=5, unc_multiplier=1, 
                    d_labfit_main=True, new_type=False, features_new=None): 
      
    if d_labfit_main is True: d_labfit_main = d_labfit
    
    features_safe_good = [] # no errors, good uncertainty
    features_safe_bad = [] # features that don't throw errors, but have bad uncertainties even all by themselves (safe_but_bad was too long)

    features_dangerous = []
        
    features_dict = {}
    
    for feature in features_test: # sniff out which feature(s) are causing you grief
        print('\nfeature {}, #{} out of {}, prop_which = {}, new_type = {}'.format(feature, features_test.index(feature)+1, len(features_test), prop_which, new_type))
        
        if new_type is not False: 
            add_features(d_labfit, bin_name, [feature], use_which='rei_saved', d_folder_input=d_labfit_main, new_type=new_type) 

        else: 
            # float lines, most recent saved REI in -> INP out
            float_lines(d_labfit, bin_name, [feature], props[prop_which], 'rei_saved', d_folder_input=d_labfit_main,
            features_new=features_new) 
            # INP -> INP, testing two at once (typically nu or n_self)
            if prop_which2 is not False: float_lines(d_labfit, bin_name, [feature], props[prop_which2], 'inp_new', [],
            features_new=features_new) 
            if prop_which3 is not False: float_lines(d_labfit, bin_name, [feature], props[prop_which3], 'inp_new', [],
            features_new=features_new)
        
        feature_error = None
        features_reject = [] # reset with each iteration

        try:  # be very careful when editing - any error will trigger the except (even if the feature is fine)
            
            feature_error = run_labfit(d_labfit, bin_name) # need to run one time to send INP info -> REI
            [df_props, _] = compare_dfs(d_labfit, bins, bin_name, props_which, props[prop_which], plots=False) # read results into python
            features_dict[feature] = df_props.copy()
            
            i = 1 # start at 1 because we already ran things once
            while i < iter_sniff: # run x number of times
                i += 1
                print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
                feature_error = run_labfit(d_labfit, bin_name, use_rei=True) 
                [df_props, _] = compare_dfs(d_labfit, bins, bin_name, props_which, props[prop_which], plots=False) # read results into python            
                features_dict[feature] = pd.concat([features_dict[feature], df_props])
                
                if feature_error is not None: 
                    throw = thaterrorplease
                        
                else: 
                    
                    if i > iter_sniff//2:  # start checking halfway through so we get a couple chances
                        # start checking uncertainty after the second iteration (it bounces around sometimes)                                        
                        for prop_i in props_which: 
                            if prop_i == 'sw': prop_i_df = 'sw_perc' # we want to look at the fractional change
                            else: prop_i_df = prop_i
                                                    
                            features_reject.extend(df_props[df_props['uc_'+prop_i_df] > props[prop_i][4] * unc_multiplier].index.values.tolist())
                            
                        try: features_reject.extend(df_props[df_props['gamma_self'] < 0.01].index.values.tolist()) # ditch features where gamma is going to 0
                        except: features_reject.extend(df_props[df_props['gamma_air'] < 0.01].index.values.tolist()) # ditch features where gamma is going to 0
        
            if new_type is not False: feature_id = new_feature_number(bin_name, new_type, 0) # assumes we're always looking at feature #1 (one-by-one)
            else: feature_id = feature
              
            if feature_id in features_reject and feature not in features_safe_bad:
                features_safe_bad.append(feature)
            
            else:             
                features_safe_good.append(feature)
                
        except:     

            feature_error = feature
            features_dangerous.append(feature)


                
                
    return features_safe_good, features_safe_bad, features_dangerous, features_dict

def wait_for_kernal(d_labfit, minutes=3): 

    time_since_update = time() - os.path.getmtime(d_labfit + r'/outdat.txt')
    print('checked')
    print('   {} seconds since outdat was updated'.format(int(time_since_update)))
    
    # average run time for these bins = 20 seconds, large allowance given for other calculations
    while time_since_update < 60*minutes: # wait for the kernal to go unused for 5 minutes
        
        sleep(60*minutes) # wait a minute and check again
        
        time_since_update = time() - os.path.getmtime(d_labfit + r'/outdat.txt')
        print('checked')
        print('   {} seconds since outdat was updated'.format(int(time_since_update))) # TODO - print the current time (hour is plenty and minute)
    
    return