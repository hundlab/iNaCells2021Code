#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:53:55 2020

@author: dgratz
"""

import pymc
import numpy as np

def make_model(run_biophysical, key_groups, datas, model_params_initial, mp_locs, model):
    ## create all sim means and variances per biophys model parameter
    
    # overall model parameter mean prior ~N
    mu = np.zeros_like(mp_locs)
    tau = 0.001* np.ones_like(mp_locs)
    model_bounds = np.array(model.param_bounds)
    model_param_mean = pymc.TruncatedNormal("model_param_mean",
                               mu = mu,
                               tau = tau, 
                               a = model_bounds[mp_locs,0],
                               b = model_bounds[mp_locs,1],
                               value = model_params_initial[mp_locs])
    
    # overall model parameter precision prior ~Gamma
    alpha = 0.001* np.ones_like(mp_locs)
    beta = 0.001* np.ones_like(mp_locs)
    value = 5* np.ones_like(mp_locs) #estimated from './optimize_Koval_0423_0326.pkl'
    model_param_tau = pymc.Gamma("model_param_tau", 
                              alpha = alpha,
                              beta = beta,
                              value = value)
    

    # model parameter  ~ N
    model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
    model_param_index = np.tile(model_param_index, (sum(map(len, key_groups)),1))
    model_param_sim =  pymc.TruncatedNormal("model_param",
                                     mu = model_param_mean[model_param_index],
                                     tau = model_param_tau[model_param_index],
                                     a = model_bounds[mp_locs,0][model_param_index],
                                     b = model_bounds[mp_locs,1][model_param_index])


    # precision for each protocol ~ Gamma
    alpha = 0.001* np.ones(len(key_groups))
    beta = 0.001* np.ones(len(key_groups))
    value = np.ones(len(key_groups))
    error_tau = pymc.Gamma("error_tau",
                              alpha = alpha,
                              beta = beta,
                              value = value)
    keys_list = []
    error_tau_index = []
    data_array = []
#    biophys_liks = []
#    biophys_results = []
    k = 0
    for i, keys in enumerate(key_groups):
        for j, key in enumerate(keys):
            exp_data = list(datas[key][:,1])
            data_array += exp_data
            
#            identif = "__{key[0]}__{key[1]}__gr{}".format(i, key=key)
        

#            model_param_sim.append(model_param)
#            biophys_liks.append(biophys_lik)
#            biophys_results.append(biophys_result)
            keys_list.append(key)
            error_tau_index += [i]*len(exp_data)
            k += 1
    error_tau_index = np.array(error_tau_index)
    data_array = np.array(data_array)

    # liklihood ~N
    biophys_result = pymc.Deterministic(eval=run_biophysical,
                               name='biophys_res',
                               parents={'model_parameters_list': model_param_sim,
                                        'keys': keys_list},
                               doc='run biophysical model',
                               dtype=np.ndarray,
                               cache_depth=15)

    biophys_lik = pymc.Normal('lik',
                                  mu=biophys_result,
                                  tau=error_tau[error_tau_index],
                                  value=data_array,
                                  observed=True)
    stat_model = [model_param_sim, model_param_mean, model_param_tau, biophys_lik, biophys_result, error_tau]
    
    return stat_model
