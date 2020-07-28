#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:53:55 2020

@author: dgratz
"""

import pymc
import numpy as np

def make_model(run_biophysical, key_groups, datas, model_params_initial, mp_locs, model, temperatures = None):
    ## create all sim means and variances per biophys model parameter
    
    num_sims = sum(map(len, key_groups))

    
    # overall model parameter mean prior ~N
    mu = np.zeros_like(mp_locs)
    tau = 0.001* np.ones_like(mp_locs)
    model_bounds = np.array(model.param_bounds)
    model_param_mean = pymc.Normal("model_param_mean",
                               mu = mu,
                               tau = tau, 
#                               a = model_bounds[mp_locs,0],
#                               b = model_bounds[mp_locs,1],
                               value = model_params_initial[mp_locs])
    
    # overall model parameter precision prior ~Gamma
    alpha = 0.001* np.ones_like(mp_locs)
    beta = 0.001* np.ones_like(mp_locs)
    value = 5* np.ones_like(mp_locs) #estimated from './optimize_Koval_0423_0326.pkl'
    model_param_tau = pymc.Gamma("model_param_tau", 
                              alpha = alpha,
                              beta = beta,
                              value = value)
    

    if not temperatures is None:
        # temerature beta
        temperature_arr = np.empty(num_sims)
        k = 0
        for i, keys in enumerate(key_groups):
            for j, key in enumerate(keys):
                #temperature adjusted to minimum in group
                temperature_arr[k] = temperatures[key] -290.15
                k += 1
        
        #temperature coefficiant ~ N
        mu = np.zeros_like(mp_locs)# update to fit q10 (0.2)
        tau = 0.001* np.ones_like(mp_locs) # .1
        b_temp = pymc.Normal("b_temp",
                                   mu = mu,
                                   tau = tau, 
                                   value = mu)
        
        #linear mean
        model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
        model_param_index = np.tile(model_param_index, (num_sims,1))
        temperature_arr = np.broadcast_to(temperature_arr[...,None], shape=model_param_index.shape)
        #temperature_arr[:,0] = 0
        mu = model_param_mean[model_param_index] + b_temp[model_param_index]*temperature_arr

        # model parameter  ~ N
        model_param_sim =  pymc.Normal("model_param",
                                         mu = mu,
                                         tau = model_param_tau[model_param_index]
#                                         a = model_bounds[mp_locs,0][model_param_index],
#                                         b = model_bounds[mp_locs,1][model_param_index]
                                         )
    else:
        # model parameter  ~ N
        model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
        model_param_index = np.tile(model_param_index, (num_sims,1))
        model_param_sim =  pymc.Normal("model_param",
                                         mu = model_param_mean[model_param_index],
                                         tau = model_param_tau[model_param_index]
#                                         a = model_bounds[mp_locs,0][model_param_index],
#                                         b = model_bounds[mp_locs,1][model_param_index]
                                         )        


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
    stat_model = [model_param_sim, model_param_mean, model_param_tau, 
                  biophys_lik, biophys_result, error_tau]
    if not temperatures is None:
        stat_model.append(b_temp)

    return stat_model
