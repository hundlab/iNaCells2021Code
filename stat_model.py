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
    

    # precision for each protocol ~ Gamma
    alpha = 0.001* np.ones(len(key_groups))
    beta = 0.001* np.ones(len(key_groups))
    value = np.ones(len(key_groups))
    error_tau = pymc.Gamma("error_tau",
                              alpha = alpha,
                              beta = beta,
                              value = value)

    model_param_sim = []    
    biophys_liks = []
    biophys_results = []
    for i, keys in enumerate(key_groups):
        for j, key in enumerate(keys):
            data = np.array(datas[key][:,1])
            
            identif = "__{key[0]}__{key[1]}__gr{}".format(i, key=key)
            
            # model parameter  ~ N
            model_param = pymc.TruncatedNormal("model_param"+identif,
                                             mu = model_param_mean,
                                             tau = model_param_tau,
                                             a = model_bounds[mp_locs,0],
                                             b = model_bounds[mp_locs,1])
        
            # liklihood ~N
            biophys_result = pymc.Deterministic(eval=run_biophysical,
                                       name='biophys_res'+identif,
                                       parents={'model_parameters': model_param,
                                                'keys': frozenset([key])},
                                       doc='run biophysical model',
                                       cache_depth=15)
    
            biophys_lik = pymc.Normal('lik'+identif,
                                          mu=biophys_result,
                                          tau=error_tau[i],
                                          value=data,
                                          observed=True)
            model_param_sim.append(model_param)
            biophys_liks.append(biophys_lik)
            biophys_results.append(biophys_result)

    stat_model = [model_param_sim, model_param_mean, model_param_tau, biophys_liks, biophys_results, error_tau]
    
    return stat_model
