#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:10:27 2020

@author: grat05
"""

import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
from contextlib import contextmanager



@contextmanager
def StatModel(run_biophysical, key_groups, datas, mp_locs, model, temperatures):
    ## create all sim means and variances per biophys model parameter
    with pm.Model() as model:
        num_sims = sum(map(len, key_groups))
    
        
        # overall model parameter mean prior ~N
        mu = np.zeros_like(mp_locs, dtype=float)
        sigma = 100* np.ones_like(mp_locs, dtype=float)
#        model_bounds = np.array(model.param_bounds)
        model_param_intercept = pm.Normal("model_param_intercept",
                                   mu = mu,
                                   sigma = sigma,
                                   shape = mu.shape)
        
        # overall model parameter precision prior ~Gamma
        alpha = 0.001* np.ones_like(mp_locs)
        beta = 0.001* np.ones_like(mp_locs)
#        value = 5* np.ones_like(mp_locs) #estimated from './optimize_Koval_0423_0326.pkl'
        model_param_sigma = pm.Gamma("model_param_sigma", 
                                  alpha = alpha,
                                  beta = beta,
                                  shape = alpha.shape)
        
    

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
        sigma = 100* np.ones_like(mp_locs) # .1
        b_temp = pm.Normal("b_temp",
                                   mu = mu,
                                   sigma = sigma,
                                   shape = mu.shape)
        
        #linear mean
        model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
        model_param_index = np.tile(model_param_index, (num_sims,1))
        mu = model_param_intercept[model_param_index] + b_temp[model_param_index]*temperature_arr[...,None]

        # model parameter  ~ N
        model_param_sim =  pm.Normal("model_param",
                                         mu = mu,
                                         tau = model_param_sigma[model_param_index],
                                         shape = model_param_index.shape)

    
    
        # precision for each protocol ~ Gamma
        alpha = 0.001* np.ones(len(key_groups))
        beta = 0.001* np.ones(len(key_groups))
        error_sigma = pm.Gamma("error_sigma",
                                  alpha = alpha,
                                  beta = beta,
                                  shape = alpha.shape)
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
#                data_array.append(datas[key][:,1])
                
    #            identif = "__{key[0]}__{key[1]}__gr{}".format(i, key=key)
            
    
    #            model_param_sim.append(model_param)
    #            biophys_liks.append(biophys_lik)
    #            biophys_results.append(biophys_result)
                keys_list.append(key)
                error_tau_index += [i]*len(exp_data)
#                error_tau_index.append(i)
                k += 1
        error_tau_index = np.array(error_tau_index)
        data_array = np.array(data_array)
    
        class BiophysicalModel(tt.Op):
            itypes = [tt.dmatrix]
            otypes = [tt.dvector]
            
            def perform(self, node, inputs, outputs):
                model_param_sim, = inputs
                outputs[0][0] = run_biophysical(model_param_sim, keys_list)
                
        tt_biophysical_model = BiophysicalModel()
    
        # liklihood ~N
        biophys_result = pm.Deterministic('biophys_res', 
                                          tt_biophysical_model(model_param_sim))
    
        biophys_lik = pm.Normal('lik',
                                      mu=biophys_result,
                                      tau=error_sigma[error_tau_index],
                                      observed=data_array)
        yield model
