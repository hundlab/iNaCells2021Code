#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:10:27 2020

@author: grat05
"""

import pymc3 as pm
import numpy as np
import pandas as pd
import theano
import theano.tensor as tt
from contextlib import contextmanager

def key_frame(key_groups, exp_parameters):
    keys_list = [key for keys in key_groups for key in keys]
    return exp_parameters.loc[keys_list, ['Sim Group', 'temp ( K )']]


class BiophysicalModel(tt.Op):
    itypes = [tt.dmatrix]
    otypes = [tt.dvector]
    
    def __init__(self, keys_list, run_biophysical):
        self.keys_list = keys_list
        self.run_biophysical = run_biophysical
        super(BiophysicalModel, self).__init__()
    
    def perform(self, node, inputs, outputs):
        model_param_sim, = inputs
        outputs[0][0] = self.run_biophysical(model_param_sim, self.keys_list)

@contextmanager
def StatModel(run_biophysical, key_frame, datas, mp_locs, model):
    ## create all sim means and variances per biophys model parameter
    with pm.Model() as model:
        num_sims = len(key_frame)
    
        
        # overall model parameter mean prior ~N
        mu = 0.
        sigma = 100.
#        model_bounds = np.array(model.param_bounds)
        model_param_intercept = pm.Normal("model_param_intercept",
                                   mu = mu,
                                   sigma = sigma,
                                   shape = mp_locs.shape)
        
        # overall model parameter precision prior ~Gamma
        alpha = 0.001
        beta = 0.001
#        value = 5* np.ones_like(mp_locs) #estimated from './optimize_Koval_0423_0326.pkl'
        model_param_sigma = pm.Gamma("model_param_sigma", 
                                  alpha = alpha,
                                  beta = beta,
                                  shape = mp_locs.shape)
        
    

        # temerature beta
        #temperature adjusted to minimum in group
        temperature_arr = np.array(key_frame['temp ( K )'], dtype=float) -290.15

        #temperature coefficiant ~ N
        mu = 0# update to fit q10 (0.2)
        sigma = 100 # .1
        b_temp = pm.Normal("b_temp",
                                   mu = mu,
                                   sigma = sigma,
                                   shape = mp_locs.shape)
        
        #linear mean
        model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
        model_param_index = np.tile(model_param_index, (num_sims,1))
        mu = model_param_intercept[model_param_index] + b_temp[model_param_index]*temperature_arr[...,None]

        # model parameter  ~ N
        model_param_sim =  pm.Normal("model_param",
                                         mu = mu,
                                         sigma = model_param_sigma[model_param_index],
                                         shape = model_param_index.shape)

    
    
        # precision for each protocol ~ Gamma
        key_groups = list(key_frame['Sim Group'].unique())
        alpha = 0.001
        beta = 0.001
        error_sigma = pm.Gamma("error_sigma",
                                  alpha = alpha,
                                  beta = beta,
                                  shape = len(key_groups))
        error_sigma_index = []
        data_array = []
        for key in key_frame.index:
            exp_data = list(datas[key][:,1])
            data_array += exp_data

            group_num = key_groups.index(key_frame.loc[key, 'Sim Group'])
            error_sigma_index += [group_num]*len(exp_data)

        error_sigma_index = np.array(error_sigma_index)
        data_array = np.array(data_array)
                  
        tt_biophysical_model = BiophysicalModel(list(key_frame.index), run_biophysical)
    
        # liklihood ~N
        biophys_result = pm.Deterministic('biophys_res', 
                                          tt_biophysical_model(model_param_sim))
    
        biophys_lik = pm.Normal('lik',
                                      mu=biophys_result,
                                      sigma=error_sigma[error_sigma_index],
                                      observed=data_array)
        yield model
