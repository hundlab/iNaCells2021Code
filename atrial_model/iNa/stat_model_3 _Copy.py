#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:10:27 2020

@author: grat05
"""

import pymc3 as pm
import numpy as np
import pandas as pd
import inspect
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
def StatModel(run_biophysical, key_frame, datas, mp_locs, biophys_model):
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
        model_param_tau = pm.Gamma("model_param_tau", 
                                  alpha = alpha,
                                  beta = beta,
                                  shape = mp_locs.shape)
        
    

        # temerature beta
        #temperature adjusted to room temperature
        #temperature_arr = np.array(key_frame['temp ( K )'], dtype=float) -290.15
        # temperatures = pm.Normal('temperatures',
        #                          mu = temperature_arr,
        #                          sigma = 1,
        #                          shape = temperature_arr.shape)

        #temperature coefficiant ~ N
        mu = 0# update to fit q10 (0.2)
        sigma = 100 # .1
        b_temp = pm.Normal("b_temp",
                                   mu = mu,
                                   sigma = sigma,
                                   shape = mp_locs.shape)
        
        # #linear mean
        # model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
        # model_param_index = np.tile(model_param_index, (num_sims,1))
        # mu = model_param_intercept[model_param_index] + b_temp[model_param_index]*temperature_arr[...,None]

        # # model parameter  ~ N
        # model_param_sim =  pm.Normal("model_param",
        #                                  mu = mu,
        #                                  tau = model_param_tau[model_param_index],
        #                                  shape = model_param_index.shape)



        key_groups = list(key_frame['Sim Group'].unique())

        model_params_sim = []
        model_params_vars = []
        model_param_names = np.array(inspect.getfullargspec(biophys_model.__init__).args)[1:][mp_locs]
        
        for grp_idx, sim_group in enumerate(key_frame['Sim Group'].unique()):
            grp_len = (key_frame['Sim Group'] == sim_group).sum()
            grp_keys = list((key_frame[key_frame['Sim Group'] == sim_group]).index)
            
            grp_data = []
            mp_grp = []
            for i, mp_name in enumerate(model_param_names):
                temperature = key_frame.loc[grp_keys,'temp ( K )']
                mu = model_param_intercept[i] + b_temp[i]*temperature
                mp_grp.append(pm.Normal("model_param ("+mp_name+") ("+sim_group+")",
                                 mu = mu,
                                 tau = model_param_tau[i],
                                 shape = (grp_len,)))
            model_params_vars += mp_grp
            model_params_sim.append(tt.stack(mp_grp, axis=1))

#                grp_data += list(datas[grp_keys[i]][:,1])
#            grp_data = np.array(grp_data)
            
            # tt_biophysical_model = BiophysicalModel(grp_keys, run_biophysical)
            # biophys_result = pm.Deterministic("biophys_res ("+sim_group+")",
            #                               tt_biophysical_model(mp_grp))
        model_param_sim = tt.concatenate(model_params_sim, axis=0)
        step1 = pm.Metropolis(vars=model_params_vars)
        
        # precision for each protocol ~ Gamma      
        alpha = 0.001
        beta = 0.001
        error_tau = pm.Gamma("error_tau",
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
                                      tau=error_tau[error_sigma_index],
                                      observed=data_array)
        yield model, step1
