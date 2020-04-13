#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:53:55 2020

@author: dgratz
"""

import pymc
import numpy as np

from iNa_fit_functions import SimResults

def make_model(calc_fn, key_groups, datas, model_params_initial, model):
    # biophysical model parameters
    model_params = []
    
    for i in range(len(model_params_initial)):
        # model parameter mean prior ~N
        model_param_mean = pymc.Normal("model_param_mean_{}".format(i),
                                       mu = model_params_initial[i],
                                       tau = 1/10**2,
                                       value = model_params_initial[i])
    
        # model parameter variance prior ~IG
        model_param_tau = pymc.Gamma("model_param_var_{}".format(i), 
                                                  alpha=0.001, beta=0.001,
                                                  value = 1)
        
        # model parameter  ~ N
        model_param = pymc.Normal("model_param_{}".format(i),
                                         mu=model_param_mean, tau=model_param_tau)
        model_params.append(model_param)
        
    model_params = np.array(model_params)
    
    # likelihood
    run_biophysical = SimResults(calc_fn)
    biophysical_liks = []
    for i, keys in enumerate(key_groups):
        data = []
        for key in keys:
            data += list(datas[key])
        
        # error for each protocol ~ IG
        error_tau = pymc.Gamma("error_var_{}".format(i),
                                      alpha=0.001, beta=0.001,
                                      value = 1)
        
        # liklihood ~N
        biophysical_results = pymc.Deterministic(eval=SimResults.__call__,
                                   name='biophysical_results_{}'.format(i),
                                   parents={'self': run_biophysical,
                                            'model_parameters': model_params,
                                            'keys': frozenset(keys)},
                                   doc='run biophysical model')

        biophysical_lik = pymc.Normal('biophysical_lik',
                                      mu=biophysical_results,
                                      tau=error_tau,
                                      data=np.array(data),
                                      observed=True)
        biophysical_liks.append(biophysical_lik)
    return locals()
