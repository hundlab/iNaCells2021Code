#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:46:20 2020

@author: grat05
"""

from iNa_models import Koval_ina
from scripts import load_data_parameters, load_all_data
from iNa_fit_functions import setup_sim
from iNa_defines import all_data
import numpy as np
from scipy.optimize import minimize


def fit_one_G(model, data, exp_parameters, tol=5, maxiter=5):
    all_model_parameters = {}
    for idx in exp_parameters.index:
        one_data = data[idx]
        one_exp_parameters = exp_parameters.loc[idx]
        curr_real = calc_currs_real(one_data, one_exp_parameters)
        model_parameters = np.ones(22)
        curr_pred = calc_currs(model_parameters, model, one_data, one_exp_parameters, 0.05)
        func = lambda gNaFactor: np.sum((gNaFactor[0]*curr_pred-curr_real)**2)
        res = minimize(func, [1],method='SLSQP')
        if res['success']:
            model_parameters[len(model_parameters)-1] = res['x'][0]
        else:
            print('error on ',idx)
        plt.figure()
        plt.scatter(one_data[:,0], curr_real)
        plt.plot(one_data[:,0], curr_pred*res['x'][0])
        all_model_parameters[idx] = model_parameters
    return all_model_parameters

def fit_one_minimise(model, data, exp_parameters, model_parameters):
    all_model_parameters = {}
    for idx in exp_parameters.index:
        print('new exp')
        idx = ('12890054_5', 'Dataset D Control')
        one_data = data[idx]
        one_exp_parameters = exp_parameters.loc[idx]
        one_model_parameters = model_parameters[idx]
        func = lambda model_parameters: \
            calc_diff(model_parameters, model, one_data, one_exp_parameters, 0.05)
        res = minimize(func, one_model_parameters,method='SLSQP', bounds=[(0,None)]*len(one_model_parameters))
        all_model_parameters[idx] = res['x']
        print('minimized')
        break
    return all_model_parameters


exp_parameters, data = load_data_parameters('iNa_dims.xlsx','iv_curve', data=all_data)
idx = ('27694909_8', 'Dataset B Control')
one_data = data[idx]
one_exp_parameters = exp_parameters.loc[idx]
one_model_parameters = np.ones(22)#all_model_parameters[idx]
model = Koval_ina
#all_model_parameters = fit_one_G(model, data, exp_parameters)
print(calc_diff(one_model_parameters, model, one_data, one_exp_parameters, dt=0.05))
#all_model_parameters = fit_one_minimise(Koval_ina, data, exp_parameters, all_model_parameters)
