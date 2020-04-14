#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:12:23 2020

@author: dgratz
"""

from iNa_fit_functions import setup_sim, peakCurr

import numpy as np
from functools import partial
import copy
from sklearn.preprocessing import minmax_scale


def setupSimExp(sim_fs, datas, data, exp_parameters, keys_iin, model, process,\
                dt, post_process=None, process_data=None, setup_sim_args={}):
    for key in keys_iin:
        key_data = data[key]
        if not process_data is None:
            key_data = process_data(key_data)
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, process, dt=dt, **setup_sim_args)
        if not post_process is None:
            sim_fw = partial(post_process, sim_f=sim_f)
        else:
            sim_fw = sim_f

        sim_fs[key] = sim_fw
        datas[key] = key_data

def resort(model_parameters, sim_f):
    vals = sim_f(model_parameters)
    return vals.flatten('F')
  

def normalizeToBaseline(model_params, sim_f):
    sim_f_baseline = copy.deepcopy(sim_f)
    sim_f_baseline.keywords['durs'] = [1,10]
    sim_f_baseline.keywords['voltages'] = [\
                           sim_f_baseline.keywords['voltages'][0], \
                           sim_f_baseline.keywords['voltages'][3]]
    sim_f_baseline.keywords['process'] = peakCurr

    baseline = sim_f_baseline(model_params)
    process = sim_f.keywords['process']
    new_process = partial(process, val=baseline[0])
    sim_f.keywords['process'] = new_process

    return sim_f(model_params)

def normalizeToFirst(model_params, sim_f):
    out = sim_f(model_params)
    return out/out[0]

def minMaxNorm(model_params, sim_f, feature_range=(0, 1)):
    return minmax_scale(sim_f(model_params))

def minMaxNorm_data(data, feature_range=(0, 1)):
    data = np.copy(data)
    minmax_scale(data[:,1], copy=False)
    return data

def minNorm(model_params, sim_f):
    sim_res = sim_f(model_params)
    return sim_res/np.abs(np.min(sim_res))

def minNorm_data(data):
    data = np.copy(data)
    data[:,1] = data[:,1]/np.abs(np.min(data[:,1]))
    return data

