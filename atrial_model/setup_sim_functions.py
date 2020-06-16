#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:12:23 2020

@author: dgratz
"""

from .setup_sim import setup_sim
from .run_sims import SimRunner
from .run_sims_functions import peakCurr, normalized2val

import numpy as np
from functools import partial, wraps
import copy
from sklearn.preprocessing import minmax_scale
import types


def setupSimExp(sim_fs, datas, data, exp_parameters, keys_iin, model, process,\
                dt, post_process=None, process_data=None, setup_sim_args={}):
    sim_args = {}
    if 'sim_args' in setup_sim_args:
        sim_args = setup_sim_args['sim_args']
    sim_args['dt'] = dt
    sim_args['process'] = process
    sim_args['post_process'] = post_process
    setup_sim_args['sim_args'] = sim_args
    
    for key in keys_iin:
        key_data = data[key]
        if not process_data is None:
            key_data = process_data(key_data)
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, **setup_sim_args)

        sim_fs[key] = sim_f
        datas[key] = key_data


def normalizeToBaseline(sim_f, baseline_locs=[0,3]):
    run_sim_base = sim_f.run_sim
    
    sim_f_baseline = copy.deepcopy(sim_f)
    durs = sim_f_baseline.durs[np.newaxis, 0, baseline_locs]
    voltages = sim_f_baseline.voltages[np.newaxis, 0, baseline_locs]
    sim_f_baseline.durs = durs
    sim_f_baseline.voltages = voltages
    sim_f_baseline.process = peakCurr
    
    @wraps(sim_f.run_sim)
    def run_sim(self, model_parameters, pool=None):
        try:
            sim_f_baseline.run_sim(model_parameters, pool=pool)
            baseline = sim_f_baseline.get_output()
#            print(baseline)
            process = partial(normalized2val, durn=3, val=baseline)
            self.process = process
            run_sim_base(model_parameters, pool=pool)
        except Exception as e: 
            self.exception = e
    sim_f.run_sim = types.MethodType(run_sim, sim_f)
    return sim_f

def resort(vals, **kwargs):
    return vals.flatten('F')

def normalizeToFirst(vals, **kwargs):
    return vals/vals[0]

def normalizeToFirst_data(data):
    data = np.copy(data)
    data[:,1] = data[:,1]/data[0,1]
    return data

def minMaxNorm(vals, feature_range=(0, 1), **kwargs):
    return minmax_scale(vals, feature_range=feature_range)

def minMaxNorm_data(data, feature_range=(0, 1)):
    data = np.copy(data)
    minmax_scale(data[:,1], copy=False)
    return data

def func_norm(vals, func, **kwargs):
    return func(vals)

def func_norm_data(data, func):
    data = np.copy(data)
    data[:,1] = func(data[:,1])
    return data

def minNorm(vals, **kwargs):
    normed = vals/np.abs(np.min(vals))
    normed = np.sign(normed)*np.sqrt(np.abs(normed))
#    normed = np.cbrt(normed)
    return normed

def minNorm_data(data):
    data = np.copy(data)
    normed = data[:,1]
    normed = normed/np.abs(np.min(normed))
    normed = np.sign(normed)*np.sqrt(np.abs(normed))
#    normed = np.cbrt(normed)
    data[:,1] = normed
    return data

