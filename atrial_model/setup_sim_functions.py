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
from scipy import stats
import matplotlib.pyplot as plt

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
            key_data = process_data(key_data, key=key)
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, **setup_sim_args)

        sim_fs[key] = sim_f
        datas[key] = key_data


def iNaCurve(t, tau_m, tau_f, tau_s, fs_prop):
    A1 = fs_prop
    A2 = 1 - fs_prop
    res = np.power(1-np.exp(-t/tau_m),3) * (-A1*np.exp(-t/tau_f) -A2*np.exp(-t/tau_s))
    res = minmax_scale(res, feature_range=(-1,0))
    return res

def inaTausFromData(datas, tau_keys, fs_prop=0.9):
    tau_m_key, tau_f_key, *rest_key = tau_keys
    taus = []

    tau_m_data = datas[tau_m_key]
    sorted_locs = np.argsort(tau_m_data[:,0])
    taus.append(tau_m_data[sorted_locs,0])
    taus.append(tau_m_data[sorted_locs,1])

    tau_f_data = datas[tau_f_key]
    sorted_locs = np.argsort(tau_f_data[:,0])
    taus.append(tau_f_data[sorted_locs,1])

    if len(rest_key) >= 1:
        tau_s_key = rest_key[0]
        tau_s_data = datas[tau_s_key]
        sorted_locs = np.argsort(tau_s_data[:,0])
        taus.append(tau_s_data[sorted_locs,1])
    else:
        taus.append(tau_f_data[sorted_locs,1])

    if len(rest_key) >= 2:
        fs_prop_key = rest_key[1]
        fs_prop_data = datas[fs_prop_key]
        sorted_locs = np.argsort(fs_prop_data[:,0])
        taus.append(fs_prop_data[sorted_locs,1])
    else:
        taus.append([fs_prop]*len(sorted_locs))
        fs_prop_key = None

    return np.array(taus).T
    

def inaCurvesFromData(times, data, tau_keys, fs_prop=0.9):
    taus = inaTausFromData(data, tau_keys, fs_prop=fs_prop)
    ina_curves = []
    for i in range(taus.shape[0]):
        ina_curves.append(iNaCurve(times, *taus[i,1:]))
    ina_curves = np.array(ina_curves)
    return ina_curves
    

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

def normalizeToFirst_data(data, **kwargs):
    data = np.copy(data)
    data[:,1] = data[:,1]/data[0,1]
    return data

def normalizeToMax(vals, **kwargs):
    return np.abs(vals)/np.abs(vals).max()

def normalizeToMax_data(data, **kwargs):
    data = np.copy(data)
    data[:,1] = data[:,1]/data[:,1].max()
    return data

def minMaxNorm(vals, feature_range=(0, 1), **kwargs):
    return minmax_scale(vals, feature_range=feature_range)

def minMaxNorm_data(data, feature_range=(0, 1), **kwargs):
    data = np.copy(data)
    minmax_scale(data[:,1], copy=False)
    return data

def func_norm(vals, func, **kwargs):
    return func(vals)

def func_norm_data(data, func, **kwargs):
    data = np.copy(data)
    data[:,1] = func(data[:,1])
    return data

def minNorm(vals, **kwargs):
    normed = vals/np.abs(np.min(vals))
    normed = np.sign(normed)*np.sqrt(np.abs(normed))
#    normed = np.cbrt(normed)
    return normed

def flattenResult(vals, **kwargs):
    return vals.flatten()


def minNorm_data(data, **kwargs):
    data = np.copy(data)
    normed = data[:,1]
    normed = normed/np.abs(np.min(normed))
    normed = np.sign(normed)*np.sqrt(np.abs(normed))
#    normed = np.cbrt(normed)
    data[:,1] = normed
    return data

def signsqrt(vals, **kwargs):
    return np.sign(vals)*np.sqrt(np.abs(vals))

def signsqrt_data(data, **kwargs):
    data = np.copy(data)
    normed = data[:,1]
    normed = np.sign(normed)*np.sqrt(np.abs(normed))
    data[:, 1] = normed
    return data

def correctShift_data(data, key, exp_parameters, model, **kwargs):
    data = np.copy(data)
    peak_loc = np.argmin(data[:,1])
    res = stats.linregress(data[peak_loc+1:])
    act_ena = res.intercept
    model_inst = model(naO=exp_parameters.loc[key, '[Na]o (mM)'],
                                naI=exp_parameters.loc[key, '[Na]I (mM)'],
                                TEMP=exp_parameters.loc[key, 'temp ( K )'])
    thr_ena = model_inst.getRevPot()
    
    # plt.figure(str(key))
    # plt.scatter(data[:,0], data[:,1])
    # plt.plot(data[peak_loc+1:,0], data[peak_loc+1:,0]*res.slope+res.intercept)
    # plt.plot(data[peak_loc+1:,0], data[peak_loc+1:,0]*res.slope-thr_ena)
    # print(-thr_ena - act_ena)
    
    shift = -thr_ena - act_ena
    data[:,0] -= shift
    
    # plt.scatter(data[:,0], data[:,1])

    return data

def chainProcess_data(data, new_process, prev_process, **kwargs):
    data = prev_process(data, **kwargs)
    data = new_process(data, **kwargs)
    return data

def addNoise_data(data, sdFact=2e5):
    sd = data.shape[0] / sdFact
    #print(sd, data.shape[0])
    data = np.copy(data)
    data[:,1] = np.random.normal(loc=data[:,1], scale=sd)
    return data