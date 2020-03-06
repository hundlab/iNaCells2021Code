#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:06:36 2020

@author: grat05
"""

from iNa_models import Koval_ina, OHaraRudy_INa
from scripts import load_data_parameters, load_all_data, all_data
import iNa_fit_functions
from iNa_fit_functions import normalize2prepulse, setup_sim, run_sim, \
calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp, biExp,\
calcExpTauAct, triExp


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial
import copy
from multiprocessing import Pool


iNa_fit_functions.plot1 = False #sim
iNa_fit_functions.plot2 = False #diff
iNa_fit_functions.plot3 = False #tau

model = OHaraRudy_INa#Koval_ina##
dt = 0.05
monoExp_params = [-1,1,0]
biExp_params = np.array([-1,1,-1,1000,0])
triExp_params = np.array([-1,10,-1,100,-1,10000,0])


exp_parameters, data = load_data_parameters('iNa_dims.xlsx','gating', data=all_data)

def getTAndCurr(times, current, **kwargs):
    return times, current

def getCurr(times, current, **kwargs):
    return current

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


sim_fs = []
datas = []

model_params = np.ones(model.num_params)#res.x#
mp_locs = np.arange(model.num_params)#[2,3,13]#4
sub_mps = model_params[mp_locs]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# I2/I1 Recovery
keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
            ('1323431_8',	'Dataset A -100'),\
            ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
            ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
            ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')]
#keys_all.add(keys_iin)

setupSimExp(sim_fs=sim_fs,\
            datas=datas,\
            data=data,\
            exp_parameters=exp_parameters,\
            keys_iin=keys_iin,\
            model=model,\
            process=normalize2prepulse,\
            dt=dt,\
            post_process=None)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                  mp_locs=mp_locs, sim_func=sim_fs, data=datas,l=0)

#res = optimize.least_squares(diff_fn, sub_mps, \
#                             bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
iNa_fit_functions.plot1 = False #sim
iNa_fit_functions.plot2 = True #diff
iNa_fit_functions.plot3 = False #tau
diff_fn(model_params)
plt.show()
