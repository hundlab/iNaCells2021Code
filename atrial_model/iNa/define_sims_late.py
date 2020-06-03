#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:12:17 2020

@author: grat05
"""

#from iNa_models import Koval_ina, OHaraRudy_INa
from iNa_models_ode import OHaraRudy_INa, Koval_ina
from scripts import load_data_parameters, all_data
from iNa_fit_functions import normalize2prepulse, setup_sim, \
calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp, biExp,\
calcExpTauAct, biExp_params, monoExp_params, integrateDur, medianValFromEnd,\
multipleProcess
from setup_sim_functions import setupSimExp, normalizeToBaseline, normalizeToFirst,\
    resort, minNorm_data, minNorm, minMaxNorm, func_norm, func_norm_data, minMaxNorm_data,\
    normalizeToFirst_data
from iNa_model_setup import model, mp_locs, sub_mps, sub_mp_bounds, dt, run_fits


import numpy as np
from scipy import optimize
from functools import partial
from multiprocessing import Pool
from scipy import integrate


#import sys
#sys.path.append('./models/build/Debug/')
#import models

np.seterr(all='ignore')

exp_parameters, data = load_data_parameters('/exp parameters/INaL.xlsx','Sheet1', data=all_data)


sim_fs = {}
datas = {}
keys_all = []

solver = partial(integrate.solve_ivp, method='BDF')


if run_fits['Late']:
   # peak current
    keys_iin = [('20488304_1', 'Dataset B SR'), ('20488304_2', 'Dataset B Vehicle')]

    keys_all.append(keys_iin)

    setupSimExp(sim_fs=sim_fs,
                datas=datas,
                data=data,
                exp_parameters=exp_parameters,
                keys_iin=keys_iin,
                model=model,
                process=partial(peakCurr, durn=-2),
                dt=dt,
#                post_process=minMaxNorm,
#                process_data=minMaxNorm_data,
                setup_sim_args={'sim_args':{
                                'solver': solver},
                                'num_repeats':5})
        
    # iv curve
    keys_iin = [('20488304_2', 'Dataset D Vehicle')]
    
    keys_all.append(keys_iin)
    setupSimExp(sim_fs=sim_fs,\
            datas=datas,\
            data=data,\
            exp_parameters=exp_parameters,\
            keys_iin=keys_iin,\
            model=model,\
            process=peakCurr,\
            dt=dt,\
            process_data=minNorm_data,#partial(minMaxNorm_data, feature_range=(-1, 0)),\
            post_process=minNorm,#partial(minMaxNorm, feature_range=(-1, 0)),
            setup_sim_args={'sim_args':{'solver': solver}, 'hold_dur':500})
        
    # curve integral
    keys_iin = [('20488304_1', 'Dataset F SR'), ('20488304_3', 'Dataset B Vehicle')]
    
    setupSimExp(sim_fs=sim_fs,
                datas=datas,
                data=data,
                exp_parameters=exp_parameters,
                keys_iin=keys_iin,
                model=model,
                process=partial(integrateDur, dur_loc=-2, begin_offset=50),
                dt=dt,
#                post_process=normalizeToFirst,
#                process_data=normalizeToFirst_data,
                setup_sim_args={'sim_args':{
                                'solver': solver},
                                'num_repeats':5})
        
    # late current
    keys_iin = [('26121051_2', 'Dataset B SR Control BT'), ('26121051_2', 'Dataset B SR Control RT')]
    
    processes = [partial(medianValFromEnd, dur_loc=1, window=5),
                 partial(medianValFromEnd, dur_loc=0, window=5),
                 partial(medianValFromEnd, dur_loc=-1, window=5)]
    
    setupSimExp(sim_fs=sim_fs,
            datas=datas,
            data=data,
            exp_parameters=exp_parameters,
            keys_iin=keys_iin,
            model=model,
            process=partial(multipleProcess, processes=processes),
            dt=dt,
#            post_process=normalizeToFirst,
#            process_data=normalizeToFirst_data,
            setup_sim_args={'sim_args':{
                            'solver': solver},
                            'num_repeats':5,
                            'hold_dur':500,
                            'data_len':1})
        
