#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:09:01 2020

@author: dgratz
"""

#from iNa_models import Koval_ina, OHaraRudy_INa
from iNa_models_ode import OHaraRudy_INa, Koval_ina
from data_loader import load_data_parameters, all_data
from iNa_fit_functions import normalize2prepulse, setup_sim, \
calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp, biExp,\
calcExpTauAct, biExp_params, monoExp_params
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

exp_parameters_gating, data_gating = load_data_parameters('/params old/INa_old.xlsx','gating', data=all_data)
exp_parameters_iv, data_iv = load_data_parameters('/params old/INa_old.xlsx','iv_curve', data=all_data)

exp_parameters = exp_parameters_gating.append(exp_parameters_iv, sort=False)
data_gating.update(data_iv)
data = data_gating


sim_fs = {}
datas = {}
keys_all = []

solver = partial(integrate.solve_ivp, method='BDF')


if run_fits['Recovery']:

    # I2/I1 Recovery
    keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
                ('1323431_8',	'Dataset A -100'),\
                ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
                ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
                ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')]
    keys_all.append(keys_iin)

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=normalize2prepulse,\
                dt=dt,\
                post_process=None,
                setup_sim_args={'sim_args':{'solver': solver},
                                'hold_dur':50})

    # recovery normalized to preprepulse
    keys_iin = [\
    ('7971163_6', 'Dataset -75'),\
    ('7971163_6', 'Dataset -85'),\
    ('7971163_6', 'Dataset -95'),\
    ('7971163_6', 'Dataset -105'),\
    ('7971163_6', 'Dataset -115'),\
    ('7971163_6', 'Dataset -125'),\
    ('7971163_6', 'Dataset -135')]
    keys_all.append(keys_iin)

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(normalize2prepulse, pulse1=1, pulse2=5),\
                dt=dt,\
                post_process=None,
                setup_sim_args={'sim_args':{'solver': solver}})



if run_fits['Inactivation']:

    # inactivation normalized to no prepulse
    keys_iin = [('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),\
                ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),\
                ('7971163_4', 'Dataset 512ms'),\
                ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
                ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5')]#,
#                ('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')]
    keys_all.append(keys_iin)

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,
                keys_iin=keys_iin,\
                model=model,\
                process=None, #to be set by normalizToBaseline
                dt=dt,\
#                post_process=normalizeToBaseline,
                setup_sim_args={'sim_args':{'solver': solver}})
    for key in keys_iin:
        sim_fs[key] = partial(normalizeToBaseline, sim_f=sim_fs[key])
    

    # inactivation normalized to first
    keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
                ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
                ('7971163_5',	'Dataset A -105')]
    keys_all.append(keys_iin)

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(peakCurr, durn=3),\
                dt=dt,\
                post_process=normalizeToFirst,
                process_data=normalizeToFirst_data,
                setup_sim_args={'sim_args':{'solver': solver}})

    #tau inactivation
    keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
                ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]#,\
    #            ('1323431_5',	'Dataset B fast'),\
    #            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
    keys_all.append(keys_iin)
    
    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(calcExpTauInact,func=biExp,x0=biExp_params,\
                      keep=0,calc_dur=1),\
                dt=dt,\
                post_process=partial(func_norm, func=np.log),
                process_data=partial(func_norm_data, func=np.log),
                setup_sim_args={'sim_args':{'solver': solver}})

    # #tau inactivation normalized to first
    # keys_iin = [('1323431_6',	'Dataset -80'), ('1323431_6',	'Dataset -100')]
    # keys_all += keys_iin
    
    # setupSimExp(sim_fs=sim_fs,\
    #             datas=datas,\
    #             data=data,\
    #             exp_parameters=exp_parameters,\
    #             keys_iin=keys_iin,\
    #             model=model,\
    #             process=partial(calcExpTauInact,func=biExp,x0=biExp_params,\
    #                       keep=1,calc_dur=3),\
    #             dt=dt,\
    #             post_process=None,#normalizeToFirst
    #             setup_sim_args={'sim_args':{'solver': solver}})



    # #tau inactivation fast & slow
    # keysf_iin = [('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
    # keyss_iin = [('21647304_2',	'Dataset D Adults'), ('21647304_2',	'Dataset D Pediactric')]
    # #('1323431_5',	'Dataset B fast'),('1323431_5',	'Dataset B slow'),\
    # keys_all.append(keysf_iin)
    # process = partial(calcExpTauInact,func=biExp,x0=biExp_params,\
    #                   keep=[0,1],calc_dur=1)
    # setup_sim_args = {'sim_args':{'solver': solver,
    #                               'retOptions': \
    #                                       {'G': True, 'INa': True, 'INaL': True,\
    #                                         'Open': True, 'RevPot': True},
    #                               'dt' : dt,
    #                               'process' : process,
    #                               'post_process' : partial(func_norm, 
    #                                                         func=lambda vals: np.log(resort(vals)))}} 
    # for keyf, keys in zip(keysf_iin, keyss_iin):
    #     key_dataf = func_norm_data(data[keyf], np.log)
    #     key_datas = func_norm_data(data[keys], np.log)
    #     key_exp_p = exp_parameters.loc[keyf]
    #     voltages, durs, sim_f = setup_sim(model, key_dataf, key_exp_p, **setup_sim_args)
    
    #     sim_fs[keyf] = sim_f
    #     datas[keyf] = np.concatenate((key_dataf, key_datas))

if run_fits['Activation']:

    # activation normalized to driving force
    keys_iin = [('1323431_2',	'Dataset'),\
                ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
                ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
                ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')]
    keys_all.append(keys_iin)

    setupSimExp(sim_fs=sim_fs,
                datas=datas,
                data=data,
                exp_parameters=exp_parameters,
                keys_iin=keys_iin,
                model=model,
                process=peakCurr,
                dt=dt,
                post_process=minMaxNorm,
                process_data=minMaxNorm_data,
                setup_sim_args={'sim_args':{'retOptions': 
                                {'G': False, 'INa': True, 'INaL': False,
                                 'Open': True, 'RevPot': False},
                                'solver': solver}})#'ret': [False,True,False]

   ## iv curve
    keys_iin = [
    ('8928874_7',	'Dataset C day 1'), ('8928874_7',	'Dataset C day 3'),
    ('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
#    ('12890054_3',	'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
#    ('12890054_5',	'Dataset C Control'), ('12890054_5',	'Dataset D Control'),
    ('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
    ('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
    ('1323431_4',	'Dataset B Control'),
    ('21647304_1',	'Dataset B Adults'), ('21647304_1', 'Dataset B Pediatrics')
    ]
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
                setup_sim_args={'sim_args':{'solver': solver}})


if run_fits['Tau Act']:

    # tau activation
    keys_iin = [('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
                ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5'),
                ('7971163_3',	'Dataset C')]
    keys_all.append(keys_iin)

    process = partial(calcExpTauAct,func=monoExp,x0=monoExp_params,\
                      keep=1,calc_dur=(1,1))

   
    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=process,
                dt=dt,\
                post_process=partial(func_norm, func=np.log),
                process_data=partial(func_norm_data, func=np.log),
                setup_sim_args={'sim_args':{'solver': solver}, 'hold_dur':50})




