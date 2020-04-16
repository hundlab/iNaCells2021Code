#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:09:01 2020

@author: dgratz
"""

#from iNa_models import Koval_ina, OHaraRudy_INa
from iNa_models_ode import OHaraRudy_INa
from scripts import load_data_parameters, all_data
from iNa_fit_functions import normalize2prepulse, setup_sim, \
calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp, biExp,\
calcExpTauAct, biExp_params, monoExp_params
from setup_sim_functions import setupSimExp, normalizeToBaseline, normalizeToFirst,\
    resort, minNorm_data, minNorm


import numpy as np
from scipy import optimize
from functools import partial
from multiprocessing import Pool
from scipy import integrate


#import sys
#sys.path.append('./models/build/Debug/')
#import models

np.seterr(all='ignore')

try: run_fits
except NameError: 
    run_fits = {'Activation':   True,\
                'Inactivation': True,\
                'Recovery':     True,\
                'Tau Act':      False
                }

model = OHaraRudy_INa#models.iNa.OharaRudy_INa#OHaraRudy_INa
retOptions  = model().retOptions
dt = 0.05

exp_parameters_gating, data_gating = load_data_parameters('/params old/INa_old.xlsx','gating', data=all_data)
exp_parameters_iv, data_iv = load_data_parameters('/params old/INa_old.xlsx','iv_curve', data=all_data)

exp_parameters = exp_parameters_gating.append(exp_parameters_iv, sort=False)
data_gating.update(data_iv)
data = data_gating

#fits_results_joint = pickle.load(open('./fits_res_joint_ohara_0306.pkl','rb'))
#res = fits_results_joint['group']

model_params_initial = np.zeros(model.num_params)#np.array([ 0.        ,  0.        ,  1.49431475, -1.84448536, -1.21581823,
#        0.04750437,  0.09809738,  0.        ,  0.        ,  0.        ,
#        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#        0.        ,  0.        ,  1.487     , -1.788     , -0.254     ,
#       -3.423     ,  4.661     ,  0.        ,  0.        ,  0.        ,
#        0.        ,  0.        ,  0.        ,  0.        ])#np.zeros(model().num_params)
#model_params_initial[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = [ 0.505, -0.533, -0.286 , 2.558 , 0.92 ,  0.333 ,14.476 , 2.169 , 1.876, -0.487
# , 3.163 , 3.814 , 2.584 ,-0.772 ,-0.129 , 0.528, -0.868 , 0.319 ,-3.578 , 0.149]


sim_fs = {}
datas = {}
keys_all = []

solver = partial(integrate.solve_ivp, method='BDF')
mp_locs = []


if run_fits['Recovery']:
    mp_locs += [7] + list(range(17,22))# + [31,32] # [7]

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
                setup_sim_args={'sim_args':{'solver': solver}})

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
    mp_locs += list(range(7,12)) + [16,30] #7,17

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
                setup_sim_args={'sim_args':{'solver': solver}})

    # #tau inactivation
    # keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
    #             ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]#,\
    # #            ('1323431_5',	'Dataset B fast'),\
    # #            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
    # keys_all.append(keys_iin)
    
    # setupSimExp(sim_fs=sim_fs,\
    #             datas=datas,\
    #             data=data,\
    #             exp_parameters=exp_parameters,\
    #             keys_iin=keys_iin,\
    #             model=model,\
    #             process=partial(calcExpTauInact,func=biExp,x0=biExp_params,\
    #                   keep=1,calc_dur=1),\
    #             dt=dt,\
    #             post_process=None,
    #             setup_sim_args={'sim_args':{'solver': solver}})

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
    # keys_iin = [('21647304_2', 'Dataset C Adults'), ('21647304_2',	'Dataset D Adults'),\
    #             ('21647304_2', 'Dataset C Pediactric'), ('21647304_2',	'Dataset D Pediactric')]
    # #('1323431_5',	'Dataset B fast'),('1323431_5',	'Dataset B slow'),\
    # keys_all += keys_iin
    # process = partial(calcExpTauInact,func=biExp,x0=biExp_params,\
    #                   keep=[0,1],calc_dur=1)
    # setup_sim_args = {'sim_args':{'solver': solver,
    #                               'retOptions': \
    #                                       {'G': True, 'INa': True, 'INaL': True,\
    #                                         'Open': True, 'RevPot': True},
    #                               'dt' : dt,
    #                               'process' : process,
    #                               'post_process' : resort}}
    
    # for i in range(0,len(keys_iin),2):
    #     keyf = keys_iin[i]
    #     keys = keys_iin[i+1]
    #     key_dataf = data[keyf]
    #     key_datas = data[keys]
    #     key_exp_p = exp_parameters.loc[keyf]
    #     voltages, durs, sim_f = setup_sim(model, key_dataf, key_exp_p, process, dt=dt, **setup_sim_args)
    
    #     sim_fs.append(sim_f)
    #     datas.append(np.concatenate((key_dataf, key_datas)))

if run_fits['Activation']:
    mp_locs += list(range(2,7)) + [29]#list(range(2,5))

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
                post_process=None,
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
    sim_fs = []
    datas = []

    # tau activation
    keys_iin = [('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
                ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5')]
    process = partial(calcExpTauAct,func=monoExp,x0=monoExp_params,\
                      keep=1,calc_dur=(1,1))

    model_params_initial = np.ones(model().num_params)
    mp_locs = np.arange(model().num_params)#[2,3,4,13]
    sub_mps = model_params_initial[mp_locs]


    for key in keys_iin:
        key_data = data[key]
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, process, dt=dt/10, hold_dur=50)
        sim_f.keywords['durs'][1] = 10

        sim_fs.append(sim_f)
        datas.append(key_data)

    if __name__ == '__main__':
        np.seterr(all='ignore')
        with Pool(processes=20) as proc_pool:
            diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
                                  mp_locs=mp_locs, sim_func=sim_fs, data=datas,l=0, pool=proc_pool)
            res = optimize.least_squares(diff_fn, sub_mps, \
                                             bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
            fit_results['Group Tau Act'] = res

mp_locs = list(set(mp_locs))
sub_mps = model_params_initial[mp_locs]
sub_mp_bounds = np.array(model().param_bounds)[mp_locs]
