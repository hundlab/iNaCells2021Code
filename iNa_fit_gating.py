#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:20:18 2020

@author: grat05
"""

from iNa_models import Koval_ina, OHaraRudy_INa
from scripts import load_data_parameters, load_all_data, all_data
import iNa_fit_functions
from iNa_fit_functions import normalize2prepulse, setup_sim, run_sim, \
calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp, biExp,\
calcExpTauAct


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial
import copy
from multiprocessing import Pool
import pickle
import datetime


iNa_fit_functions.plot1 = False #sim
iNa_fit_functions.plot2 = False #diff
iNa_fit_functions.plot3 = False #tau

#ina_fits = h5py.File('ina_fits.h5','w')


run_fits = {'Recovery':     False,\
            'Activation':   False,\
            'Inactivation': False,
            'TauInact':     False,
            'Group':        True,
            'Tau Act':         False
            }

try: fit_results
except NameError: fit_results = {}
try: fit_results_joint
except NameError: fit_results_joint = {}
try: fit_params
except NameError: fit_params = {}


model = OHaraRudy_INa
dt = 0.05
monoExp_params = [-1,1,0]
biExp_params = np.array([-1,1,-1,1000,0])

useJoint = True


exp_parameters, data = load_data_parameters('iNa_dims.xlsx','gating', data=all_data)

#tau inact
#partial(calcExpTauInact,func=monoExp,x0=np.ones(3))
# [-1,1,-1,1000,0]
# [-1,1,0]
# bounds=[(0, np.inf)]*len(sub_mps)


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

def setupSimExp(sim_fs, datas, data, exp_parameters, keys_iin, model, process, dt, post_process=None, setup_sim_args={}):
    for key in keys_iin:
        key_data = data[key]
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, process, dt=dt, **setup_sim_args)
        if not post_process is None:
            sim_fw = partial(post_process, sim_f=sim_f)
        else:
            sim_fw = sim_f

        sim_fs.append(sim_fw)
        datas.append(key_data)

#get group optimization first
if run_fits['Group']:
    sim_fs = []
    datas = []
    keys_all = []

    model_params = np.ones(model.num_params)
    mp_locs = np.arange(model.num_params)
    sub_mps = model_params[mp_locs]

    # inactivation normalized to no prepulse
    keys_iin = [('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),\
                ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),\
                ('7971163_4', 'Dataset 512ms'),\
                ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
                ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5')]
    keys_all += keys_iin

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(normalized2val, durn=3),\
                dt=dt,\
                post_process=normalizeToBaseline)

    # inactivation normalized to first
    keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
                ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
                ('7971163_5',	'Dataset A -105'),\
                ('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')]
    keys_all += keys_iin

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(peakCurr, durn=3),\
                dt=dt,\
                post_process=normalizeToFirst)

    # I2/I1 Recovery
    keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
                ('1323431_8',	'Dataset A -100'),\
                ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
                ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
                ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')]
    keys_all += keys_iin

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=normalize2prepulse,\
                dt=dt,\
                post_process=None)

    # recovery normalized to preprepulse
    keys_iin = [\
    ('7971163_6', 'Dataset -75'),\
    ('7971163_6', 'Dataset -85'),\
    ('7971163_6', 'Dataset -95'),\
    ('7971163_6', 'Dataset -105'),\
    ('7971163_6', 'Dataset -115'),\
    ('7971163_6', 'Dataset -125'),\
    ('7971163_6', 'Dataset -135')]
    keys_all += keys_iin

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(normalize2prepulse, pulse1=1, pulse2=5),\
                dt=dt,\
                post_process=None)

    #tau inactivation
    keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
                ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5'),\
                ('1323431_5',	'Dataset B fast'),\
                ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
    keys_all += keys_iin

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(calcExpTauInact,func=biExp,x0=biExp_params,\
                      keep=1,calc_dur=1),\
                dt=dt,\
                post_process=None)

#    #tau inactivation normalized to first
#    keys_iin = [('1323431_6',	'Dataset -80'), ('1323431_6',	'Dataset -100')]
#    keys_all.add(keys_iin)
#
#    setupSimExp(sim_fs=sim_fs,\
#                datas=datas,\
#                data=data,\
#                exp_parameters=exp_parameters,\
#                keys_iin=keys_iin,\
#                model=model,\
#                process=partial(calcExpTauInact,func=biExp,x0=biExp_params,\
#                          keep=1,calc_dur=3),\
#                dt=dt,\
#                post_process=normalizeToFirst)



    # Activation normalized to driving force
    keys_iin = [('1323431_2',	'Dataset'),('8928874_7',	'Dataset D fresh'),\
                ('8928874_7',	'Dataset D day 1'),('8928874_7',	'Dataset D day 3'),\
                ('8928874_7',	'Dataset D day 5'),\
                ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')]
    keys_all += keys_iin

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=peakCurr,\
                dt=dt,\
                post_process=None,
                setup_sim_args={'ret': [False,True,False]})

    if __name__ == '__main__':
        np.seterr(all='ignore')
        with Pool(processes=20) as proc_pool:
            diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                            mp_locs=mp_locs, sim_func=sim_fs, data=datas, l=0,pool=proc_pool)
            res = optimize.least_squares(diff_fn, sub_mps, \
                            bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
            res.keys_all = keys_all
            fit_results_joint['group'] = res

            filename = './fits_res_joint_ohara_{cdate.month:02d}{cdate.day:02d}.pkl'
            filename = filename.format(cdate=datetime.datetime.now())
            with open(filename, 'wb') as file:
                pickle.dump(fit_results_joint, file)

            iNa_fit_functions.plot1 = False #sim
            iNa_fit_functions.plot2 = True #diff
            iNa_fit_functions.plot3 = False #tau

            diff_fn(res.x)



if run_fits['Recovery']:
    # I2/I1
    keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
                ('1323431_8',	'Dataset A -100')]
    process = normalize2prepulse

    for key in keys_iin:
        key_data = data[key]
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, process, dt=dt)

        model_params = np.ones(model.num_params)
        mp_locs = np.arange(model.num_params)#[2,3,13]#4
        sub_mps = model_params[mp_locs]

        diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                          mp_locs=mp_locs, sim_func=sim_f, data=key_data, l=0)
        res = optimize.least_squares(diff_fn, sub_mps, \
                                     bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
        fit_results[key] = res

#next get the individual optimums
if run_fits['Inactivation']:
    # normalized to no prepulse
    keys_iin = [('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),\
                ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),\
                ('7971163_4', 'Dataset 512ms')]
    process = partial(normalized2val, durn=3)

    for key in keys_iin:
        key_data = data[key]
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, process, dt=dt)

        sim_f_baseline = copy.deepcopy(sim_f)
        sim_f_baseline.keywords['durs'] = [1,10]
        sim_f_baseline.keywords['voltages'] = [\
                               sim_f_baseline.keywords['voltages'][0], \
                               sim_f_baseline.keywords['voltages'][3]]
        sim_f_baseline.keywords['process'] = peakCurr

        model_params = np.ones(model.num_params)
        if useJoint:
            model_params_joint = [result.x for joint, result in fit_results_joint.items()\
                                  if key in joint][0]
            model_params = model_params_joint
        mp_locs = np.arange(model.num_params)#[0,1,3,5,6,12]
        sub_mps = model_params[mp_locs]
        baseline = sim_f_baseline(model_params)
        process = partial(process, val=baseline[0])
        sim_f.keywords['process'] = process

        diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                          mp_locs=mp_locs, sim_func=sim_f, data=key_data, l=0)
        res = optimize.least_squares(diff_fn, sub_mps, \
                                     bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
        fit_results[key] = res



if run_fits['TauInact']:

    # tau inactivation
    keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
                ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]
    process = partial(calcExpTauInact,func=biExp,x0=biExp_params,\
                      keep=1,calc_dur=1)

    for key in keys_iin:
        key_data = data[key]
        key_exp_p = exp_parameters.loc[key]
        voltages, durs, sim_f = setup_sim(model, key_data, key_exp_p, process, dt=dt)

        model_params = np.ones(model.num_params)
        mp_locs = np.arange(model.num_params)#[2,3,4,13]
        sub_mps = model_params[mp_locs]

        diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                          mp_locs=mp_locs, sim_func=sim_f, data=key_data,l=0)
        res = optimize.least_squares(diff_fn, sub_mps, \
                                     bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
        fit_results[key] = res

if run_fits['Tau Act']:
    sim_fs = []
    datas = []

    # tau activation
    keys_iin = [('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
                ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5')]
    process = partial(calcExpTauAct,func=monoExp,x0=monoExp_params,\
                      keep=1,calc_dur=(1,1))

    model_params = np.ones(model.num_params)
    mp_locs = np.arange(model.num_params)#[2,3,4,13]
    sub_mps = model_params[mp_locs]


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
            diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                                  mp_locs=mp_locs, sim_func=sim_fs, data=datas,l=0, pool=proc_pool)
            res = optimize.least_squares(diff_fn, sub_mps, \
                                             bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
            fit_results['Group Tau Act'] = res
