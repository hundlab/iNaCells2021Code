#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:20:18 2020

@author: grat05
"""

#from iNa_models import Koval_ina, OHaraRudy_INa
from iNa_models_ode import OHaraRudy_INa
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
from sklearn.preprocessing import minmax_scale
from scipy import integrate


#import sys
#sys.path.append('./models/build/Debug/')
#import models

np.seterr(all='ignore')

iNa_fit_functions.plot1 = False #sim
iNa_fit_functions.plot2 = False #diff
iNa_fit_functions.plot3 = False #tau

#ina_fits = h5py.File('ina_fits.h5','w')


run_fits = {'Activation':   False,\
            'Inactivation': True,\
            'Recovery':     True,\
            'Tau Act':      False
            }

try: fit_results
except NameError: fit_results = {}
try: fit_results_joint
except NameError: fit_results_joint = {}
try: fit_params
except NameError: fit_params = {}


model = OHaraRudy_INa#models.iNa.OharaRudy_INa#OHaraRudy_INa
retOptions  = model().retOptions
dt = 0.05
monoExp_params = [-1,1,0]
biExp_params = np.array([1,10,1])#np.array([0,1,0,10,0])

useJoint = True


exp_parameters_gating, data_gating = load_data_parameters('/params old/iNa_old.xlsx','gating', data=all_data)
exp_parameters_iv, data_iv = load_data_parameters('/params old/iNa_old.xlsx','iv_curve', data=all_data)

exp_parameters = exp_parameters_gating.append(exp_parameters_iv, sort=False)
data_gating.update(data_iv)
data = data_gating

#tau inact
#partial(calcExpTauInact,func=monoExp,x0=np.ones(3))
# [-1,1,-1,1000,0]
# [-1,1,0]
# bounds=[(0, np.inf)]*len(sub_mps)

def resort(model_parameters, sim_f):
    vals = sim_f(model_parameters)
    return vals.flatten('F')

def print_fun(x, f, accepted):
    print("Minimum found:", bool(accepted), ", Cost:", f)
    print("At:", x)

def save_results(x, f, accepted, results=None):
    print("Minimum found:", bool(accepted), ", Cost:", f)
    print("At:", x)
    if not results is None:
        results.append((x,f))

#called after, doesn't work
def check_bounds(f_new, x_new, f_old, x_old, bounds=None, **kwargs):
    print("---")
    print(f_new, x_new, f_old, x_old)
    print("---")
    if bounds is None:
        return True
    else:
        aboveMin = bool(np.all(x_new > bounds[:,0]))
        belowMax = bool(np.all(x_new < bounds[:,1]))
        print("---")
        print(x_new, aboveMin and belowMax)
        print("---")
        return aboveMin and belowMax

def lstsq_wrap(fun, x0, bounds=None, **kwargs):
    if bounds is None:
        bounds = (-np.inf,np.inf)
    else:
        #it had best be convertable to a numpy array
        bounds = np.array(bounds).T
    options = None
    if 'ssq' in kwargs:
        options = {'ssq': kwargs['ssq']}
    try:
        res = optimize.least_squares(fun, x0, bounds=bounds, kwargs=options)
        res.resid = res.fun
        res.fun = res.cost
        return res
    except ValueError:
        return optimize.OptimizeResult(x=x0, success=False, status=-1, fun=np.inf)


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

        sim_fs.append(sim_fw)
        datas.append(key_data)

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


#fits_results_joint = pickle.load(open('./fits_res_joint_ohara_0306.pkl','rb'))
#res = fits_results_joint['group']


sim_fs = []
datas = []
keys_all = []

solver = partial(integrate.solve_ivp, method='BDF')

model_params = np.zeros(model.num_params)#np.array([ 0.        ,  0.        ,  1.49431475, -1.84448536, -1.21581823,
#        0.04750437,  0.09809738,  0.        ,  0.        ,  0.        ,
#        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#        0.        ,  0.        ,  1.487     , -1.788     , -0.254     ,
#       -3.423     ,  4.661     ,  0.        ,  0.        ,  0.        ,
#        0.        ,  0.        ,  0.        ,  0.        ])#np.zeros(model().num_params)
#model_params[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = [ 0.505, -0.533, -0.286 , 2.558 , 0.92 ,  0.333 ,14.476 , 2.169 , 1.876, -0.487
# , 3.163 , 3.814 , 2.584 ,-0.772 ,-0.129 , 0.528, -0.868 , 0.319 ,-3.578 , 0.149]
#model_params[[7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 30]] = np.array([ 6.58208045e-07,  1.17536742e-06, -8.42258691e-07,  3.37525667e-07,
#        4.50068931e-07, -5,  9.53840262e-08,  3.55121179e-07,
#        9.86234378e-07,  4.09994171e-07,  4.64260091e-07, -1.51089218e-06])

# model_params[[7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 30]] = np.array([-0.16394553,  1.42641913,  1.28038251,  1.29642736, -1.16718484,
#        -4.44375614,  1.30915332, -4.0239836 , 13.42958559,  2.20538925,
#         1.69031741, -1.93559858])
#model_params[[8,9]] = [np.log(1/100), 16]
#model_params[[18,19]] = [np.log(1/100), 16]

mp_locs = []



if run_fits['Recovery']:
    mp_locs += list(range(17,22)) + [31,32] # [7]

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
    keys_all += keys_iin

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
    keys_all += keys_iin

    setupSimExp(sim_fs=sim_fs,\
                datas=datas,\
                data=data,\
                exp_parameters=exp_parameters,\
                keys_iin=keys_iin,\
                model=model,\
                process=partial(normalized2val, durn=3),\
                dt=dt,\
                post_process=normalizeToBaseline,
                setup_sim_args={'sim_args':{'solver': solver}})

    # inactivation normalized to first
    keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
                ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
                ('7971163_5',	'Dataset A -105')]
    keys_all += keys_iin

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

    #tau inactivation
    # keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
    #             ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]#,\
    # #            ('1323431_5',	'Dataset B fast'),\
    # #            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
    # keys_all += keys_iin
    
    # setupSimExp(sim_fs=sim_fs,\
    #             datas=datas,\
    #             data=data,\
    #             exp_parameters=exp_parameters,\
    #             keys_iin=keys_iin,\
    #             model=model,\
    #             process=partial(calcExpTauInact,func=biExp,x0=biExp_params,\
    #                   keep=0,calc_dur=1),\
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



#     #tau inactivation fast & slow
#     keys_iin = [('21647304_2', 'Dataset C Adults'), ('21647304_2',	'Dataset D Adults'),\
#                 ('21647304_2', 'Dataset C Pediactric'), ('21647304_2',	'Dataset D Pediactric')]
# #('1323431_5',	'Dataset B fast'),('1323431_5',	'Dataset B slow'),\
#     keys_all += keys_iin
#     process = partial(calcExpTauInact,func=biExp,x0=biExp_params,\
#                       keep=[0,1],calc_dur=1)
#     post_process = resort
#     setup_sim_args = {'sim_args':{'solver': solver}}
    
#     for i in range(0,len(keys_iin),2):
#         keyf = keys_iin[i]
#         keys = keys_iin[i+1]
#         key_dataf = data[keyf]
#         key_datas = data[keys]
#         key_exp_p = exp_parameters.loc[keyf]
#         voltages, durs, sim_f = setup_sim(model, key_dataf, key_exp_p, process, dt=dt, **setup_sim_args)
#         if not post_process is None:
#             sim_fw = partial(post_process, sim_f=sim_f)
#         else:
#             sim_fw = sim_f
    
#         sim_fs.append(sim_fw)
#         datas.append(np.concatenate((key_dataf, key_datas)))


if run_fits['Activation']:
    mp_locs += list(range(2,7)) + [29]#list(range(2,5))

    # activation normalized to driving force
    keys_iin = [('1323431_2',	'Dataset'),\
                ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
                ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
                ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')]
    keys_all += keys_iin

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

    #iv curve
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
    keys_all += keys_iin
    
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

if __name__ == '__main__':
    with Pool(processes=20) as proc_pool:
        mp_locs = list(set(mp_locs))
        sub_mps = model_params[mp_locs]
        sub_mp_bounds = np.array(model().param_bounds)[mp_locs]
        sub_mp_bounds = sub_mp_bounds + sub_mps[...,None]
        min_res = []
        all_res = []

        diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas,\
                            l=0,pool=proc_pool,ssq=True,\
                            results=all_res)
        minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False, "x_scale": 'jac'}}#"bounds": sub_mp_bounds,
        # res = optimize.basinhopping(diff_fn, sub_mps, \
        #                             minimizer_kwargs=minimizer_kwargs,\
        #                             niter=10, T=1,\
        #                             callback=partial(save_results, results=min_res),\
        #                             stepsize=1)#T=80
        # accept_test=partial(check_bounds, bounds=sub_mp_bounds))
#            minimizer_kwargs = {"method": "BFGS", "options": {"maxiter":100}}
        res = optimize.dual_annealing(diff_fn, bounds=sub_mp_bounds,\
                                          local_search_options=minimizer_kwargs,\
                                          maxiter=100)
#        res = optimize.least_squares(diff_fn, sub_mps, \
#                        bounds=np.array(model().param_bounds)[mp_locs].T)
        res.keys_all = keys_all
        res.all_res = all_res
        res.min_res = min_res
        res.mp_locs = mp_locs
        fit_key = frozenset(rfs for rfs in run_fits if run_fits[rfs])
        fit_results_joint[fit_key] = res

        filename = './fits_res_joint_ohara_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}.pkl'
        filename = filename.format(cdate=datetime.datetime.now())
        with open(filename, 'wb') as file:
            pickle.dump(fit_results_joint, file)

        #plot!
        iNa_fit_functions.plot1 = False #sim
        iNa_fit_functions.plot2 = True #diff
        iNa_fit_functions.plot3 = False #tau

        error = diff_fn(res.x, exp_params=exp_parameters, keys=keys_all)


if run_fits['Tau Act']:
    sim_fs = []
    datas = []

    # tau activation
    keys_iin = [('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
                ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5')]
    process = partial(calcExpTauAct,func=monoExp,x0=monoExp_params,\
                      keep=1,calc_dur=(1,1))

    model_params = np.ones(model().num_params)
    mp_locs = np.arange(model().num_params)#[2,3,4,13]
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
