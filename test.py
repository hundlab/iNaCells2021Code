#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:06:36 2020

@author: grat05
"""

#from iNa_models import Koval_ina, OHaraRudy_INa
from iNa_models_ode import OHaraRudy_INa
from scripts import load_data_parameters, load_all_data, all_data
import iNa_fit_functions
from iNa_fit_functions import normalize2prepulse, setup_sim, run_sim, \
calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp,\
calcExpTauAct, triExp, biExp
from sklearn.preprocessing import minmax_scale
from scipy import integrate

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial
import copy
from multiprocessing import Pool
from sklearn.preprocessing import minmax_scale

np.seterr(all='ignore')

iNa_fit_functions.plot1 = False #sim
iNa_fit_functions.plot2 = False #diff
iNa_fit_functions.plot3 = True #tau

model = OHaraRudy_INa
dt = 0.05
monoExp_params = [-1,1,0]
biExp_params = np.array([1,10,1])#np.array([0,1,0,10,0])
triExp_params = np.array([-1,2,-1,3,-1,10000,0])

exp_parameters_gating, data_gating = load_data_parameters('/params old/iNa_old.xlsx','gating', data=all_data)
exp_parameters_iv, data_iv = load_data_parameters('/params old/iNa_old.xlsx','iv_curve', data=all_data)

exp_parameters = exp_parameters_gating.append(exp_parameters_iv, sort=False)
data_gating.update(data_iv)
data = data_gating

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

def minNorm(model_params, sim_f):
    sim_res = sim_f(model_params)
    return sim_res/np.abs(np.min(sim_res))

def minNorm_data(data):
    data = np.copy(data)
    data[:,1] = data[:,1]/np.abs(np.min(data[:,1]))
    return data

def minMaxNorm(model_params, sim_f, feature_range=(0, 1)):
    return minmax_scale(sim_f(model_params), feature_range=feature_range)

def minMaxNorm_data(data, feature_range=(0, 1)):
    data = np.copy(data)
    minmax_scale(data[:,1], feature_range=feature_range, copy=False)
    return data

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



#def biExp(t,A1,tau1,A2,tau2,A0,sign=-1):
#    return -np.exp(-t/tau1+A1)-np.exp(sign*t/tau2+A2)+A0

sim_fs = []
datas = []


#best fit at one point
#model_params = np.array([ 0.        ,  0.        ,  0, -1.84448536, -1.21581823,
      #   0.04750437,  0.09809738,  0.78      ,  0.725     , -0.031     ,
      # -2.144     ,  0.84      ,  2.018     ,  0.276     ,  7.073     ,
      #   4.641     , -1.427     ,  1.513     , -1.345     , -0.26      ,
      # -3.422     ,  6.159     ,  0.        ,  0.        ,  0.        ,
      #   0.        ,  0.        ,  0.        ,  0.        ])
#a terrible fit
#model_params = np.array([ 0.06721265, -0.08104161, -0.9442514 ,  0.93791262,  0.24704036,
       # -0.54269396, -1.62755681,  0.41342681, -0.07028288, -1.84152805,
       # -0.35309849, -0.23240665, -0.3949827 , -0.77066988,  0.74966051,
       # -0.74232144, -0.6377425 ,  0.34105554])
model_params = np.zeros(model.num_params)
#model_params = np.random.normal(loc=0, scale=0.5, size=model.num_params)
#model_params = np.random.uniform(low=-5, high=5, size=model.num_params)
#model_params[[7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 30]] = np.array([-0.16394553,  1.42641913,  1.28038251,  1.29642736, -1.16718484,
#       -4.44375614,  1.30915332, -4.0239836 , 13.42958559,  2.20538925,
#        1.69031741, -1.93559858])
model_params[[7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 30]] = np.array([ 1.81552155e-01, -5.05609163e+00,  2.82459638e+01,  2.20175018e-01,
       -3.55661129e-03, -6.74500269e+01,  1.66096238e-02, -3.33020753e+00,
        1.62150070e+01,  7.80529744e-01,  5.35441384e-01, -5.99818295e-01])
model_params[[5,6]] = [np.log(1/100), 16]

#model_params[2:7] = 1.49431475, -1.84448536, -1.21581823,  0.04750437,  0.09809738
#model_params[2:7] = 0
#model_params[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = [ 0.505, -0.533, -0.286 , 2.558 , 0.92 ,  0.333 ,14.476 , 2.169 , 1.876, -0.487
# , 3.163 , 3.814 , 2.584 ,-0.772 ,-0.129 , 0.528, -0.868 , 0.319 ,-3.578 , 0.149]
#model_params[7] = 2
# model_params[7] = np.log(2)
# model_params[8] = np.log(1/10 * 1/1.2)
# model_params[9] = np.log(1/1.2)
# model_params[10] = np.log(1/1.2)
# model_params[11] = np.log(1/1.2)
# model_params[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = [ 0.505, -0.533, -0.286 , 2.558 , 0.92 ,  0.333 ,14.476 , 2.169 , 1.876, -0.487
#  , 3.163 , 3.814 , 2.584 ,-0.772 ,-0.129 , 0.528, -0.868 , 0.319 ,-3.578 , 0.149]
#model_params[16] = -2
mp_locs = []
sub_mps = model_params[mp_locs]
keys_all = []
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

solver = partial(integrate.solve_ivp, method='BDF')

# ##iv curve
# keys_iin = [
# ('8928874_7',	'Dataset C day 1')]#, ('8928874_7',	'Dataset C day 3'),
# [('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
# #('12890054_3',	'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
# #('12890054_5',	'Dataset C Control'), ('12890054_5',	'Dataset D Control'),
# ('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
# ('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
# ('1323431_4',	'Dataset B Control'),
# ('21647304_1',	'Dataset B Adults'), ('21647304_1', 'Dataset B Pediatrics')
# ]
# keys_all += keys_iin

# setupSimExp(sim_fs=sim_fs,\
#             datas=datas,\
#             data=data,\
#             exp_parameters=exp_parameters,\
#             keys_iin=keys_iin,\
#             model=model,\
#             process=peakCurr,\
#             dt=dt,\
#             process_data=minNorm_data,#partial(minMaxNorm_data, feature_range=(-1, 0)),\
#             post_process=minNorm,#partial(minMaxNorm, feature_range=(-1, 0)),
#             setup_sim_args={'sim_args':{'solver': solver}})



mp_locs += [7] + list(range(17,22))

# I2/I1 Recovery
keys_iin = [('1323431_8', 'Dataset A -140')]#, ('1323431_8',	'Dataset A -120'),\
[            ('1323431_8',	'Dataset A -100'),\
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
('7971163_6', 'Dataset -75')]#,\
[('7971163_6', 'Dataset -85'),\
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



# mp_locs += list(range(7,17))

# #inactivation normalized to no prepulse
# keys_iin = [('7971163_4', 'Dataset 32ms')]#, ('7971163_4', 'Dataset 64ms'),\
# [            ('7971163_4', 'Dataset 512ms'),\
#             ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),\

#             ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
#             ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5')]
# #('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
# keys_all += keys_iin

# setupSimExp(sim_fs=sim_fs,\
#             datas=datas,\
#             data=data,\
#             exp_parameters=exp_parameters,\
#             keys_iin=keys_iin,\
#             model=model,\
#             process=partial(normalized2val, durn=3),\
#             dt=dt,\
#             post_process=normalizeToBaseline,
#             setup_sim_args={'sim_args':{'solver': solver}})

# inactivation normalized to first
keys_iin = [('7971163_5',	'Dataset A -65')]#, ('7971163_5',	'Dataset A -75')]#,\
[            ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
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


# mp_locs += list(range(2,7))

# #activation normalized to driving force
# keys_iin = [('1323431_2',	'Dataset')]#,\
# [            ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
#             ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
#             ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')]
# keys_all += keys_iin

# setupSimExp(sim_fs=sim_fs,\
#             datas=datas,\
#             data=data,\
#             exp_parameters=exp_parameters,\
#             keys_iin=keys_iin,\
#             model=model,\
#             process=peakCurr,\
#             dt=dt,\
#             post_process=None,
#             setup_sim_args={"hold_dur": 50,\
#                             'sim_args':{
#                                 'retOptions': \
#                                     {'G': False, 'INa': True, 'INaL': False,\
#                                       'Open': True, 'RevPot': False},\
#                                 'solver': solver}})





# #tau inactivation
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



# #tau inactivation fast & slow
# keys_iin = [('21647304_2', 'Dataset C Adults'), ('21647304_2',	'Dataset D Adults'),\
#             ('21647304_2', 'Dataset C Pediactric'), ('21647304_2',	'Dataset D Pediactric')]
# #('1323431_5',	'Dataset B fast'),('1323431_5',	'Dataset B slow'),\
# keys_all += keys_iin
# process = partial(calcExpTauInact,func=biExp,x0=biExp_params,\
#                   keep=[0,1],calc_dur=1)
# post_process = resort
# setup_sim_args = {'sim_args':{'solver': solver,
#                               'retOptions': \
#                                       {'G': True, 'INa': True, 'INaL': True,\
#                                         'Open': True, 'RevPot': True}}}

# for i in range(0,len(keys_iin),2):
#     keyf = keys_iin[i]
#     keys = keys_iin[i+1]
#     key_dataf = data[keyf]
#     key_datas = data[keys]
#     key_exp_p = exp_parameters.loc[keyf]
#     voltages, durs, sim_f = setup_sim(model, key_dataf, key_exp_p, process, dt=dt, **setup_sim_args)
#     if not post_process is None:
#         sim_fw = partial(post_process, sim_f=sim_f)
#     else:
#         sim_fw = sim_f

#     sim_fs.append(sim_fw)
#     datas.append(np.concatenate((key_dataf, key_datas)))

model_params_full = np.zeros(model.num_params)
mp_locs = np.arange(model.num_params)#[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 29, 30]#np.arange(model.num_params)#  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
diff_fn = partial(calc_diff, model_parameters_full=model_params_full,\
                  mp_locs=mp_locs, sim_func=sim_fs, data=datas,l=0)

#res = optimize.least_squares(diff_fn, sub_mps, \
#                             bounds=([0]*len(sub_mps), [np.inf]*len(sub_mps)))
iNa_fit_functions.plot1 = True #sim
iNa_fit_functions.plot2 = True #diff
iNa_fit_functions.plot3 = False #tau
diff_fn(model_params, exp_params=exp_parameters, keys=keys_all)
vOld = np.arange(-150,50)
self = model(*model_params)

tj = self.tj_baseline + 1.0 / (self.tj_mult1 * np.exp(-(vOld + 100.6) / self.tj_tau1) +
                                   self.tj_mult2 * np.exp((vOld + 0.9941) / self.tj_tau2));
plt.figure('tau inactivation')
plt.plot(vOld, tj)

thf = 1.0 / (self.thf_mult1 * np.exp(-(vOld + 1.196) / self.thf_tau1) +
                    self.thf_mult2 * np.exp((vOld + 0.5096) / self.thf_tau2));

plt.figure('tau inactivation')             
plt.plot(vOld, thf)

plt.show()
