#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:09:26 2020

@author: grat05
"""


import PyLongQt as pylqt

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))


import atrial_model
from atrial_model.iNa.model_setup import model_param_names, model, mp_locs, sub_mps

import numpy as np
import pickle
import matplotlib.pyplot as plt

class ObjContainer():
    pass

chain = 0
burn_till =  1000

#filename = 'mcmc_OHaraRudy_wMark_INa_0924_1205'

filename = 'mcmc_OHaraRudy_wMark_INa_1012_1149'

filename = 'mcmc_OHaraRudy_wMark_INa_1213_1353'

filename = 'mcmc_OHaraRudy_wMark_INa_0127_1525'
filename = 'mcmc_OHaraRudy_wMark_INa_0215_0722'


base_dir = atrial_model.fit_data_dir+'/'
with open(base_dir+'/'+filename+'.pickle','rb') as file:
    db_full = pickle.load(file)
db = db_full['trace']
db_post = db.warmup_posterior

# base_dir = atrial_model.fit_data_dir+'/'
# with open(base_dir+'/'+filename+'_metadata.pickle','rb') as file:
#     model_metadata = pickle.load(file)
# with open(base_dir+model_metadata.trace_pickel_file,'rb') as file:
#     db = pickle.load(file)
# if db['_state_']['sampler']['status'] == 'paused':
#     current_iter = db['_state_']['sampler']['_current_iter']
#     current_iter -= db['_state_']['sampler']['_burn']
#     for key in db.keys():
#         if key != '_state_':
#             db[key][chain] = db[key][chain][:current_iter]

        

temp = 20
# b_temp = np.median(db_post['b_temp'][chain][burn_till:], axis=0)
# for i in range(db['b_temp'][chain].shape[1]):
#     trace = db['b_temp'][chain][burn_till:, i]
#     f_sig = np.sum(trace > 0)/len(trace)
#     if not (f_sig < 0.05 or f_sig > 0.95):
#         b_temp[i] = 0
        
b_temp = np.zeros_like(mp_locs, dtype=float)
b_temp[[1,4,10,14,21]] = -0.7/10
b_temp[[3,6,9,11,15,20,22]] = 0.6
b_temp[[3]] = -0.9

b_temp[2] = -0.1
b_temp[8] = -0.1
## to be fit
b_temp[0] = 0.08#0.05

#b_temp[[2,4,10,12]] = 0
intercept = np.median(db_post['model_param_intercept'][chain][burn_till:], axis=0)
mp_mean = intercept + b_temp*temp
mp_sd = np.median(db_post['model_param_sd'][chain][burn_till:], axis=0)
mp_sd[[3,11]] = 0.4 #iv curve added variability

#med_model_param = np.median(db_post['model_param'][chain][burn_till:], axis=0)

mp_cor = np.median(db_post['model_param_corr'][chain][burn_till:], axis=0)

mp_cor[4,10] = 0.9

mp_cov = np.outer(mp_sd, mp_sd)*mp_cor
#mp_cov = np.cov((med_model_param.T-np.mean(med_model_param, axis=1)))



#mp_cov[:,18] /= np.sqrt(3)
#mp_cov[18,:] /= np.sqrt(3)
#corrections
#        mp_sd[1] = 0.5
#mp_sd[3] = mp_sd[9]
#        mp_sd[[3,6,9,11,15,20,22]] = 0
#        mp_sd[[1,4,10,14,21]] = 0
#        mp_sd[[5,7,12, 13]] = 0.5
#        mp_sd[[0, 2, 5, 7, 8, 12, 13, 16, 17, 18, 19, 23, 24]] = 0

#        mp_sd[17] = 0.5
#mp_sd[18] = 0.1

#??
#mp_sd = mp_sd

num_trials = 100

#sub_mp_draws = np.random.normal(loc=mp_mean, scale=mp_sd, 
#                                size=(num_trials, len(mp_mean)))


sub_mp_draws = np.random.multivariate_normal(mean=mp_mean, cov=mp_cov, size=num_trials)

# sub_mp_draws[sub_mp_draws[:,2] < -2, 2] = -2
# sub_mp_draws[sub_mp_draws[:,3] < -15.5, 3] = -15
# sub_mp_draws[sub_mp_draws[:,8] < -2.5, 8] = -2
# sub_mp_draws[sub_mp_draws[:,9] > 12.5, 9] = 12

proto = pylqt.Protocols.CurrentClamp()
proto.setCellByName('Human Atrial (Modified)')

cell = proto.cell
cell.setOption('Novel2020', True)

proto.numtrials = num_trials
proto.tMax = 500000
proto.writetime = 490000

ina_name = 'Novel'
cell_params_names = [ina_name+'.'+mp_name for mp_name in model_param_names]

for i in range(len(model_param_names)):
    pvar = proto.pvars.IonChanParam(proto.pvars.normal,
                                    val1 = mp_mean[i],
                                    val2 = mp_sd[i])
    proto.pvars[cell_params_names[i]] = pvar
    pvar = proto.pvars[cell_params_names[i]]
    pvar.trials = sub_mp_draws[:,i]

to_trace = cell.variableSelection
to_trace.add('iNa')
to_trace.add('Novel.m')
to_trace.add('Novel.hf')
to_trace.add('Novel.i1f')
to_trace.add('Novel.jf')
to_trace.add('Novel.hs')
to_trace.add('Novel.i1s')
to_trace.add('Novel.js')
cell.variableSelection = to_trace

proto.measureMgr.addMeasure('vOld', {'maxderiv'})

def calback(*args, **kwargs):
    sim_data = pylqt.Misc.DataReader.readDir(proto.datadir)
    dvdt = np.array([sim_data.meas[i].data[0][-1] for i in range(len(sim_data.meas))])
    plt.boxplot(dvdt)
#proto.runSim()
sim_runner = pylqt.Misc.RunSim(proto)
sim_runner.run()

settings = pylqt.Misc.SettingsIO.getInstance()
settings.writeSettings(proto.datadir+"/simvars.xml", proto)