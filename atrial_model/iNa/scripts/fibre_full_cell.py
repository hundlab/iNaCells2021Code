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

import datetime
from time import sleep
import xarray


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


settings = pylqt.Misc.SettingsIO.getInstance()
proto = settings.readSettings(base_dir + 'fiber_full_cell.xml')
proto.setDataDir(basedir = 'U:/data')
#proto.grid.removeColumns(60, 40)
grid_shape = proto.grid.shape

num_sims = 10#40

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
b_temp[[1,4,10,14,21]] = -0.07
b_temp[[3,6,9,11,15,20,22]] = 0.4
b_temp[[3]] = -0.4

#modifications
b_temp[2] = -0.08
b_temp[8] = -0.08
b_temp[9] = 0.8
#b_temp[4] = -0.1
#b_temp[22] += 0.9
#b_temp[23] = -0.1
b_temp[[6,11,15,22]] = 0

#b_temp[5] = -0.08

## to be fit
b_temp[0] = 0.05

#b_temp[[2,4,10,12]] = 0
intercept = np.median(db_post['model_param_intercept'][chain][burn_till:], axis=0)
mp_mean = intercept + b_temp*temp
mp_sd = np.median(db_post['model_param_sd'][chain][burn_till:], axis=0)
mp_sd[[3,11]] = 0.4 #iv curve added variability

#med_model_param = np.median(db_post['model_param'][chain][burn_till:], axis=0)

mp_cor = np.mean(db_post['model_param_corr'][chain][burn_till:], axis=0)

#mp_cor[4,10] = mp_cor[10,4] = 0.4

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


#sub_mp_draws = np.random.normal(loc=mp_mean, scale=mp_sd, 
#                                size=(num_trials, len(mp_mean)))



class lowerBCL():
    
    # the constructors in python are called __init__
    # bcls is a list of the bcls which will be used during the simulation
    def __init__(self, bcls):
        self.bcls = bcls
    def hz_to_bcl(hz):
        return 1000/hz
    def __len__(self):
        return len(self.bcls)
    # this is the method that will be called to change the bcl
    # proto is the currently runing protocol (so this is will be many copies
    # of the proto we loaded from the settings)
    def __call__(self, proto):
        try:
            idx = self.bcls.index(int(proto.bcl))
        except:
            idx = -1
        idx += 1
        if idx < len(self.bcls):
            bcl = self.bcls[idx]
            proto.bcl = bcl
        else:
            proto.paceflag = False
            


bcls = [1600, 1200, 800, 500]
func = lowerBCL(bcls)

#proto.setRunDuring(func, firstRun=15_000, runEvery=20_000, numruns=len(func))
proto.bcl = 1000#1600

ina_name = 'Novel'
cell_params_names = [ina_name+'.'+mp_name for mp_name in model_param_names]

sub_mp_draws = np.random.multivariate_normal(mean=mp_mean, cov=mp_cov, size=(grid_shape[1], num_sims))
#np.tile(mp_mean, (grid_shape[1], num_sims, 1))#

protos = []
proto.numtrials = num_sims


for trial_num in range(num_sims):
    proto_cpy = proto.clone()
    proto_cpy.trial = trial_num
    
    for i in range(len(model_param_names)):
        pvar = proto_cpy.pvars.IonChanParam(proto_cpy.pvars.normal,
                                        val1 = mp_mean[i],
                                        val2 = mp_sd[i])
        proto_cpy.pvars[cell_params_names[i]] = pvar
        pvar = proto_cpy.pvars[cell_params_names[i]]
        cells_factors = pvar.cells
        for cell_loc in range(grid_shape[1]):
            cell = proto_cpy.grid[0,cell_loc].cell
            cell.setOption('Novel2020', True)
            cells_factors[(0, cell_loc)] = sub_mp_draws[cell_loc, trial_num, i]
        pvar.cells = cells_factors
    
    for cell_loc in range(grid_shape[1]):
        # pvar = proto_cpy.pvars.IonChanParam(proto_cpy.pvars.none,
        #                         val1 = 3.5,
        #                         val2 = 0)
        # proto_cpy.pvars["IkrFactor"] = pvar
        
        cell = proto_cpy.grid[0,cell_loc].cell
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
    
    #proto_cpy.tMax = 3000
    #proto_cpy.writetime = 0
    
    proto_cpy.tMax = 20_000#100000
    proto_cpy.writetime = 19_000#490000
    
    protos.append(proto_cpy)


sim_runner = pylqt.Misc.RunSim(protos)
sim_runner.run()

sleep(0.5)
settings.writeSettings(proto.datadir+"/simvars.xml", proto)

mp_draws = xarray.DataArray(sub_mp_draws, 
                            coords = {'fiber_length': range(grid_shape[1]),
                                      'num_sims': range(num_sims),
                                      'model_param_names': model_param_names},
                            dims = ['fiber_length', 'num_sims', 'model_param_names'])
mp_draws.to_netcdf(proto.datadir+"/factor_values.nc")

t0 = datetime.datetime.now()
taken_char = '\u2588'
left_char = '-'
term_len= 60
text = '\r[{}{}] {prog}% elsapsed {deltat}'
while sim_runner.finished() == False:
    prog = sim_runner.progressPercent()
    n_taken = int(prog/100 * term_len)
    n_left = term_len - n_taken
    taken = taken_char*n_taken
    left = left_char*n_left
    t1 = datetime.datetime.now()
    deltat = t1 - t0
    print(text.format(taken, left, prog=round(prog, 3),
                        deltat=str(deltat)), end='')
    sleep(0.5)
print()
input("Press return key to exit")



def calback(*args, **kwargs):
    sim_data = pylqt.Misc.DataReader.readDir(proto.datadir)
    dvdt = np.array([sim_data.meas[i].data[0][-1] for i in range(len(sim_data.meas))])
    plt.boxplot(dvdt)
#proto.runSim()


