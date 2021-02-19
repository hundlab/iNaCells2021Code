#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:48:26 2020

@author: grat05
"""


import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


import sys
sys.path.append('../../../')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az

import atrial_model
from atrial_model.iNa.define_sims import exp_parameters
from atrial_model.iNa.model_setup import model_param_names
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_results
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all
from atrial_model.iNa.model_setup import model_params_initial, mp_locs, sub_mps, model


from multiprocessing import Pool
from functools import partial
import os


plot_trace = False
plot_sim = True
plot_regressions = False
plot_pymc_diag = False


chain = 0#5
#burn_till = 60000
stack = True

class ObjContainer():
    pass
#filename = 'mcmc_OHaraRudy_wMark_INa_0824_1344'

#filename = 'mcmc_OHaraRudy_wMark_INa_0919_1832_sc'
filename = 'mcmc_OHaraRudy_wMark_INa_0924_1205'
#filename = 'mcmc_OHaraRudy_wMark_INa_0831_1043_sc'
#filename = 'mcmc_OHaraRudy_wMark_INa_0829_1748'
#filename = 'mcmc_OHaraRudy_wMark_INa_0829_1334'
#filename = 'mcmc_OHaraRudy_wMark_INa_0827_1055'
#filename = 'mcmc_OHaraRudy_wMark_INa_0826_0958'
#filename = 'mcmc_OHaraRudy_wMark_INa_0821_1132'
#filename = 'mcmc_OHaraRudy_wMark_INa_0702_1656'

filename = 'mcmc_OHaraRudy_wMark_INa_1012_1149'


#filename = 'mcmc_OHaraRudy_wMark_INa_0627_1152'
#filename = 'mcmc_OHaraRudy_wMark_INa_0626_0808'
#filename = 'mcmc_OHaraRudy_wMark_INa_0606_0047'
#filename = 'mcmc_Koval_0601_1835'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0603_1051'
#filename = 'mcmc_OHara_0528_1805'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0528_1833'
#filename = 'mcmc_Koval_0526_1728'
#filename = 'mcmc_Koval_0519_1830'

base_dir = atrial_model.fit_data_dir+'/'
with open(base_dir+'/'+filename+'_metadata.pickle','rb') as file:
    model_metadata = pickle.load(file)
with open(base_dir+model_metadata.trace_pickel_file,'rb') as file:
    db = pickle.load(file)
if db['_state_']['sampler']['status'] == 'paused':
    current_iter = db['_state_']['sampler']['_current_iter']
    current_iter -= db['_state_']['sampler']['_burn']
    for key in db.keys():
        if key != '_state_':
            db[key][chain] = db[key][chain][:current_iter]
if stack:
    for key in db.keys():
        if key != '_state_' and key != 'AdaptiveSDMetropolis_model_param_adaptive_scale_factor':
            stacked = [db[key][chain] for chain in db[key]]
            db[key] = [np.concatenate(stacked)]
  
group_names = []
sim_groups = []
sim_names = []
for key_group in model_metadata.keys_all:
    group_names.append(exp_parameters.loc[key_group[0], 'Sim Group'])
    for key in key_group:
        sim_names.append(key)
        sim_groups.append(group_names[-1])
        
biophys_res = {}
pos = 0
for key in sim_names:
    end_pos = pos + datas[key].shape[0]
    biophys_res[key] = db['biophys_res'][0][:,pos:end_pos]
    pos = end_pos

def createMovie(key, fps=1,moviename="test.mp4",metadata=None, ff_path = os.path.join('C:/', 'Users', 'grat05', 'AppData', 'Local', 'Programs', 'ffmpeg', 'bin', 'ffmpeg.exe')):
    
    n_frames = biophys_res[key].shape[0]
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(datas[key][:,0], datas[key][:,1])
    line, = ax.plot(datas[key][:,0], biophys_res[key][0])
    
    plt.rcParams['animation.ffmpeg_path'] = ff_path
    if ff_path not in sys.path: sys.path.append(ff_path)
    writer = animation.FFMpegWriter(fps=fps,metadata=metadata)
   
    with writer.saving(fig, str(moviename), dpi=100):
        for i in range(1,n_frames,10):
            line.set_ydata(biophys_res[key][i])
            fig.canvas.draw()
            writer.grab_frame()
