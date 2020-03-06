#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:37:25 2020

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

iNa_fit_functions.plot1 = True #sim
iNa_fit_functions.plot2 = True #diff
iNa_fit_functions.plot3 = True #tau

model = OHaraRudy_INa#Koval_ina##
dt = 0.05
monoExp_params = [-1,1,0]
biExp_params = np.array([-1,1,-1,1000,0])

exp_parameters, data = load_data_parameters('iNa_dims.xlsx','gating', data=all_data)


fits_results_joint = pickle.load(open('./fits_res_joint_ohara_0303.pkl','rb'))

sim_fs = []
datas = []

model_params = fits_results_joint['group'].x#np.ones(model.num_params)
mp_locs = np.arange(model.num_params)
sub_mps = model_params[mp_locs]


# inactivation normalized to no prepulse
keys_iin = [('7971163_4', 'Dataset 32ms'),\
            ('7971163_4', 'Dataset 512ms')]
process = partial(normalized2val, peakn=2)

model_params = np.ones(model.num_params)
mp_locs = np.arange(model.num_params)
sub_mps = model_params[mp_locs]

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

    baseline = sim_f_baseline(model_params)
    n_process = partial(process, val=baseline[0])
    sim_f.keywords['process'] = n_process

    sim_fs.append(sim_f)
    datas.append(key_data)


if __name__ == '__main__':
    np.seterr(all='ignore')
    with Pool(processes=20) as proc_pool:
        diff_fn = partial(calc_diff, model_parameters_full=model_params,\
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas, l=0,pool=proc_pool)

        diff_fn(model_params)
plt.show()