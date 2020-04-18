#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 08:13:21 2020

@author: grat05
"""


#from iNa_models import Koval_ina, OHaraRudy_INa
from iNa_models_ode import OHaraRudy_INa
from scripts import load_data_parameters, load_all_data, all_data, out_dir
import iNa_fit_functions
from iNa_fit_functions import normalize2prepulse, setup_sim, run_sim, \
calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp, biExp,\
calcExpTauAct
from optimization_functions import lstsq_wrap, save_results


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
from iNa_sims import sim_fs, datas, keys_all, exp_parameters
from iNa_model_setup import model, sub_mps, sub_mp_bounds, dt, run_fits,\
    model_params_initial, run_fits, model_name


#full optimization
res = pickle.load(open('./optimize_ohara_0417_0041.pkl','rb'))
#partial
#res = pickle.load(open('./fits_res_joint_ohara_0416_1913.pkl','rb'))
mp_locs = res.mp_locs
res.x = np.zeros_like(res.x)


keys_keep = set(key for keys in res.keys_all for key in keys)
#keys_keep = [('7971163_6', 'Dataset -95')]
sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_keep}
datas = {key: data for key, data in datas.items() if key in keys_keep}



iNa_fit_functions.plot1 = False #sim
iNa_fit_functions.plot2 = True #diff
iNa_fit_functions.plot3 = False #tau

if __name__ == '__main__':
    with Pool() as proc_pool:
#        proc_pool = None
        diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas,\
                            l=0,pool=proc_pool,ssq=True)
                    
        error = diff_fn(res.x, exp_params=exp_parameters, 
                        keys=[key for key_group in keys_all for key in key_group])
        print(error)