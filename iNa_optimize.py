#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:55:44 2020

@author: dgratz
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
from iNa_sims import sim_fs, datas, model_params_initial, mp_locs, model,\
    keys_all, exp_parameters, run_fits
import os
#import sys
#sys.path.append('./models/build/Debug/')
#import models

np.seterr(all='ignore')

iNa_fit_functions.plot1 = False #sim
iNa_fit_functions.plot2 = False #diff
iNa_fit_functions.plot3 = False #tau



try: fit_results
except NameError: fit_results = {}
try: fit_results_joint
except NameError: fit_results_joint = {}
try: fit_params
except NameError: fit_params = {}


if __name__ == '__main__':
    print("Running Pool with", os.cpu_count(), "processes")
    with Pool() as proc_pool:
        mp_locs = list(set(mp_locs))
        sub_mps = model_params_initial[mp_locs]
        sub_mp_bounds = np.array(model().param_bounds)[mp_locs]
        min_res = []
        all_res = []

        diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas,\
                            l=0,ssq=True,pool=proc_pool,\
                            results=all_res)
        minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False}}#"bounds": sub_mp_bounds,
        # res = optimize.basinhopping(diff_fn, sub_mps, \
        #                             minimizer_kwargs=minimizer_kwargs,\
        #                             niter=10, T=80,\
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
        res.fits = set(rfs for rfs in run_fits if run_fits[rfs])

        filename = 'fits_res_joint_ohara_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}.pkl'
        filename = filename.format(cdate=datetime.datetime.now())
        filepath = out_dir+'/'+filename
        with open(filepath, 'wb') as file:
            pickle.dump(res, file)

        #plot!
        iNa_fit_functions.plot1 = False #sim
        iNa_fit_functions.plot2 = True #diff
        iNa_fit_functions.plot3 = False #tau

        error = diff_fn(res.x, exp_params=exp_parameters, 
                        keys=[key for key in key_group for key_group in keys_all])
