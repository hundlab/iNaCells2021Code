#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:40:40 2020

@author: grat05
"""


import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

from atrial_model.parse_cmd_args import args
args.normalize_all = True
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_diff
from atrial_model.optimization_functions import lstsq_wrap, save_results
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all, exp_parameters
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, dt, run_fits,\
    model_params_initial, run_fits


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

atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau


#

def calc_params_and_run(betas, mp_locs, temperature, **kwargs):
    intercept = betas[:len(mp_locs)]
    b_temp = betas[len(mp_locs):]
    model_params = {key:
                    intercept + temp*b_temp
                    for key, temp in temperature.items()}
    
    return calc_diff(model_params, mp_locs=mp_locs, **kwargs)
    

if __name__ == '__main__':
    with Pool() as proc_pool:
        mp_locs = list(set(mp_locs))
        sub_mps = model_params_initial[mp_locs]
        sub_mp_bounds = np.array(model().param_bounds)[mp_locs]
        temp_b_bounds = np.ones_like(sub_mp_bounds)*np.array([-1,1])
        betas_bounds = np.concatenate((sub_mp_bounds, temp_b_bounds), axis=0)
        min_res = []
        all_res = []

        # accept_test=partial(check_bounds, bounds=sub_mp_bounds))
#            minimizer_kwargs = {"method": "BFGS", "options": {"maxiter":100}}

        
        diff_fn = partial(calc_params_and_run, temperature=exp_parameters['temp ( K )'],
                          model_parameters_full=model_params_initial,
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas,
                            pool=proc_pool,ssq=True,
                            results=all_res)
        minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False}}#"bounds": sub_mp_bounds,
        # res = optimize.basinhopping(diff_fn, sub_mps, \
        #                             minimizer_kwargs=minimizer_kwargs,\
        #                             niter=10, T=80,\
        #                             callback=partial(save_results, results=min_res),\
        #                             stepsize=1)#T=80
        res = optimize.dual_annealing(diff_fn, bounds=betas_bounds,
                                          no_local_search=True,
                                          local_search_options=minimizer_kwargs,
                                          maxiter=100,maxfun=6000)
#        res = optimize.least_squares(diff_fn, sub_mps, \
#                        bounds=np.array(model().param_bounds)[mp_locs].T)
        res.keys_all = keys_all
        res.all_res = all_res
        res.min_res = min_res
        res.fits = set(rfs for rfs in run_fits if run_fits[rfs])
        res.mp_locs = mp_locs
        res.model_name = args.model_name
        res.intersept = res.x[:len(mp_locs)]
        res.b_temp = res.x[len(mp_locs):]

        filename = 'optimize_temp_'+args.model_name+'_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}.pickle'
        filename = filename.format(cdate=datetime.datetime.now())
        filepath = args.out_dir+'/'+filename
        with open(filepath, 'wb') as file:
            pickle.dump(res, file)
        print("Pickle File Written to:")
        print(filepath)

        # #plot!
        # atrial_model.run_sims_functions.plot1 = False #sim
        # atrial_model.run_sims_functions.plot2 = True #diff
        # atrial_model.run_sims_functions.plot3 = False #tau

        # error = diff_fn(res.x, exp_params=exp_parameters, 
        #                 keys=[key for key in key_group for key_group in keys_all])
