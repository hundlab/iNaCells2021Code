#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:05:59 2020

@author: grat05
"""

import pymc3 as pm
import datetime
import numpy as np
import pickle
from threading import Timer
from multiprocessing import Pool
from functools import partial
import os

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

from atrial_model.parse_cmd_args import args
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_results, SimResults
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all, exp_parameters
from atrial_model.iNa.stat_model_3 import StatModel, key_frame
from atrial_model.iNa.model_setup import model_params_initial, mp_locs, sub_mps, model

def stop_sim(pymc_model):
    raise KeyboardInterrupt
    print("Sampling Canceled")
    
if __name__ == '__main__':
    
    atrial_model.run_sims_functions.plot1 = False #sim
    atrial_model.run_sims_functions.plot2 = False #diff
    atrial_model.run_sims_functions.plot3 = False #tau

    model_name = './mcmc_'
    model_name +=  args.model_name
    model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    model_name = model_name.format(cdate=datetime.datetime.now())
    meta_data_name = model_name
    model_name += '.pickle'
    db_path = args.out_dir+'/'+model_name
    
    model_db = {}
    model_db['model_params_initial'] = model_params_initial
    model_db['mp_locs'] = mp_locs
    model_db['param_bounds'] = model.param_bounds
    model_db['bio_model_name'] = args.model_name

    print("Running Pool with", os.cpu_count(), "processes")
    with Pool() as proc_pool:

        calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, data=datas,error_fill=0,\
                        pool=proc_pool)
        run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs)
        key_frame = key_frame(keys_all, exp_parameters)
        
        with StatModel(run_biophysical, key_frame, datas,
                                mp_locs, model) as stat_model:
            trace = None
            start = None
            if not args.previous_run is None:
                model_db['previous'] = {}
                model_db['previous']['file'] = args.previous_run
                model_db['previous']['is_trace'] = not args.previous_run_manual
                with open(args.previous_run,'rb') as file:
                    db = pickle.load(file)
                if args.previous_run_manual:
                    start = {}
                    if 'model_param' in db['start']:
                        start_vals = db['start']['model_param']
                        start['model_param'] = np.array([
                            start_vals[key] for key in key_frame.index])
                else:
                    trace = db['trace']
                del db

            trace = pm.sample(draws=5, tune=5, trace=trace, start=start,
                              cores=1, discard_tuned_samples=False)
    
            if not args.max_time is None:
                #max_time is in hours
                sample_timer = Timer(args.max_time*60*60, stop_sim, args=(stat_model,))
                sample_timer.start()
                
            if not args.max_time is None:
                sample_timer.join()
            
        model_db['num_calls'] = run_biophysical.call_counter
        model_db['key_frame'] = key_frame
        model_db['trace'] = trace
        with open(db_path, 'wb') as file:
            pickle.dump(model_db, file)
            
        print("Pickle File Written to:")
        print(model_name)
        #pymc.Matplot.plot(S)
