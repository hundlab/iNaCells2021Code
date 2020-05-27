#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:43:47 2020

@author: dgratz
"""

import pymc
import datetime
import numpy as np
import pickle
from threading import Timer

from parse_cmd_args import args
import iNa_fit_functions
from iNa_fit_functions import calc_results, SimResults
from multiprocessing import Pool
from functools import partial
import os

from iNa_sims import sim_fs, datas, keys_all, exp_parameters
from stat_model import make_model
from iNa_model_setup import model_params_initial, mp_locs, sub_mps, model

#from './optimize_Koval_0423_0326.pkl'
# model_params_initial[mp_locs] = np.array(
#     [-0.80680888,  0.63417512, -0.69291108, -2.04633128, -0.01836348,
#         0.35378153,  0.64030428, -0.8010144 ,  0.72033717, -1.68578422,
#         5.87859494, -1.00653083, -1.67532066,  0.84144004,  0.88200433,
#        -2.70056045, -2.26745786,  2.2395883 , -0.48703343])

#previous_run = './mcmc_Koval_0511_1609_2.pickle'

class ObjContainer():
    pass

def stop_sim(pymc_model):
    pymc_model.pause()
    pymc_model.save_state()
    pymc_model.tally()
    pymc_model.halt()
    print("Sampling Canceled")
    
if __name__ == '__main__':
    
    iNa_fit_functions.plot1 = False #sim
    iNa_fit_functions.plot2 = False #diff
    iNa_fit_functions.plot3 = False #tau

    model_name = './mcmc_'
    model_name +=  args.model_name
    model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    model_name = model_name.format(cdate=datetime.datetime.now())
    meta_data_name = model_name
    model_name += '.pickle'
    db_path = args.out_dir+'/'+model_name
    
    meta_data_name += '_metadata.pickle'
    meta_data_path = args.out_dir+'/'+meta_data_name
    
    model_metadata = ObjContainer()
    model_metadata.model_params_initial = model_params_initial
    model_metadata.mp_locs = mp_locs
    model_metadata.keys_all = keys_all
    model_metadata.param_bounds = model.param_bounds
    model_metadata.bio_model_name = args.model_name

    print("Running Pool with", os.cpu_count(), "processes")
    with Pool() as proc_pool:
        
        calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, data=datas,error_fill=0,\
                        pool=proc_pool)
        run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs)

        
        made_model = make_model(run_biophysical, keys_all, datas, 
                                model_params_initial, mp_locs, model, 
                                exp_parameters['temp ( K )'])
        if args.previous_run is None:
            db = 'pickle'
        else:
            db = pymc.database.pickle.load(args.previous_run)
        pymc_model = pymc.MCMC(made_model, db=db, dbname=db_path)

        if not args.max_time is None:
            #max_time is in hours
            sample_timer = Timer(args.max_time*60*60, stop_sim, args=(pymc_model,))
            sample_timer.start()
            
        pymc_model.sample(iter=10000, burn=0, thin=1, tune_throughout=True, save_interval=100)#, burn_till_tuned=True)
        pymc_model.db.close()
        
        if not args.max_time is None:
            sample_timer.join()
        
        model_metadata.num_calls = run_biophysical.call_counter
        model_metadata.trace_pickel_file = model_name
        with open(meta_data_path, 'wb') as file:
            pickle.dump(model_metadata, file)
            
        print("Pickle File Written to:")
        print(model_name)
        #pymc.Matplot.plot(S)
