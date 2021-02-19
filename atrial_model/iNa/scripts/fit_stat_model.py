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
#from atrial_model.iNa.stat_model import make_model
from atrial_model.iNa.stat_model import make_model
from atrial_model.iNa.model_setup import model_params_initial, mp_locs, sub_mps, model
from atrial_model.pymc2step import AdaptiveSDMetropolis

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
    pymc_model.halt()
    #pymc_model.tally()
    pymc_model.save_state()
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
        run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs, disp_print=False)

        start = {}
        if args.previous_run_manual and not args.previous_run is None:
            with open(args.previous_run,'rb') as file:
                    old_db = pickle.load(file)
            start = old_db['MAP']

        made_model = make_model(run_biophysical, keys_all, datas, 
                                mp_locs, model, start=start, 
                                temperatures=exp_parameters['temp ( K )'])
        db = 'pickle'
        if not args.previous_run is None and not args.previous_run_manual:
            db = pymc.database.pickle.load(args.previous_run)
        # elif not args.previous_run is None:
        #         with open(args.previous_run,'rb') as file:
        #             old_db = pickle.load(file)
        #         prev_state = old_db['_state_']['stochastics']
        #         for var in made_model:
        #             if var.__name__ == 'model_param':
        #                 var.value = prev_state[var.__name__]
        #         del old_db
        pymc_model = pymc.MCMC(made_model, db=db, dbname=db_path)
        
        # nodes = [pymc_model.get_node(node_name) 
        #          for node_name in ['model_param_mean', 'b_temp']]
        # cov = np.load(args.out_dir+'/model_mean.npy')
        # pymc_model.use_step_method(pymc.AdaptiveMetropolis, nodes,
        #                                shrink_if_necessary=True,
        #                                cov=cov,
        #                                delay=500, interval=400)
        
        
        # node = pymc_model.get_node('model_param_mean')
        # sd = np.load(args.out_dir+'/model_param_mean.npy')
        # pymc_model.use_step_method(pymc.Metropolis, node,
        #                      proposal_sd=sd)
        
        # node = pymc_model.get_node('b_temp')
        # sd = np.load(args.out_dir+'/b_temp.npy')
        # pymc_model.use_step_method(pymc.Metropolis, node,
        #                      proposal_sd=sd)


        node = pymc_model.get_node('model_param')
        sd = np.load(args.out_dir+'/model_param.npy')#0.01*np.ones(node.value.shape)
        pymc_model.use_step_method(AdaptiveSDMetropolis, node,
                              proposal_sd=sd,
                              delay=100, interval=200)

        
        # node = pymc_model.get_node('model_param_tau')
        # sd = np.load(args.out_dir+'/model_param_tau.npy')
        # pymc_model.use_step_method(pymc.Metropolis, node,
        #                            proposal_sd=sd)
        
        # node = pymc_model.get_node('error_tau')
        # sd = np.load(args.out_dir+'/error_tau.npy')
        # pymc_model.use_step_method(pymc.Metropolis, node,
        #                            proposal_sd=sd)
        
        
        
        # adaptive_nodes = ['model_param', 'b_temp','model_param_mean']
        # node_scales = [0.0187618294, 0.00531441, 0.0215233]
        
        # for i, node_name in enumerate(adaptive_nodes):
        #     node = pymc_model.get_node(node_name)
        #     scale = node_scales[i]*np.ones(node.value.size)
        #     pymc_model.use_step_method(pymc.AdaptiveMetropolis, node,
        #                                shrink_if_necessary=True,
        #                                scales={node: scale})

        if not args.max_time is None:
            #max_time is in hours
            sample_timer = Timer(args.max_time*60*60, stop_sim, args=(pymc_model,))
            sample_timer.start()
        
        pymc_model.sample(iter=100_000, burn=0, thin=1, tune_throughout=True, 
                          save_interval=100, tune_interval=600)#, burn_till_tuned=True)
        pymc_model.db.close()
        
        if not args.max_time is None:
            sample_timer.cancel()
            sample_timer.join()
        
        model_metadata.num_calls = run_biophysical.call_counter
        model_metadata.trace_pickel_file = model_name
        with open(meta_data_path, 'wb') as file:
            pickle.dump(model_metadata, file)
            
        print("Pickle File Written to:")
        print(model_name)
        #pymc.Matplot.plot(S)
