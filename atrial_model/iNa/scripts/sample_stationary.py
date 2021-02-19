#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:23:22 2020

@author: grat05
"""

import pymc
import datetime
import numpy as np
import pickle
from threading import Timer
from multiprocessing import Pool
from functools import partial
import os

from scipy.optimize._numdiff import approx_derivative

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

from atrial_model.parse_cmd_args import args
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_results, SimResults
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all, exp_parameters
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

   
if __name__ == '__main__':
    
    atrial_model.run_sims_functions.plot1 = False #sim
    atrial_model.run_sims_functions.plot2 = False #diff
    atrial_model.run_sims_functions.plot3 = False #tau

    model_name = './sample_iid_'
    model_name +=  args.model_name
    model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    model_name = model_name.format(cdate=datetime.datetime.now())
    meta_data_name = model_name
    model_name += '.pickle'
    model_path = args.out_dir+'/'+model_name
    
    
    model_metadata = {}
#    model_metadata.start = model_params_initial
    model_metadata['mp_locs'] = mp_locs
    model_metadata['keys_all'] = keys_all
    model_metadata['bio_model_name'] = args.model_name

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
        pymc_model = pymc.MCMC(made_model, db=db)
        
        nodes = [pymc_model.get_node(node_name) 
                 for node_name in ['model_param_mean', 'b_temp']]
        cov = np.load(args.out_dir+'/model_mean.npy')
        pymc_model.use_step_method(pymc.AdaptiveMetropolis, nodes,
                                       shrink_if_necessary=True,
                                       cov=cov,
                                       delay=500, interval=400)
        
        # node = pymc_model.get_node('model_param')
        # scale = 0.001*np.ones(node.value.size)
        # pymc_model.use_step_method(pymc.AdaptiveMetropolis, node,
        #                              shrink_if_necessary=True,
        #                              scales={node: scale},
        #                              delay=500, interval=1400)
        node = pymc_model.get_node('model_param')
        sd = 0.01*np.ones(node.value.shape)
        pymc_model.use_step_method(AdaptiveSDMetropolis, node,
                             proposal_sd=sd,
                             delay=100, interval=200)

        
        node = pymc_model.get_node('model_param_tau')
        sd = np.load(args.out_dir+'/model_param_tau.npy')
        pymc_model.use_step_method(pymc.Metropolis, node,
                                   proposal_sd=sd)
        
        node = pymc_model.get_node('error_tau')
        sd = np.load(args.out_dir+'/error_tau.npy')
        pymc_model.use_step_method(pymc.Metropolis, node,
                                   proposal_sd=sd)
        
        # adaptive_nodes = ['model_param', 'b_temp','model_param_mean']
        # node_scales = [0.0187618294, 0.00531441, 0.0215233]
        
        # for i, node_name in enumerate(adaptive_nodes):
        #     node = pymc_model.get_node(node_name)
        #     scale = node_scales[i]*np.ones(node.value.size)
        #     pymc_model.use_step_method(pymc.AdaptiveMetropolis, node,
        #                                shrink_if_necessary=True,
        #                                scales={node: scale})

        
        sample_timer = Timer(100, lambda pymc_model: pymc_model.pause(), args=(pymc_model,))
        sample_timer.start()
        pymc_model.sample(iter=10, burn=10, thin=1)
        sample_timer.join()
        
        node = pymc_model.get_node('model_param')
        for steper in pymc_model.step_methods:
            if node in steper.stochastics:
                break
        find_jac = True
        sample_stationary = False
        
        if find_jac:
            print("begin jac estimation")
            start_val = node.value
            val_shape = start_val.shape
            def eval_fun(val, start_val, i):
                new_single = start_val.copy()
                new_single[i] = val
                steper.stochastic.value = new_single
                log_p = steper.logp_plus_loglike
                steper.reject()
                return log_p
                        
            all_jacs = []
            for i in range(start_val.shape[0]):
                print(i)
                fun = lambda x: eval_fun(x, start_val, i)
                jac = approx_derivative(fun, start_val[i,:])
                all_jacs.append(jac)
                
            
            model_metadata['jac'] = np.array(all_jacs)
        
        
        if sample_stationary:
            mean_params = node.value
            mean = steper.logp_plus_loglike
            num_samples = 1000
            samples = []
            log_p = []
            log_p_sim = []
            print("begin sampling")
            for sample_n in range(num_samples):
                steper.propose()
                sample = node.value
                samples.append(sample)
                log_p.append(steper.logp_plus_loglike)
                steper.reject()
                sub_lp = []
                for i in range(sample.shape[0]):
                    new_single = mean_params.copy()
                    new_single[i] = sample[i]
                    steper.stochastic.value = new_single
                    logp_sub = steper.logp_plus_loglike
                    sub_lp.append(logp_sub)
                    steper.reject()
                log_p_sim.append(np.array(sub_lp))
                print(sample_n)
        
            model_metadata['mean'] = mean
            model_metadata['mean_params'] = mean_params
            model_metadata['samples'] = np.array(samples)
            model_metadata['log_p'] = np.array(log_p)
            model_metadata['log_p_sim'] = np.array(log_p_sim)
        
        pymc_model.halt()
        pymc_model.save_state()
        pymc_model.db.close()
        
        model_name
        
        with open(model_path, 'wb') as file:
            pickle.dump(model_metadata, file)
            
        print("Pickle File Written to:")
        print(model_name)
        #pymc.Matplot.plot(S)
