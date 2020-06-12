#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:47:10 2020

@author: grat05
"""

import sys
sys.path.append('../../../')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


plot_trace = True
plot_sim = False
plot_pymc_diag = False

atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau

burn_till = 5000
chain = 0
#burn_till = 60000

if __name__ == '__main__':
    class ObjContainer():
        pass
    filename = 'mcmc_OHaraRudy_wMark_INa_0606_0047'
    #filename = 'mcmc_Koval_0601_1835'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0603_1051'
    #filename = 'mcmc_OHara_0528_1805'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0528_1833'
    #filename = 'mcmc_Koval_0526_1728'
#    filename = 'mcmc_Koval_0519_1830'

    filepath = atrial_model.fit_data_dir+'/'+filename
    with open(filepath+'.pickle','rb') as file:
        db = pickle.load(file)
    if db['_state_']['sampler']['status'] == 'paused':
        current_iter = db['_state_']['sampler']['_current_iter']
        for key in db.keys():
            if key != '_state_':
                db[key][chain] = db[key][chain][:current_iter]
    with open(filepath+'_metadata.pickle','rb') as file:
        model_metadata = pickle.load(file)
    group_names = []
    for key_group in model_metadata.keys_all:
        group_names.append(exp_parameters.loc[key_group[0], 'Sim Group'])
    bounds = np.array(model_metadata.param_bounds)[model_metadata.mp_locs, :]
        
    if plot_trace:
        
        trace = 'deviance'
        trace_data = db[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        ax[0].plot(trace_data)
        ax[1].hist(trace_data, orientation='horizontal')
        
        trace = 'model_param_mean'
        trace_data = db[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        for i in range(trace_data.shape[1]):
            ax[0].plot(trace_data[:,i], label=model_param_names[i])
            _,_,hist = ax[1].hist(trace_data[:,i], orientation='horizontal', label=model_param_names[i])
#            color = hist[0].get_facecolor()
#            ax[0].axhline(bounds[i, 0]+i/100, c=color, label=model_param_names[i]+'_lower')
#            ax[0].axhline(bounds[i, 1]+i/100, c=color, label=model_param_names[i]+'_upper')
        ax[1].legend(frameon=False)
        
            
        trace = 'model_param_tau'
        trace_data = db[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        for i in range(trace_data.shape[1]):
            ax[0].plot(1/np.sqrt(trace_data[:,i]), label=model_param_names[i])
            ax[1].hist(1/np.sqrt(trace_data[:,i]), orientation='horizontal', label=model_param_names[i])
        ax[1].legend(frameon=False)
        
            
        trace = 'b_temp'
        trace_data = db[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        for i in range(trace_data.shape[1]):
            ax[0].plot(trace_data[:,i], label=model_param_names[i])
            ax[1].hist(trace_data[:,i], orientation='horizontal', label=model_param_names[i])
        ax[1].legend(frameon=False)
        
        trace = 'error_tau'
        trace_data = db[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        for i in range(trace_data.shape[1]):
            ax[0].plot(1/np.sqrt(trace_data[:,i]), label=group_names[i])
            ax[1].hist(1/np.sqrt(trace_data[:,i]), orientation='horizontal', label=group_names[i])
        ax[1].legend(frameon=False)
        

        
        trace = 'model_param'
        trace_data = db[trace][chain]
        for param_i in range(trace_data.shape[2]):
            fig = plt.figure(model_param_names[param_i])
            ax = [fig.add_subplot()]
    #        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
            for sim_i in range(trace_data.shape[1]):
                ax[0].plot(trace_data[:,sim_i,param_i])#, label=model_param_names[sim_i])
    #            ax[1].hist(1/np.sqrt(trace_data[:,i]), orientation='horizontal', label=group_names[i])
    #        ax[0].legend(frameon=False)
        
        # trace = 'biophys_res'
        # trace_data = db[trace][0]
        # fig = plt.figure(trace)
        # ax = [fig.add_subplot(1,2,1)]
        # ax[0].plot(trace_data)
        # ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        # for i in range(trace_data.shape[1]):
        #     ax[1].hist(trace_data[:,i], orientation='horizontal')
        
        
    if plot_sim:
        with Pool() as proc_pool:
    #        proc_pool = None
            b_temp = np.median(db['b_temp'][0][burn_till:], axis=0)
            intercept = np.median(db['model_param_mean'][0][burn_till:], axis=0)
            num_sims = sum(map(len, model_metadata.keys_all))
            model_params = {}
            k = 0
            for i, keys in enumerate(model_metadata.keys_all):
                for j, key in enumerate(keys):
                    #temperature adjusted to minimum in group
                    temperature = exp_parameters.loc[key, 'temp ( K )'] -290.15
                    b_temp_eff = b_temp * temperature
                    sub_mps = intercept + b_temp_eff
                    model_params[key] = sub_mps
                    k += 1
    
            res_overall = calc_results(model_params, sim_funcs=sim_fs,\
                                      model_parameters_full=model_params_initial,\
                            mp_locs=mp_locs, data=datas,error_fill=0,\
                            pool=proc_pool)
                
            model_param_mean = np.mean(db['model_param'][0], axis=0)
            model_param_sim_mean = {}
            k = 0
            for key_group in model_metadata.keys_all:
                for key in key_group:
                    model_param_sim_mean[key] = model_param_mean[k]
                    k += 1
            res_indiv = calc_results(model_param_sim_mean, sim_funcs=sim_fs,\
                                      model_parameters_full=model_params_initial,\
                            mp_locs=mp_locs, data=datas,error_fill=0,\
                            pool=proc_pool)
        
        for key_group in keys_all:
            for key in key_group:
                figname = exp_parameters.loc[key, 'Sim Group']
                figname = figname if not pd.isna(figname) else 'Missing Label'
                plt.figure(figname + " overall fit")
                plt.plot(datas[key][:,0], res_overall[key], label=key)
                plt.scatter(datas[key][:,0], datas[key][:,1])
                plt.legend()
                
        for key_group in keys_all:
            for key in key_group:
                figname = exp_parameters.loc[key, 'Sim Group']
                figname = figname if not pd.isna(figname) else 'Missing Label'
                plt.figure(figname + " individual fit")
                plt.plot(datas[key][:,0], res_indiv[key], label=key)
                plt.scatter(datas[key][:,0], datas[key][:,1])
                plt.legend()
    
    
    if plot_pymc_diag:
        try:
            import pymc
            from pymc import Matplot
            from pymc import diagnostics
#            db = pymc.database.pickle.load(trace_file)
            geweke_scores = diagnostics.geweke(db['model_param_tau'][0][:,2])
            Matplot.geweke_plot(geweke_scores, name="Gweke Scores")
            
        except ImportError:
            print("pymc not installed")
    