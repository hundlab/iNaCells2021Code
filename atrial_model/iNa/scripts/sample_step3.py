#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:37:19 2020

@author: grat05
"""


from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle
import copy
import os
import datetime
from scipy import stats

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

import pymc3 as pm
import arviz as az
import datetime
import numpy as np
import pickle
from threading import Timer
from multiprocessing import Manager
from functools import partial
import os
import warnings

#from iNa_models import Koval_ina, OHaraRudy_INa
import atrial_model
from atrial_model.iNa.models import OHaraRudy_INa, Koval_ina
import atrial_model.run_sims_functions
from atrial_model.run_sims_functions import peakCurr, normalized2val, calcExpTauInact, monoExp,\
calcExpTauAct, triExp, biExp
from atrial_model.run_sims import calc_diff, calc_results
from atrial_model.iNa.define_sims import sim_fs, datas,\
    keys_all, exp_parameters, data
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial, model_param_names
from atrial_model.iNa.stat_model_3 import key_frame
from atrial_model.parse_cmd_args import args


import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_results, SimResults
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all, exp_parameters
from atrial_model.iNa.stat_model_3 import StatModel, key_frame
from atrial_model.iNa.model_setup import model_params_initial, mp_locs, sub_mps, model

warnings.simplefilter(action='ignore', category=FutureWarning)

#import pickle
#res = pickle.load(open(atrial_model.fit_data_dir+'/optimize_Koval_0423_0326.pkl','rb'))
#mp_locs = res.mp_locs
#sub_mps = res.x
keys_all_list = [key for key_grp in keys_all for key in key_grp ]
sim_groups = dict(exp_parameters.loc[keys_all_list, 'Sim Group'])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = True #diff
atrial_model.run_sims_functions.plot3 = False #tau

if __name__ == '__main__':
       

    filename = 'mcmc_OHaraRudy_wMark_INa_1210_1420'
    

    with Manager() as manager:
#    print("Running Pool with", os.cpu_count(), "processes")
        with manager.Pool() as proc_pool:
    #        proc_pool = None
            
   
            calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                            mp_locs=mp_locs, data=datas,error_fill=0,\
                            pool=proc_pool)
            run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs, disp_print=False)
            key_frame = key_frame(keys_all, exp_parameters)
            
            with StatModel(run_biophysical, key_frame, datas,
                                    mp_locs, model) as stat_model:

            
                base_dir = atrial_model.fit_data_dir+'/'
                # with open(base_dir+'/'+filename+'_trace.pickle','rb') as file:
                #     db_full = pickle.load(file)
                # db = db_full['trace']
                # trace = db[-1]
                
                with open(base_dir+'/'+filename+'.pickle','rb') as file:
                    db_full = pickle.load(file)
                db = db_full['trace']
                trace = db.warmup_posterior.sel(
                    {'draw': db.warmup_posterior.indexes['draw'][-1]})
                trace =  {key: np.squeeze(np.array(val)) for key, val in trace.items()}
                trace2 = trace.copy()
                for key in  trace:
                    var = stat_model[key]
                    if hasattr(var, 'transformation'):
                        transformed_name = var.name+'_'+\
                            var.transformation.name+'__'
                        transformed_val = var.transformation.forward_val(trace[key])
                        trace2[transformed_name] = transformed_val
                trace = trace2
                
                # #starting model params
                # with open(os.path.dirname(__file__)+'/sub_mps_sigma_est.pickle','rb') as file:
                #     sub_mps_base = pickle.load(file)
                    
                group_names = key_frame['Sim Group'].unique()
            #    sim_groups = []
                sim_names = key_frame.index

            #            sim_groups.append(group_names[-1])
            
                inital_step = {key: 
                               np.sqrt(np.diag(
                                   db_full['proposal_cov']['mp '+str(key)]))
                               for key in sim_names}
                    
                sub_mps_base = dict(zip(key_frame.index, trace['model_param']))
                error_sigma = dict(zip(group_names, trace['error_sd']))
                
                def log_lik_from_data(key, dat):
                    log_liks = stats.norm.logpdf(datas[key[0:2]][:,1],
                                           loc=dat, 
                                           scale=error_sigma[sim_groups[key[0:2]]])
                    return np.sum(log_liks)

                log_p_funcs = {sim_name: stat_model['mp '+str(sim_name)].logp 
                               for sim_name in sim_names}
                def log_lik_from_prior(key, sub_mp_vals):
                    save = trace['mp '+str(key[:2])]
                    trace['mp '+str(key[:2])] = sub_mp_vals
                    logp = log_p_funcs[key[:2]](trace)
                    trace['mp '+str(key[:2])] = save
                    return logp

                step_all = {}
                sub_mps_all = {}
                for key in keys_all_list:
                    for i in range(len(model_param_names)):
                        key_up = (key[0],key[1],i,"up")
                        key_down = (key[0],key[1],i,"down")
                        step_all[key_up] = inital_step[key][i]
                        step_all[key_down] = -inital_step[key][i]
                        
                        base = sub_mps_base[key].copy()
                        base[i] += step_all[key_up]
                        sub_mps_all[key_up] = base
                        base = sub_mps_base[key].copy()
                        base[i] += step_all[key_down]
                        sub_mps_all[key_down] = base




                diff_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                                 mp_locs=mp_locs, sim_funcs=sim_fs, data=datas,\
                                 pool=proc_pool)
                
                start = {}
                res = diff_fn(sub_mps_base, exp_params=exp_parameters)
                for key, dat in res.items():
                    llik_data = log_lik_from_data(key, dat)
                    llik_prior = log_lik_from_prior(key, sub_mps_base[key])
                    start[key] = llik_data+llik_prior
                
                sim_fs_all = {key: sim_fs[key[:2]] for key in sub_mps_all}
                diff_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                         mp_locs=mp_locs, sim_funcs=sim_fs_all, data=datas,\
                         pool=proc_pool)
                    
                n_iter = 20
                results = []
                logp_vals = []
                steps = []
        
                for it in range(n_iter):
                    print(it, end=' ', flush=True)
                    res = diff_fn(sub_mps_all, exp_params=exp_parameters)
                    results.append(copy.deepcopy(res))
                    steps.append(copy.deepcopy(step_all))
                    logp_vals.append({})
                    for key, dat in res.items():
                        log_lik_data = log_lik_from_data(key, dat)
                        log_lik_prior = log_lik_from_prior(key, sub_mps_all[key])
                        log_lik = log_lik_data + log_lik_prior
                        change = abs(start[key[0:2]] - log_lik)
                        new_step = step_all[key]*np.double(change)**-1
                        new_step = max(min(new_step, 1.5), -1.5)
                        if np.isinf(log_lik):
                            new_step = step_all[key]/10
                        else:
                            new_step = new_step*0.5 + step_all[key]*0.5
                        step_all[key] = new_step
                        base = sub_mps_base[key[0:2]].copy()
                        base[key[2]] += new_step
                        sub_mps_all[key] = base
                        logp_vals[it][key] = log_lik, change
        
        
 
                data = {}
                data['results'] = results
                data['logp'] = logp_vals
                data['steps'] = steps
                
            
            # sub_mps_list = [sub_mps]
            # for i in range(len(mp_locs)):
            #     mod_sum_mps = copy.deepcopy(sub_mps)
            #     for key in mod_sum_mps:
            #         mod_sum_mps[key][i] += 0.1
            #     sub_mps_list.append(mod_sum_mps)
            #     mod_sum_mps = copy.deepcopy(sub_mps)
            #     for key in mod_sum_mps:
            #         mod_sum_mps[key][i] -= 0.1
            #     sub_mps_list.append(mod_sum_mps)
            
        
            # with Pool() as proc_pool:
            #     #proc_pool = None
            #     diff_fn = partial(calc_results, model_parameters_full=model_params_initial,\
            #                     mp_locs=mp_locs, sim_funcs=sim_fs, data=datas,\
            #                        pool=proc_pool)
            #     #diff_fn(sub_mps, exp_params=exp_parameters, 
            #                     #keys=keys_keep)
            #     res_list = []
            #     for i,sub_mps in enumerate(sub_mps_list):
            #         print(i)
            #         res = diff_fn(sub_mps, exp_params=exp_parameters)
            #         res_list.append(res)
            
                try:
                    proposal_sd = np.empty((len(sim_names), len(sub_mps)))
                    for i in range(len(sub_mps)):
                        for j,key in enumerate(sim_names):
                            up = steps[-1][(key[0],key[1], i, 'up')]
                            down = steps[-1][(key[0],key[1], i, 'down')]
                            proposal_sd[j,i] = min(up, abs(down))
                    
                    proposal_cov = {name: np.diag(proposal_sd[i]**2) 
                                    for i, name in enumerate(sim_names)}
                    proposal_cov_str = {'mp '+str(key): value 
                                        for key, value in proposal_cov.items()}
                    data['proposal_sd'] = proposal_sd
                    data['proposal_cov'] = proposal_cov
                    data['proposal_cov_str'] = proposal_cov_str
                except Exception as e:
                    print("proposal_sd calculation failed")
                    print(e)
         
                
                
                model_name = './sample_step_'
                model_name +=  args.model_name
                model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
                model_name = model_name.format(cdate=datetime.datetime.now())
                meta_data_name = model_name
                model_name += '.pickle'
                model_path = args.out_dir+'/'+model_name
                
                with open(model_path, 'wb') as file:
                    pickle.dump(data, file)
                
                print("Saved to: ", model_name)


                with open(base_dir+'/'+filename+'.pickle','rb') as file:
                    db_full = pickle.load(file)
                db_full['proposal_cov'] = proposal_cov_str
                with open(base_dir+'/'+filename+'_tuned.pickle','wb') as file:
                    pickle.dump(db_full, file)
                print("Saved tuned trace to: ", base_dir+'/'+filename+'_tuned.pickle')

           
# param = 18
# for key in db['step'][0]:
#     liks = []
#     steps = [0.1]
#     step = 0.1
#     for i in range(10):
#         lik = db['logp'][i][param][key][1]
#         log_lik = db['logp'][i][param][key][0]
#         liks.append(lik)
#         change = lik
#         new_step = step*np.double(change)**-1
#         new_step = max(min(new_step, 1.5), -1.5)
#         if np.isinf(log_lik):
#             new_step = step/10
#         step = new_step*0.1 + step*0.9
#         steps.append(step)
#     plt.figure()
#     plt.plot(liks)
#     plt.scatter(np.arange(0,11), steps)

# sd = np.zeros((56, 25))
# i = 0
# for key_grp in model_metadata.keys_all:
#     for key in key_grp:
#         for j in range(25):
#             sd[i, j] = abs(step[j][key])
#         i += 1
        


# vOld = np.arange(-150,50)
# self = model(*model_params)

# tj = self.tj_baseline + 1.0 / (self.tj_mult1 * np.exp(-(vOld + 100.6) / self.tj_tau1) +
#                                     self.tj_mult2 * np.exp((vOld + 0.9941) / self.tj_tau2));
# plt.figure('tau inactivation')
# plt.plot(vOld, tj)

# thf = 1.0 / (self.thf_mult1 * np.exp(-(vOld + 1.196) / self.thf_tau1) +
#                     self.thf_mult2 * np.exp((vOld + 0.5096) / self.thf_tau2));

# plt.figure('tau inactivation')             
# plt.plot(vOld, thf)

# plt.show()
