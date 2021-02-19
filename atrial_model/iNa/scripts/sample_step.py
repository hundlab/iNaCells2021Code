#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:51:07 2020

@author: grat05
"""

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

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


from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle
import copy
import os
import datetime
from scipy import stats


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
    class ObjContainer():
        pass
    
    filename = 'mcmc_OHaraRudy_wMark_INa_1113_1714'

    base_dir = atrial_model.fit_data_dir+'/'
    with open(base_dir+'/'+filename+'_metadata.pickle','rb') as file:
        model_metadata = pickle.load(file)
    with open(base_dir+model_metadata.trace_pickel_file,'rb') as file:
        db = pickle.load(file)
    
    # #starting model params
    # with open(os.path.dirname(__file__)+'/sub_mps_sigma_est.pickle','rb') as file:
    #     sub_mps_base = pickle.load(file)
        
    group_names = []
#    sim_groups = []
    sim_names = []
    for key_group in model_metadata.keys_all:
        group_names.append(exp_parameters.loc[key_group[0], 'Sim Group'])
        for key in key_group:
            sim_names.append(key)
#            sim_groups.append(group_names[-1])

        
    sub_mps_base = dict(zip(sim_names, db['_state_']['stochastics']['model_param']))
    error_sigma = dict(zip(group_names, db['_state_']['stochastics']['error_tau']**(-1/2)))
    
    
    # 
#    with open(os.path.dirname(__file__)+'/step_res_sigma_est.pickle','rb') as file:
#        step_res = pickle.load(file)
        
 #    error_sigma = {'I2/I1 Recovery': 0.029173699199566576,
 # 'inactivation normalized to no prepulse': 0.03731537357577712,
 # 'tau activation': 0.1704122387829371,
 # 'activation normalized to driving force': 0.03221871715530066,
 # 'iv curve': 0.4036486014172108,
 # 'normalized iv curve': 0.18657794824646226}
    
    # log_ps = []
    # for res in step_res:
    #     log_p = {}
    #     for key, dat in res.items():
    #         normal = stats.norm(loc=dat, scale=error_sigma[sim_groups[key]])
    #         liks = normal.pdf(datas[key][:,1])
    #         log_lik = np.sum(np.log(liks))
    #         log_p[key] = log_lik
    #     log_ps.append(log_p)
    
#    start = log_ps[0]
    # step = []
    # for i in range(1, len(log_ps), 2):
    #     log_p_up = log_ps[i]
    #     log_p_down = log_ps[i+1]
    #     mp_step = {}
    #     for key in log_p_up:
    #         log_lik_up = abs(start[key] - log_p_up[key])
    #         log_lik_down = abs(start[key] - log_p_down[key])
    #         if log_lik_up > log_lik_down:
    #             max_change = log_lik_up
    #             direction = 1
    #         else:
    #             max_change = log_lik_down
    #             direction = -1
    #         new_step = direction* min(0.1*np.double(max_change)**-1, 1.5)
    #         mp_step[key] = new_step
    #     step.append(mp_step)
            
    # sub_mps_list = []
    # for i in range(len(step)):
    #     sub_mps_sing = {}
    #     for key in mp_step:
    #         base = sub_mps_base[key].copy()
    #         base[i] += step[i][key]
    #         sub_mps_sing[key] = base
    #     sub_mps_list.append(sub_mps_sing)
        
        
    step_all = {}
    sub_mps_all = {}
    for key in keys_all_list:
        for i in range(len(model_param_names)):
            key_up = (key[0],key[1],i,"up")
            key_down = (key[0],key[1],i,"down")
            step_all[key_up] = 0.1
            step_all[key_down] = -0.1
            
            base = sub_mps_base[key].copy()
            base[i] += step_all[key_up]
            sub_mps_all[key_up] = base
            base = sub_mps_base[key].copy()
            base[i] += step_all[key_down]
            sub_mps_all[key_down] = base

            
    with Pool() as proc_pool:
        diff_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                         mp_locs=mp_locs, sim_funcs=sim_fs, data=datas,\
                         pool=proc_pool)
        
        start = {}
        res = diff_fn(sub_mps_base, exp_params=exp_parameters)
        for key, dat in res.items():
            normal = stats.norm(loc=dat, scale=error_sigma[sim_groups[key]])
            liks = normal.pdf(datas[key][:,1])
            log_lik = np.sum(np.log(liks))
            start[key] = log_lik
        
        sim_fs_all = {key: sim_fs[key[:2]] for key in sub_mps_all}
        diff_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                 mp_locs=mp_locs, sim_funcs=sim_fs_all, data=datas,\
                 pool=proc_pool)
            
        n_iter = 10
        results = []
        logp_vals = []
        steps = []

        for it in range(n_iter):
            print(it, end=' ', flush=True)
            res = diff_fn(sub_mps_all, exp_params=exp_parameters)
            results.append(res)
            steps.append(step_all)
            logp_vals.append({})
            for key, dat in res.items():
                normal = stats.norm(loc=dat, scale=error_sigma[sim_groups[key[0:2]]])
                liks = normal.pdf(datas[key[0:2]][:,1])
                log_lik = np.sum(np.log(liks))
                change = abs(start[key[0:2]] - log_lik)
                new_step = step_all[key]*np.double(change)**-1
                new_step = max(min(new_step, 1.5), -1.5)
                if np.isinf(log_lik):
                    new_step = step_all[key]/10
                else:
                    new_step = new_step*0.1 + step_all[key]*0.9
                step_all[key] = new_step
                base = sub_mps_base[key[0:2]].copy()
                base[key[2]] += new_step
                sub_mps_all[key] = base
                logp_vals[it][key] = log_lik, change


        # for it in range(n_iter):
        #     results.append([])
        #     logp_vals.append([])
        #     print(it, end=': ')
        #     sub_mps_list_n = []
        #     for i,sub_mps in enumerate(sub_mps_list):
        #         print(i, end=' ', flush=True)
        #         res = diff_fn(sub_mps, exp_params=exp_parameters)
        #         results[it].append(res)
        #         logp_vals[it].append({})
        #         sub_mps_sing = {}
        #         for key, dat in res.items():
        #             normal = stats.norm(loc=dat, scale=error_sigma[sim_groups[key]])
        #             liks = normal.pdf(datas[key][:,1])
        #             log_lik = np.sum(np.log(liks))
        #             change = abs(start[key] - log_lik)
        #             new_step = step[i][key]*np.double(change)**-1
        #             new_step = max(min(new_step, 1.5), -1.5)
        #             if np.isinf(log_lik):
        #                 new_step = step[i][key]/10
        #             else:
        #                 new_step = new_step*0.1 + step[i][key]*0.9
        #             step[i][key] = new_step
        #             base = sub_mps_base[key].copy()
        #             base[i] += new_step
        #             sub_mps_sing[key] = base
        #             logp_vals[it][i][key] = log_lik, change
        #         sub_mps_list_n.append(sub_mps_sing)
        #     sub_mps_list = sub_mps_list_n
        #     print('')
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


# proposal_sd = np.empty((len(sim_names), len(sub_mps)))
# steps = step_db['steps'][-1]
# for i in range(len(sub_mps)):
#     for j,key in enumerate(sim_names):
#         up = steps[(key[0],key[1], i, 'up')]
#         down = steps[(key[0],key[1], i, 'down')]
#         proposal_sd[j,i] = min(up, abs(down))


           
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
