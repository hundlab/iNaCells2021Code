#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:30:52 2020

@author: grat05
"""

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

import atrial_model
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_diff, calc_results
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial
from atrial_model.iNa.define_sims import sim_fs, datas,\
    keys_all, exp_parameters

from sklearn.preprocessing import minmax_scale
from multiprocessing import Pool
import numpy as np
import pickle
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd


class ObjContainer():
    pass    

keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

##iv curve
keys_iin = [
('8928874_7',	'Dataset C day 1'), #('8928874_7',	'Dataset C day 3'),
#('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
#('12890054_3',	'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
#('12890054_5',	'Dataset C Control'), ('12890054_5',	'Dataset D Control'),
#('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
#('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
#('1323431_4',	'Dataset B Control'),
#('21647304_1',	'Dataset B Adults'), ('21647304_1', 'Dataset B Pediatrics')
]
keys_keep += keys_iin


##activation normalized to driving force
keys_iin = [
            ('1323431_2',	'Dataset')#,\
#            ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
#            ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
#            ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')
]
keys_keep += keys_iin




# I2/I1 Recovery
keys_iin = [('1323431_8', 'Dataset A -140')#, ('1323431_8',	'Dataset A -120'),\
#            ('1323431_8',	'Dataset A -100'),\
#            ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
#            ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
#            ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')
]
keys_keep += keys_iin


# #recovery normalized to preprepulse
#keys_iin = [\
#('7971163_6', 'Dataset -75'),\
#('7971163_6', 'Dataset -85'),\
#('7971163_6', 'Dataset -95'),\
#('7971163_6', 'Dataset -105'),\
#('7971163_6', 'Dataset -115'),
#('7971163_6', 'Dataset -125'),\
#('7971163_6', 'Dataset -135')
#]
#keys_keep += keys_iin




##inactivation normalized to no prepulse
keys_iin = [
#    ('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),
#            ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),
              ('7971163_4', 'Dataset 512ms'),\

#            ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
#            ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5')
            ]
##('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
keys_keep += keys_iin


#inactivation normalized to first
#keys_iin = [#('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
#            ('7971163_5',	'Dataset A -85')#, ('7971163_5',	'Dataset A -95'),\
#            ('7971163_5',	'Dataset A -105')
#            ]
#keys_keep += keys_iin



#tau inactivation
#keys_iin = [('8928874_8', 'Dataset E fresh')#, ('8928874_8',	'Dataset E day 1'),\
#            ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')
#            ]#,\
#            ('1323431_5',	'Dataset B fast'),\
#            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
#keys_keep += keys_iin

#tau activation
keys_iin = [#('8928874_8',	'Dataset D fresh')#, ('8928874_8',	'Dataset D day 1'),\
#            ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5'),
            ('7971163_3',	'Dataset C')
            ]
keys_keep += keys_iin




# #tau inactivation fast & slow
# keys_iin = [('21647304_2', 'Dataset C Adults'), ('21647304_2',	'Dataset D Adults'),\
#             ('21647304_2', 'Dataset C Pediactric'), ('21647304_2',	'Dataset D Pediactric')]
# #('1323431_5',	'Dataset B fast'),('1323431_5',	'Dataset B slow'),\
# keys_keep += keys_iin



# #tau inactivation normalized to first
# keys_iin = [('1323431_6',	'Dataset -80'), ('1323431_6',	'Dataset -100')]
# keys_keep += keys_iin


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#from SaveSAP import savePlots,setAxisSizePlots
#sizes = {'deviance': (4, 4), 'deviance_zoomed': (4, 4), 'model_param_mean': (4, 4), 'b_temp': (4, 4), 'model_param_tau': (4, 4), 'model_params_legend': (2, 6), 'error_tau': (4, 4), 'sim_groups_legend': (2, 6), 'GNaFactor': (6, 2), 'baselineFactor': (6, 2), 'mss_tauFactor': (6, 2), 'mss_shiftFactor': (6, 2), 'tm_maxFactor': (6, 2), 'tm_tau1Factor': (6, 2), 'tm_shiftFactor': (6, 2), 'tm_tau2Factor': (6, 2), 'hss_tauFactor': (6, 2), 'hss_shiftFactor': (6, 2), 'thf_maxFactor': (6, 2), 'thf_shiftFactor': (6, 2), 'thf_tau1Factor': (6, 2), 'thf_tau2Factor': (6, 2), 'ths_maxFactor': (6, 2), 'ths_shiftFactor': (6, 2), 'ths_tau1Factor': (6, 2), 'ths_tau2Factor': (6, 2), 'Ahf_multFactor': (6, 2), 'jss_tauFactor': (6, 2), 'jss_shiftFactor': (6, 2), 'tj_maxFactor': (6, 2), 'tj_shiftFactor': (6, 2), 'tj_tau2Factor': (6, 2), 'tj_tau1Factor': (6, 2)}
#setAxisSizePlots(sizes)
#savePlots('R:/Hund/DanielGratz/atrial_model/plots/latest/OHaraRudy_wMark/', ftype='svg')
#setAxisSizePlots([(4,4)]*40)
#setAxisSizePlots((3,3))

keys_keep = set(keys_keep)
sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_keep}
datas = {key: data for key, data in datas.items() if key in keys_keep}

#adjust to physilogical temp and conc
sim_param = dict(naO=140, naI=9.136, TEMP=310)
for key in sim_fs:
    sim_fs[key].sim_param = sim_param
voltages = np.empty((26,2))
voltages[:,0] = -140
voltages[:,1] = np.arange(-90, 40, 5)
durs = np.empty_like(voltages)
durs[:,:] = [10, 20]
sim_fs[('8928874_7', 'Dataset C day 1')].durs = durs
sim_fs[('8928874_7', 'Dataset C day 1')].voltages = voltages
datas[('8928874_7', 'Dataset C day 1')] = np.fliplr(voltages)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
atrial_model.run_sims_functions.plot1 = False#True#True #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau

if __name__ == '__main__':
    chain = 0
    burn_till =  1000
    use_cache = False
    num_draws = 100
    
    if not use_cache:
    
        filename = 'mcmc_OHaraRudy_wMark_INa_0924_1205'
        #filename = 'mcmc_OHaraRudy_wMark_INa_0831_1043_sc'
        #filename = 'mcmc_OHaraRudy_wMark_INa_0919_1832'
        #filename = 'mcmc_OHaraRudy_wMark_INa_0831_1043'
        
        filename = 'mcmc_OHaraRudy_wMark_INa_1012_1149'


        filename = 'mcmc_OHaraRudy_wMark_INa_1213_1353'
        
        filename = 'mcmc_OHaraRudy_wMark_INa_0127_1525'

        
        #filename = './mcmc_Koval_0511_1609'
        #filename = 'mcmc_Koval_0601_1835'
        #filename = 'mcmc_OHaraRudy_wMark_INa_0528_1833'
        #filename = 'mcmc_OHaraRudy_wMark_INa_0606_0047'
        #filename = 'mcmc_OHaraRudy_wMark_INa_0627_1152'
        #filename = 'mcmc_OHaraRudy_wMark_INa_0702_1656'
        
        base_dir = atrial_model.fit_data_dir+'/'
        with open(base_dir+'/'+filename+'.pickle','rb') as file:
            db_full = pickle.load(file)
        db = db_full['trace']
        db_post = db.warmup_posterior
        
        # base_dir = atrial_model.fit_data_dir+'/'
        # with open(base_dir+'/'+filename+'_metadata.pickle','rb') as file:
        #     model_metadata = pickle.load(file)
        # with open(base_dir+model_metadata.trace_pickel_file,'rb') as file:
        #     db = pickle.load(file)
        # if db['_state_']['sampler']['status'] == 'paused':
        #     current_iter = db['_state_']['sampler']['_current_iter']
        #     current_iter -= db['_state_']['sampler']['_burn']
        #     for key in db.keys():
        #         if key != '_state_':
        #             db[key][chain] = db[key][chain][:current_iter]
        temp = 20
        # b_temp = np.median(db['b_temp'][chain][burn_till:], axis=0)
        # for i in range(db['b_temp'][chain].shape[1]):
        #     trace = db['b_temp'][chain][burn_till:, i]
        #     f_sig = np.sum(trace > 0)/len(trace)
        #     if not (f_sig < 0.05 or f_sig > 0.95):
        #         b_temp[i] = 0
                
        b_temp = np.zeros_like(mp_locs, dtype=float)
        b_temp[[1,4,10,14,21]] = -0.7/10
        b_temp[[3,6,9,11,15,20,22]] = 0.4
        b_temp[[3]] = -0.4
        
        #b_temp[[2,4,10,12]] = 0
        intercept = np.median(db_post['model_param_intercept'][chain][burn_till:], axis=0)
        intercept[18] = 1.7
        mp_mean = intercept + b_temp*temp
        mp_sd = np.median(db_post['model_param_sd'][chain][burn_till:], axis=0)
        
        mp_cor = np.median(db_post['model_param_corr'][chain][burn_till:], axis=0)
        mp_cov = np.outer(mp_sd, mp_sd)*mp_cor
#        med_model_param = np.median(db_post['model_param'][chain][burn_till:], axis=0)
#        mp_cov = np.cov((med_model_param.T-np.mean(med_model_param, axis=1)))
        
#        mp_cov[:,18] /= np.sqrt(3)
#        mp_cov[18,:] /= np.sqrt(3)
        
        #corrections
#        mp_sd[1] = 0.5
     #   mp_sd[3] = mp_sd[9]
#        mp_sd[[3,6,9,11,15,20,22]] = 0
#        mp_sd[[1,4,10,14,21]] = 0
#        mp_sd[[5,7,12, 13]] = 0.5
#        mp_sd[[0, 2, 5, 7, 8, 12, 13, 16, 17, 18, 19, 23, 24]] = 0
        
#        mp_sd[17] = 0.5
#        mp_sd[18] = 0.5
        
        # cov = np.diag(mp_sd**2)
        # cov[4,10] = cov[10,4] = 0.8
        # cov[5,13] = cov[13,5] = 0.5
        

        
        sub_mp_draws = np.random.multivariate_normal(mean=mp_mean, cov=mp_cov, size=num_draws)
        
        #sub_mp_draws = np.random.normal(loc=mp_mean, scale=mp_sd, 
        #                                size=(num_draws, len(mp_mean)))
        
        sub_mps = {}
        sim_fs_draws = {}
        for key in keys_keep:
            new_key = (key[0], key[1], "mean")
            sub_mps[new_key] = mp_mean
            sim_fs_draws[new_key] = sim_fs[key]
            for i in range(num_draws):
                new_key = (key[0], key[1], i)
                sub_mps[new_key] = sub_mp_draws[i]
                sim_fs_draws[new_key] = sim_fs[key]
        
        with Pool() as proc_pool:
    #        proc_pool = None
    
            results = calc_results(sub_mps, sim_funcs=sim_fs_draws,\
                                model_parameters_full=model_params_initial,\
                                mp_locs=mp_locs, data=datas,error_fill=0,\
                                pool=proc_pool)
        
        mean_res = {}
        reformated_res = {key: [None]*num_draws for key in keys_keep}
        for key in keys_keep:
            for res_key in results:
                if key == res_key[0:2]:
                    idx = res_key[2]
                    if type(idx) == str:
                        mean_res[key] = results[res_key]
                    else:
                        reformated_res[key][idx] = results[res_key]
        results = reformated_res
        for key in results:
            lens = np.array([len(arr) for arr in results[key]])
            max_len = max(lens)
            for i in np.where(lens < max_len)[0]:
                results[key][i] = results[key][i]*np.ones(max_len)
            results[key] = np.array(results[key])
              
        base_dir = atrial_model.fit_data_dir+'/'
        with open(base_dir+'/'+'variability_cache.pickle','wb') as file:
            to_dump = dict(results=results,
                           mean_res=mean_res,
                           sub_mps=sub_mps,
                           num_draws=num_draws,
                           sub_mp_draws=sub_mp_draws)
            pickle.dump(to_dump, file)
    else:
        base_dir = atrial_model.fit_data_dir+'/'
        with open(base_dir+'/'+'variability_cache.pickle','rb') as file:
            loaded = pickle.load(file)
            results = loaded['results']
            sub_mps = loaded['sub_mps']
            num_draws = loaded['num_draws']
            sub_mp_draws = loaded['sub_mp_draws']
            mean_res = loaded['mean_res']
    
    from SaveSAP import paultcolors
    c_scheme = 'muted'
    colors = paultcolors[c_scheme]
    
    
    for key in keys_keep:
        sl = slice(0, None)
        if key == ('7971163_3', 'Dataset C'):
            sl= slice(300, 350)
        figname = exp_parameters.loc[key, 'Sim Group']
        figname = figname if not pd.isna(figname) else 'Missing Label'
        fig = plt.figure(figname + " variability")
        ax = fig.add_subplot(1,1,1)
        for i in range(num_draws):
            try:
                ax.plot(datas[key][sl,0], results[key][i][sl], alpha=0.6, c=colors[1])
            except ValueError:
                print("Error on ", i)
        avg = np.mean(results[key][:,sl], axis=0)
        ax.plot(datas[key][sl,0], avg, c=colors[0])
        ax.plot(datas[key][sl,0], mean_res[key][sl], c='black')

# for key in keys_keep:
#     sl = slice(0, None)
#     if key == ('7971163_3', 'Dataset C'):
#         sl= slice(300, 350)
#     figname = exp_parameters.loc[key, 'Sim Group']
#     figname = figname if not pd.isna(figname) else 'Missing Label'
#     fig = plt.figure(figname + " variability")
#     ax = fig.add_subplot(1,1,1)
#     avg = np.mean(results[key][:,sl], axis=0)
#     conf = stats.sem(results[key][:,sl], axis=0)
#     ax.plot(datas[key][sl,0], avg)
#     ax.plot(datas[key][sl,0], mean_res[key][sl], c='black')
#     ax.fill_between(datas[key][sl,0], avg+conf, avg-conf, alpha=0.4)
                
        #plt.legend()
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
