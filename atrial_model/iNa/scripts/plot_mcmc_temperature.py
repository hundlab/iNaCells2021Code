#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:48:07 2020

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

keys_keep = set(keys_keep)
sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_keep}
datas = {key: data for key, data in datas.items() if key in keys_keep}

# 
# res = pickle.load(open('./optimize_Koval_0423_0326.pkl','rb'))
# #mp_locs = res.mp_locs
#sub_mps = res.x

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
atrial_model.run_sims_functions.plot1 = False#True#True #sim
atrial_model.run_sims_functions.plot2 = True #diff
atrial_model.run_sims_functions.plot3 = False #tau

if __name__ == '__main__':
    chain = 6
    burn_till =  500#31_000
    
    filename = 'mcmc_OHaraRudy_wMark_INa_0924_1205'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0919_1832_sc'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0919_1832'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0831_1043_sc'
    
    filename = 'mcmc_OHaraRudy_wMark_INa_1012_1149'
    
    #filename = './mcmc_Koval_0511_1609'
    #filename = 'mcmc_Koval_0601_1835'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0528_1833'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0606_0047'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0627_1152'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0702_1656'
    
    base_dir = atrial_model.fit_data_dir+'/'
    with open(base_dir+'/'+filename+'_metadata.pickle','rb') as file:
        model_metadata = pickle.load(file)
    with open(base_dir+model_metadata.trace_pickel_file,'rb') as file:
        db = pickle.load(file)
    if db['_state_']['sampler']['status'] == 'paused':
        current_iter = db['_state_']['sampler']['_current_iter']
        current_iter -= db['_state_']['sampler']['_burn']
        for key in db.keys():
            if key != '_state_':
                db[key][chain] = db[key][chain][:current_iter]
    
    with Pool() as proc_pool:
#        proc_pool = None
        temps = [0, 10, 20]
        res_temp = []
        for temp in temps:
            b_temp = np.median(db['b_temp'][chain][burn_till:], axis=0)
            for i in range(db['b_temp'][chain].shape[1]):
                trace = db['b_temp'][chain][burn_till:, i]
                f_sig = np.sum(trace > 0)/len(trace)
                if not (f_sig < 0.05 or f_sig > 0.95):
                    b_temp[i] = 0
            ## removing these helps reduce ill effects
            #2,4
            #b_temp[[2,4,10,12]] = 0
            
            # b_temp = np.zeros_like(mp_locs, dtype=float)
            # b_temp[[1,4,10,14,21]] = -0.7/10
            # b_temp[[3,6,9,11,15,20,22]] = 0.4
            # b_temp[[3]] = -0.4
            
            # b_temp[[2]] = 0
            # b_temp[[10]] = -0.2
            # b_temp[0] = 0.2
            
            intercept = np.median(db['model_param_mean'][chain][burn_till:], axis=0)
            b_temp_eff = b_temp*temp
            sub_mps = intercept + b_temp_eff
#            sub_mps[18] = 1.7
#            sub_mps[[6,8,10]] = intercept[[6,8,10]] - 2*b_temp_eff[[6,8,10]]
#            sub_mps[0] = intercept[0]
#            sub_mps[[1]] = intercept[[1]] - b_temp_eff[[1]]
#            mp_locs = model_metadata.mp_locs
            
            res_temp.append(calc_results(sub_mps, sim_funcs=sim_fs,\
                                model_parameters_full=model_params_initial,\
                                mp_locs=mp_locs, data=datas,error_fill=0,\
                                pool=proc_pool))
          
        for key in keys_keep:
            sl = slice(0, None)
            if key == ('7971163_3', 'Dataset C'):
                sl= slice(300, 350)
            figname = exp_parameters.loc[key, 'Sim Group']
            figname = figname if not pd.isna(figname) else 'Missing Label'
            fig = plt.figure(figname + " temperature effects")
            ax = fig.add_subplot(1,1,1)
            for i, temp in enumerate(temps):
                plt.plot(datas[key][sl,0], res_temp[i][key][sl], label=str(temp+17)+'°C')
        
        handles, labels = ax.get_legend_handles_labels()
        fig = plt.figure('temp_params_legend')
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(handles, labels, frameon=False)
        
        
#setAxisSizePlots([(3,1.5)]*7+[(1,1)])
        
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
