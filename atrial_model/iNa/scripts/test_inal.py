#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:25:26 2020

@author: grat05
"""

import sys
sys.path.append('../../../')

#from iNa_models import Koval_ina, OHaraRudy_INa
import atrial_model
from atrial_model.iNa.models import OHaraRudy_INa, Koval_ina
import atrial_model.run_sims_functions
from atrial_model.run_sims_functions import peakCurr, normalized2val, calcExpTauInact, monoExp,\
calcExpTauAct, triExp, biExp
from atrial_model.run_sims import calc_diff
from atrial_model.iNa.define_sims_late import sim_fs, datas,\
    keys_all, exp_parameters
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle

class ObjContainer():
    pass    

keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

## peak late curr
keys_iin = [('20488304_1', 'Dataset B SR')]

keys_keep += keys_iin


# iv curve
keys_iin = [('20488304_2', 'Dataset D Vehicle')]

keys_keep += keys_iin

# curve integral
keys_iin = [('20488304_1', 'Dataset F SR')]

keys_keep += keys_iin

# late current
keys_iin = [('26121051_2', 'Dataset B SR Control BT')]
keys_iin += [('26121051_2', 'Dataset B SR Control RT')]

keys_keep += keys_iin
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

keys_keep = set(keys_keep)
sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_keep}
datas = {key: data for key, data in datas.items() if key in keys_keep}


#import pickle
#res = pickle.load(open('./optimize_Koval_0423_0326.pkl','rb'))
#mp_locs = res.mp_locs
#sub_mps = res.x

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
atrial_model.run_sims_functions.plot1 = True #sim
atrial_model.run_sims_functions.plot2 = True #diff
atrial_model.run_sims_functions.plot3 = False #tau

if __name__ == '__main__':
    filename = 'mcmc_OHaraRudy_wMark_INa_0627_1152'
    filepath = atrial_model.fit_data_dir+'/'+filename
    chain = 0
    burn_till = 20000
    with open(filepath+'.pickle','rb') as file:
        db = pickle.load(file)
    with open(filepath+'_metadata.pickle','rb') as file:
        model_metadata = pickle.load(file)
    if db['_state_']['sampler']['status'] == 'paused':
        current_iter = db['_state_']['sampler']['_current_iter']
        for key in db.keys():
            if key != '_state_':
                db[key][chain] = db[key][chain][:current_iter]
    mp_locs = model_metadata.mp_locs
    b_temp = np.median(db['b_temp'][0][burn_till:], axis=0)
    intercept = np.median(db['model_param_mean'][0][burn_till:], axis=0)
    sub_mps = intercept
    
    with Pool() as proc_pool:
#        proc_pool = None
        diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas,\
                            pool=proc_pool,ssq=True)
                    
        error = diff_fn(sub_mps, exp_params=exp_parameters, 
                        keys=[key for key_group in keys_all for key in key_group])
        print(error)
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
