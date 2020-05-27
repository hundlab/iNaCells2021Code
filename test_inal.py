#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:25:26 2020

@author: grat05
"""



#from iNa_models import Koval_ina, OHaraRudy_INa
from iNa_models_ode import OHaraRudy_INa, Koval_ina
import iNa_fit_functions
from iNa_fit_functions import calc_diff, peakCurr, normalized2val, calcExpTauInact, monoExp,\
calcExpTauAct, triExp, biExp
from sklearn.preprocessing import minmax_scale
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from iNaL_sims import sim_fs, datas,\
    keys_all, exp_parameters
from iNa_model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial

    

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

import pickle
res = pickle.load(open('./optimize_Koval_0423_0326.pkl','rb'))
#mp_locs = res.mp_locs
#sub_mps = res.x

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
iNa_fit_functions.plot1 = True#True #sim
iNa_fit_functions.plot2 = True #diff
iNa_fit_functions.plot3 = False #tau

if __name__ == '__main__':
    with Pool() as proc_pool:
#        proc_pool = None
        diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas,\
                            l=0,pool=proc_pool,ssq=True)
                    
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
