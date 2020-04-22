#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:06:36 2020

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
from iNa_sims import sim_fs, datas,\
    keys_all, exp_parameters
from iNa_model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial

    

keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

##iv curve
keys_iin = [
('8928874_7',	'Dataset C day 1'), ('8928874_7',	'Dataset C day 3'),
('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
#('12890054_3',	'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
#('12890054_5',	'Dataset C Control'), ('12890054_5',	'Dataset D Control'),
('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
('1323431_4',	'Dataset B Control'),
('21647304_1',	'Dataset B Adults'), ('21647304_1', 'Dataset B Pediatrics')
]
keys_keep += keys_iin


#activation normalized to driving force
keys_iin = [
            ('1323431_2',	'Dataset'),\
            ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
            ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
            ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')]
keys_keep += keys_iin




# I2/I1 Recovery
keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
            ('1323431_8',	'Dataset A -100'),\
            ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
            ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
            ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')]
keys_keep += keys_iin


# recovery normalized to preprepulse
keys_iin = [\
('7971163_6', 'Dataset -75'),\
('7971163_6', 'Dataset -85'),\
('7971163_6', 'Dataset -95'),\
('7971163_6', 'Dataset -105'),\
('7971163_6', 'Dataset -115'),
('7971163_6', 'Dataset -125'),\
('7971163_6', 'Dataset -135')
]
keys_keep += keys_iin




#inactivation normalized to no prepulse
keys_iin = [('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),
             ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),
             ('7971163_4', 'Dataset 512ms'),\


             ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
             ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5')
            ]
#('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
keys_keep += keys_iin


# inactivation normalized to first
keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
            ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
            ('7971163_5',	'Dataset A -105')]
keys_keep += keys_iin



#tau inactivation
keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
            ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]#,\
#            ('1323431_5',	'Dataset B fast'),\
#            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
keys_keep += keys_iin



#tau inactivation fast & slow
keys_iin = [('21647304_2', 'Dataset C Adults'), ('21647304_2',	'Dataset D Adults'),\
            ('21647304_2', 'Dataset C Pediactric'), ('21647304_2',	'Dataset D Pediactric')]
#('1323431_5',	'Dataset B fast'),('1323431_5',	'Dataset B slow'),\
keys_keep += keys_iin



## #tau inactivation normalized to first
## keys_iin = [('1323431_6',	'Dataset -80'), ('1323431_6',	'Dataset -100')]
## keys_keep += keys_iin



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

keys_keep = set(keys_keep)
sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_keep}
datas = {key: data for key, data in datas.items() if key in keys_keep}


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
iNa_fit_functions.plot1 = True#False#True #sim
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
