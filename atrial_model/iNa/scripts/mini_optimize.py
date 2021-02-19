#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:48:30 2020

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
from atrial_model.run_sims import calc_diff
from atrial_model.iNa.define_sims import sim_fs, datas,\
    keys_all, exp_parameters
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial
from atrial_model.parse_cmd_args import args
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_results, SimResults
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all, exp_parameters
from atrial_model.iNa.model_setup import model_params_initial, mp_locs, sub_mps, model
from atrial_model.optimization_functions import lstsq_wrap, save_results


from multiprocessing import Pool
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle
import datetime
import numpy as np
import pickle
from threading import Timer
from multiprocessing import Manager
from functools import partial
import os
import pyDOE
from scipy.stats import distributions


keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# # #iv curve
# keys_iin = [
# ('8928874_7',	'Dataset C day 1'), ('8928874_7',	'Dataset C day 3'),
# ('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
# #('12890054_3',	'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
# #('12890054_5',	'Dataset C Control'), ('12890054_5',	'Dataset D Control'),
# ('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
# ('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
# ('1323431_4',	'Dataset B Control'),
# ('21647304_1',	'Dataset B Adults'), ('21647304_1', 'Dataset B Pediatrics')
# ]
# keys_keep += keys_iin


# ##activation normalized to driving force
# keys_iin = [
#             ('1323431_2',	'Dataset'),\
#             ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
#             ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
#             ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')
# ]
# keys_keep += keys_iin




# I2/I1 Recovery
# keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
#              ('1323431_8',	'Dataset A -100'),\
#             ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
#             ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
#             ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')
# ]
# keys_keep += keys_iin


# # #recovery normalized to preprepulse
# keys_iin = [\
# ('7971163_6', 'Dataset -75'),\
# ('7971163_6', 'Dataset -85'),\
# ('7971163_6', 'Dataset -95'),\
# ('7971163_6', 'Dataset -105'),\
# ('7971163_6', 'Dataset -115'),
# ('7971163_6', 'Dataset -125'),\
# ('7971163_6', 'Dataset -135')
# ]
# keys_keep += keys_iin




#inactivation normalized to no prepulse
keys_iin = [
    ('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),
              ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),
                ('7971163_4', 'Dataset 512ms'),\

#              ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
#              ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5')
            ]
#('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
keys_keep += keys_iin


# #inactivation normalized to first
# keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
#             ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
#             ('7971163_5',	'Dataset A -105')
#             ]
# keys_keep += keys_iin



#tau inactivation
keys_iin = [('8928874_8', 'Dataset E fresh')#, ('8928874_8',	'Dataset E day 1'),\
#            ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]#,\
#            ('1323431_5',	'Dataset B fast'),\
#            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')
]
keys_keep += keys_iin

#####tau activation
# keys_iin = [('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
#             ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5'),
#             #('7971163_3',	'Dataset C')
#             ]
# keys_keep += keys_iin




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

atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau

keys_keep = set(keys_keep)
sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_keep}
datas = {key: data for key, data in datas.items() if key in keys_keep}

if __name__ == '__main__':
    with Pool() as proc_pool:
        mp_locs = list(set(mp_locs))
        sub_mps = model_params_initial[mp_locs]
        sub_mp_bounds = np.array(model().param_bounds)[mp_locs]
        min_res = []
        all_res = []

        # accept_test=partial(check_bounds, bounds=sub_mp_bounds))
#            minimizer_kwargs = {"method": "BFGS", "options": {"maxiter":100}}
 
        diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, sim_func=sim_fs, data=datas,\
                            pool=proc_pool,ssq=True,\
                            results=all_res, prnt_err=False)#
        minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False}}#"bounds": sub_mp_bounds,
        # res = optimize.basinhopping(diff_fn, sub_mps, \
        #                             minimizer_kwargs=minimizer_kwargs,\
        #                             niter=10, T=80,\
        #                             callback=partial(save_results, results=min_res),\
        #                             stepsize=1)#T=80
        
        res = optimize.dual_annealing(diff_fn, bounds=sub_mp_bounds,
                                          no_local_search=True,
                                          local_search_options=minimizer_kwargs,
                                          maxiter=10,maxfun=2000)

        print("Optimization done")
        # param_intervals = np.squeeze(np.diff(np.array(model.param_bounds)))
        # rand_mps = pyDOE.lhs(model.num_params, samples=300)
        # rand_mps = distributions.norm(loc=res.x, scale=param_intervals/2).ppf(rand_mps)
        
        # rand_res = []
        # diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
        #          mp_locs=mp_locs, sim_func=sim_fs, data=datas,\
        #              pool=proc_pool,ssq=True,\
        #              results=rand_res, prnt_err=False)#
        # for i in range(rand_mps.shape[0]):
        #      diff_fn(rand_mps[i])
#        res = optimize.least_squares(diff_fn, sub_mps, \
#                        bounds=np.array(model().param_bounds)[mp_locs].T)
        atrial_model.run_sims_functions.plot2 = True
        diff_fn(res.x)