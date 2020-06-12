#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:08:51 2020

@author: grat05
"""



import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

from atrial_model.parse_cmd_args import args
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_diff
from atrial_model.optimization_functions import lstsq_wrap, save_results
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all, exp_parameters
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, dt, run_fits,\
    model_params_initial, run_fits


import numpy as np
from scipy import optimize
from functools import partial
from multiprocessing import Pool
import pickle
import datetime
from concurrent import futures

class ObjContainer():
    pass

keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# #iv curve
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




#I2/I1 Recovery
keys_iin = [#('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
            #('1323431_8',	'Dataset A -100'),\
            ('21647304_3',	'Dataset C Adults'),# ('21647304_3',	'Dataset C Pediatrics'),\
            ('8928874_9', 'Dataset fresh'),# ('8928874_9', 'Dataset day 1'),\
            #('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')
]
keys_keep += keys_iin


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




# ##inactivation normalized to no prepulse
# keys_iin = [
#     ('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),
#             ('7971163_4', 'Dataset 128ms'), ('7971163_4', 'Dataset 256ms'),
#               ('7971163_4', 'Dataset 512ms'),\

#             ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
#             ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5')
#             ]
# ##('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
# keys_keep += keys_iin


# #inactivation normalized to first
# keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
#             ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
#             ('7971163_5',	'Dataset A -105')
#             ]
# keys_keep += keys_iin



# #tau inactivation
# keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
#             ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]#,\
# #            ('1323431_5',	'Dataset B fast'),\
# #            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
# keys_keep += keys_iin

# #tau activation
# keys_iin = [('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
#             ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5'),
#             ('7971163_3',	'Dataset C')]
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

#keys_all = set(keys_keep)
#sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_all}
#datas = {key: data for key, data in datas.items() if key in keys_all}


np.seterr(all='ignore')

atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau


if __name__ == '__main__':
    with Pool() as proc_pool:
        mp_locs = list(set(mp_locs))
        sub_mps = model_params_initial[mp_locs]
        sub_mp_bounds = np.array(model().param_bounds)[mp_locs]

        res = ObjContainer()
        fut_results = {}
        res.all_res = {}
        res.res = {}

        with futures.ThreadPoolExecutor() as thread_pool:
            for key in sim_fs:
                all_res = []
                res.all_res[key] = all_res
                
                sim_f = dict(key=sim_fs[key])
                data = dict(key=datas[key])
                diff_fn = partial(calc_diff, model_parameters_full=model_params_initial,\
                                mp_locs=mp_locs, sim_func=sim_f, data=data,\
                                    pool=proc_pool,ssq=True,\
                                    results=all_res)
                minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False}}#"bounds": sub_mp_bounds,

                fut_results[key] = thread_pool.submit(optimize.dual_annealing, 
                                                  diff_fn, bounds=sub_mp_bounds,
                                                  no_local_search=True,
                                                  local_search_options=minimizer_kwargs,
                                                  maxiter=100,maxfun=6000)

            for key in sim_fs:
                res.res[key] = fut_results[key].result()
        res.keys_all = keys_all
        res.fits = set(rfs for rfs in run_fits if run_fits[rfs])
        res.mp_locs = mp_locs
        res.model_name = args.model_name

        filename = 'optimize_each_'+args.model_name+'_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}.pickle'
        filename = filename.format(cdate=datetime.datetime.now())
        filepath = args.out_dir+'/'+filename
        with open(filepath, 'wb') as file:
            pickle.dump(res, file)
        print("Pickle File Written to:")
        print(filepath)

        # #plot!
        # atrial_model.run_sims_functions.plot1 = False #sim
        # atrial_model.run_sims_functions.plot2 = True #diff
        # atrial_model.run_sims_functions.plot3 = False #tau

        # error = diff_fn(res.x, exp_params=exp_parameters, 
        #                 keys=[key for key in key_group for key_group in keys_all])
