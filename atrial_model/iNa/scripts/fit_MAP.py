#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:23:20 2020

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

#from atrial_model.iNa.stat_model import make_model
from atrial_model.iNa.stat_model_3 import StatModel, key_frame


from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle
import pymc3 as pm
#import pymc
import datetime
import numpy as np
import pickle
from threading import Timer
from multiprocessing import Manager
from functools import partial
import os
import copy


keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# #iv curve
keys_iin = [
('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
('1323431_4',	'Dataset B Control'),
('7971163_1', 'Dataset'),
 
('8928874_7',	'Dataset C day 1'), ('8928874_7',	'Dataset C day 3'),
('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
('21647304_1',	'Dataset B Adults'), #('21647304_1', 'Dataset B Pediatrics'),
('12890054_3', 'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
('12890054_5', 'Dataset C Control'), #('12890054_5',	'Dataset D Control'),
('23341576_2', 'Dataset C Control')
]
keys_keep += keys_iin


#idealized current traces
keys_iin = [('21647304_2', 'Dataset B Adults'), ('21647304_2', 'Dataset B Pediactric'),
            ('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
            ('8928874_8',	'Dataset D day 3'), ('8928874_8',	'Dataset D day 5'),
            ('7971163_3',	'Dataset C')
]
# keys_iin = [key for key_in in keys_iin 
#             for key in sim_fs 
#             if key[0] == key_in[0] and key_in[1] in key[1]]
keys_keep += keys_iin

##activation normalized to driving force
keys_iin = [
            ('1323431_2',	'Dataset'),\
            ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
            ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
            ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')
]
keys_keep += keys_iin




# I2/I1 Recovery
keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
              ('1323431_8',	'Dataset A -100'),\
            ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
            ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
            ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')
]
keys_keep += keys_iin


# # #recovery normalized to preprepulse
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
keys_iin = [
#    ('7971163_4', 'Dataset 32ms'), ('7971163_4', 'Dataset 64ms'),
#              ('7971163_4', 'Dataset 128ms'), 
                ('7971163_4', 'Dataset 256ms'), ('7971163_4', 'Dataset 512ms'),\

              ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
              ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5'),
              ('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
            ]

keys_keep += keys_iin


# #inactivation normalized to first
keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
            ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
            ('7971163_5',	'Dataset A -105')
            ]
keys_keep += keys_iin






#tau inactivation
# keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
#             ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5')]#,\
#            ('1323431_5',	'Dataset B fast'),\
#            ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')]
#keys_keep += keys_iin

# #####tau activation
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

keys_keep = set(keys_keep)
sim_fs = {key: sim_f for key, sim_f in sim_fs.items() if key in keys_keep}
datas = {key: data for key, data in datas.items() if key in keys_keep}
keys_keep_grps = [[key for key in key_grp if key in keys_keep] for key_grp in keys_all]
keys_keep_grps = [grp for grp in keys_keep_grps if len(grp) > 0]


if __name__ == '__main__':
    
    atrial_model.run_sims_functions.plot1 = False #sim
    atrial_model.run_sims_functions.plot2 = False #diff
    atrial_model.run_sims_functions.plot3 = False #tau
    
    key_frame = key_frame(keys_keep_grps, exp_parameters)
    
    map_res = {}
    map_res['keys'] = list(key_frame.index)
    map_res['keys_grps'] = keys_keep_grps
    map_res['sub_MAP'] = {}

    model_name = './MAP_'
    model_name +=  args.model_name
    model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    model_name = model_name.format(cdate=datetime.datetime.now())
    db_path = args.out_dir+'/'+model_name+'.pickle'
    
    start = {'model_param_intercept': np.zeros(len(mp_locs)),
             'b_temp': np.zeros(len(mp_locs)),
             'model_param_sigma': np.zeros(len(mp_locs)),
             'error_sigma': np.zeros(len(keys_keep_grps)),
             'model_param': []
        }
    counts = np.zeros(len(mp_locs), dtype=int)

    with Pool() as proc_pool:
        calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                mp_locs=mp_locs, data=datas,error_fill=0,\
                pool=proc_pool)
        run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs, disp_print=False)
        
        for i,key_grp in enumerate(keys_keep_grps):
            sub_key_frame = key_frame.loc[key_grp]
            grp_name = sub_key_frame['Sim Group'].unique()[0]
            print(grp_name)
            
            with StatModel(run_biophysical, sub_key_frame, datas,
                                mp_locs, model) as stat_model:
                map_estimates = pm.find_MAP(model=stat_model, include_transformed=False
                                            , method='powell', maxeval=10000)
                map_res['sub_MAP'][grp_name] = map_estimates
            
            fitted = map_estimates['model_param_sigma'] > 1e-3
            grp_size = len(key_grp)*fitted
            counts += grp_size
            start['model_param_intercept'] += grp_size*map_estimates['model_param_intercept']
            start['b_temp'] += grp_size*map_estimates['b_temp']
            start['model_param_sigma'] += grp_size*map_estimates['model_param_sigma']
            start['error_sigma'][i] = map_estimates['error_sigma']
            model_param = map_estimates['model_param'].copy()
            model_param[:, ~fitted] = np.nan
            start['model_param'].append(model_param)
        
        start['model_param_intercept'] /= counts
        start['b_temp'] /= counts
        start['model_param_sigma'] /= counts
        start['model_param'] = np.concatenate(start['model_param'], axis=0)
        for i in range(len(mp_locs)):
            unfit = np.isnan(start['model_param'][:,i])
            mean_vals = start['model_param_intercept'][i] +\
                key_frame['temp ( K )']*start['b_temp'][i]
            start['model_param'][unfit,i] = mean_vals[unfit]

        map_res['start'] = copy.deepcopy(start)
        
        with StatModel(run_biophysical, key_frame, datas,
                                mp_locs, model) as stat_model:
            #method='powell'method='Nelder-Mead'
            print("full model")
            map_estimates = pm.find_MAP(start=start, model=stat_model, include_transformed=False
                                        , method='powell', maxeval=40000)
            map_res['MAP'] = map_estimates

    with open(db_path, 'wb') as file:
        pickle.dump(map_res, file)

   