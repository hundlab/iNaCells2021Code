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

class ObjContainer():
    pass

def stop_sim(pymc_model):
    pymc_model.pause()
    pymc_model.save_state()
    pymc_model.tally()
    pymc_model.halt()
    print("Sampling Canceled")
    
if __name__ == '__main__':
    
    atrial_model.run_sims_functions.plot1 = False #sim
    atrial_model.run_sims_functions.plot2 = False #diff
    atrial_model.run_sims_functions.plot3 = False #tau
    
    key_frame = key_frame(keys_keep_grps, exp_parameters)
    
    map_res = {}
    map_res['keys'] = list(key_frame.index)

    model_name = './MAP_'
    model_name +=  args.model_name
    model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    model_name = model_name.format(cdate=datetime.datetime.now())
    db_path = args.out_dir+'/'+model_name+'.pickle'

    start = None
    if not args.previous_run is None:
        with open(args.previous_run,'rb') as file:
            old_db = pickle.load(file)
            if args.previous_run_manual:
                start = {}
                model_param_var = np.zeros((len(key_frame.index),model.num_params))
                for i,key in enumerate(key_frame.index):
                    try:
                        loc = old_db['keys'].index(key)
                        model_param_var[i] = old_db['MAP']['model_param'][loc,:]
                    except ValueError:
                        model_param_var[i] = np.zeros(model.num_params)
                start['model_param'] = model_param_var
                start['model_param_intercept'] = old_db['MAP']['model_param_intercept']
                start['model_param_sigma_log__'] = old_db['MAP']['model_param_sigma_log__']
                start['b_temp'] = old_db['MAP']['b_temp']
                start['model_param_sigma'] = old_db['MAP']['model_param_sigma']
            else:
                start = old_db['MAP']
            

    with Pool() as proc_pool:
        calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                mp_locs=mp_locs, data=datas,error_fill=0,\
                pool=proc_pool)
        run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs, disp_print=False)
        
        with StatModel(run_biophysical, key_frame, datas,
                                mp_locs, model) as stat_model:
            #method='powell'method='Nelder-Mead'
            map_estimates = pm.find_MAP(start=start, model=stat_model, method='powell', maxeval=1000)
            map_res['MAP'] = map_estimates

    with open(db_path, 'wb') as file:
        pickle.dump(map_res, file)

    # # model_name = './mcmc_'
    # # model_name +=  args.model_name
    # # model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    # # model_name = model_name.format(cdate=datetime.datetime.now())
    # # meta_data_name = model_name
    # # model_name += '.pickle'
    # # db_path = args.out_dir+'/'+model_name
    
    # # meta_data_name += '_metadata.pickle'
    # # meta_data_path = args.out_dir+'/'+meta_data_name
    
    # # model_metadata = ObjContainer()
    # # model_metadata.model_params_initial = model_params_initial
    # # model_metadata.mp_locs = mp_locs
    # # model_metadata.keys_all = keys_all
    # # model_metadata.param_bounds = model.param_bounds
    # # model_metadata.bio_model_name = args.model_name

    # print("Running Pool with", os.cpu_count(), "processes")
    # with Pool() as proc_pool:
        
    #     calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
    #                     mp_locs=mp_locs, data=datas,error_fill=0,\
    #                     pool=proc_pool)
    #     run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs)

        
    #     made_model = make_model(run_biophysical, keys_keep_grps, datas, 
    #                             model_params_initial, mp_locs, model, 
    #                             exp_parameters['temp ( K )'])
    #     db = 'pickle'
    #     if not args.previous_run is None and not args.previous_run_manual:
    #         db = pymc.database.pickle.load(args.previous_run)
    #     elif not args.previous_run is None:
    #             with open(args.previous_run,'rb') as file:
    #                 old_db = pickle.load(file)
    #             prev_state = old_db['_state_']['stochastics']
    #             for var in made_model:
    #                 if var.__name__ == 'model_param':
    #                     var.value = prev_state[var.__name__]
    #                 # if var.__name__ in prev_state:
    #                 #     var.value = prev_state[var.__name__]
    #             del old_db
    #     pymc_model = pymc.MCMC(made_model, db=db)#, dbname=db_path)

    #     if not args.max_time is None:
    #         #max_time is in hours
    #         sample_timer = Timer(args.max_time*60*60, stop_sim, args=(pymc_model,))
    #         sample_timer.start()
            
    #     pymc_model.sample(iter=1000, burn=0, thin=1, tune_throughout=True, save_interval=100)#, burn_till_tuned=True)
    #     pymc_model.db.close()
        
    #     if not args.max_time is None:
    #         sample_timer.join()
        
    #     # model_metadata.num_calls = run_biophysical.call_counter
    #     # model_metadata.trace_pickel_file = model_name
    #     # with open(meta_data_path, 'wb') as file:
    #     #     pickle.dump(model_metadata, file)
            
    #     # print("Pickle File Written to:")
    #     # print(model_name)
    #     #pymc.Matplot.plot(S)
