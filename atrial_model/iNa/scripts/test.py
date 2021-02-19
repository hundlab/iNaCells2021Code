#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:06:36 2020

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
from atrial_model.run_sims import calc_diff, calc_results
from atrial_model.iNa.define_sims import sim_fs, datas,\
    keys_all, exp_parameters, data
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial, model_param_names
from atrial_model.iNa.stat_model_3 import key_frame

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pickle


#groups = ['Adults', 'Pediatric', 'fresh', 'day 1', 'day 3', 'day 5', ]

keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# # #iv curve
keys_iin = [
# ('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
# ('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
# ('1323431_4',	'Dataset B Control'),
# #('7971163_1', 'Dataset'),
 
# ('8928874_7',	'Dataset C day 1')#, ('8928874_7',	'Dataset C day 3'),
 #('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
 #('21647304_1',	'Dataset B Adults'), ###('21647304_1', 'Dataset B Pediatrics'),
 #('12890054_3', 'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
 #('12890054_5', 'Dataset C Control'), ###('12890054_5',	'Dataset D Control'),
 #('23341576_2', 'Dataset C Control')
]
keys_keep += keys_iin


##activation normalized to driving force
keys_iin = [
#            ('1323431_2',	'Dataset'),\
#            ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
#            ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
#            ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')
]
keys_keep += keys_iin




# # I2/I1 Recovery
keys_iin = [#('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
#               ('1323431_8',	'Dataset A -100'),\
#             ('21647304_3',	'Dataset C Adults')#, ('21647304_3',	'Dataset C Pediatrics'),\
              ('8928874_9', 'Dataset fresh')#, ('8928874_9', 'Dataset day 1'),\
#             ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')
]
keys_keep += keys_iin


# #recovery normalized to preprepulse
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




# #inactivation normalized to no prepulse
keys_iin = [
#                   ('7971163_4', 'Dataset 256ms'), 
#('7971163_4', 'Dataset 512ms'),\

#               ('8928874_8',	'Dataset C fresh')#, ('8928874_8',	'Dataset C day 1'),\
#               ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5'),
# ('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
]
# # #
keys_keep += keys_iin


# #inactivation normalized to first
# keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
#             ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
#             ('7971163_5',	'Dataset A -105')
#             ]
# keys_keep += keys_iin


# #idealized current traces
keys_iin = [#('21647304_2', 'Dataset B Adults'), ('21647304_2', 'Dataset B Pediactric'),
#              ('8928874_8',	'Dataset D fresh'), ('8928874_8',	'Dataset D day 1'),\
#    ('8928874_8',	'Dataset D day 3'),
#                ('8928874_8',	'Dataset D day 5'),
#               ('7971163_3',	'Dataset C')
]
keys_keep += keys_iin





# ##tau inactivation
# keys_iin = [('8928874_8', 'Dataset E fresh'), ('8928874_8',	'Dataset E day 1'),\
#             ('8928874_8',	'Dataset E day 3'), ('8928874_8',	'Dataset E day 5'),#,\
#             ('1323431_5',	'Dataset B fast'),\
#             ('21647304_2', 'Dataset C Adults'), ('21647304_2', 'Dataset C Pediactric')
# ]
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
keys_keep = list(keys_keep)


#import pickle
#res = pickle.load(open(atrial_model.fit_data_dir+'/optimize_Koval_0423_0326.pkl','rb'))
#mp_locs = res.mp_locs
#sub_mps = res.x

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
atrial_model.run_sims_functions.plot1 = True #sim
atrial_model.run_sims_functions.plot2 = True #diff
atrial_model.run_sims_functions.plot3 = False #tau

if __name__ == '__main__':
    class ObjContainer():
        pass
    
    # #filename = 'MAP_OHaraRudy_wMark_INa_0805_0736'
    # #filename = 'MAP_OHaraRudy_wMark_INa_0801_1358'
    # filename = 'MAP_OHaraRudy_wMark_INa_0730_0849'
    # #filename = 'MAP_OHaraRudy_wMark_INa_0729_1715'
    # #filename = 'MAP_OHaraRudy_wMark_INa_0729_1634'
    # #filename = 'MAP_OHaraRudy_wMark_INa_0728_1137'
    # # filename = 'MAP_OHaraRudy_wMark_INa_0728_0818'
    # # # # filename = 'MAP_OHaraRudy_wMark_INa_0727_1720'
    # # # # #filename = 'MAP_OHaraRudy_wMark_INa_0727_1431'
    # # # # #filename = 'MAP_OHaraRudy_wMark_INa_0727_1210'
    # # # # #filename = 'MAP_OHaraRudy_wMark_INa_0727_0926'
    # # # # #filename = 'MAP_OHaraRudy_wMark_INa_0720_1147'
    # # # # #filename = 'MAP_OHaraRudy_wMark_INa_0720_1427'
    # # # # # filename = 'MAP_OHaraRudy_wMark_INa_0721_1012'

    # base_dir = atrial_model.fit_data_dir+'/'
    # with open(base_dir+'/'+filename+'.pickle','rb') as file:
    #     map_result = pickle.load(file)
    # keys_frame = key_frame(keys_keep_grps, exp_parameters)
    # sub_mps = {}
    # for key in keys_frame.index:
    #     if key in map_result['keys']:
    #         loc = map_result['keys'].index(key)
    #         sub_mps[key] = map_result['MAP']['model_param'][loc]
    #     else: 
    #         sub_mps[key] = map_result['MAP']['model_param_intercept']

    # import pandas as pd
    # results_table = pd.DataFrame(index = model_param_names)
    # results_table['model_param_intercept'] = map_result['MAP']['model_param_intercept']
    # results_table['b_temp'] = map_result['MAP']['b_temp']
    # results_table['model_param_sigma'] = map_result['MAP']['model_param_sigma']


    # results_model_params = pd.DataFrame(data=map_result['MAP']['model_param'],
    #                                     columns=model_param_names,
    #                                     index=map_result['keys'])

    # b_temp = map_result['MAP']['b_temp']
    # model_param_intercept = map_result['MAP']['model_param_intercept']
    # b_temp[[2,8,19]] = 0
    # sub_mps = map_result['MAP']['model_param_intercept'] + 20*map_result['MAP']['b_temp']
    
    # sub_mps[1] = -2
    # sub_mps[9] = -10
    # sub_mps[10] = -2
    # sub_mps[14] = -2
    # sub_mps[18] = 5
    # sub_mps[21] = -5
    
    #filename = 'optimize_each_OHaraRudy_wMark_INa_0615_1316'
    #filename = 'optimize_each_OHaraRudy_wMark_INa_0617_1029'
    #filename = 'optimize_each_OHaraRudy_wMark_INa_0618_1100'
#    filename = 'optimize_OHaraRudy_wMark_INa_0621_0644'

    #filename = 'optimize_each_OHaraRudy_wMark_INa_0616_1011'
    # filepath = atrial_model.fit_data_dir+'/'+filename
    # with open(filepath+'.pickle','rb') as file:
    #     db = pickle.load(file)
    
#    sub_mps = db.x
#    sub_mps[17] = -2
    #sub_mps[0] = 1000
#    sub_mps[18] = 5
#    expected_err = db.fun
#    sub_mps = {key: res.x for key, res in db.res.items()}
#    expected_err = 0
#    for res in db.res.values():
#        expected_err += res.fun
#    print(expected_err)
#    sub_mps = res.x
    # sub_mps = np.array([ 0.99853418, -3.02850451, -0.52426541, -5.30894684,  0.19784067,
    #    -1.        ,  7.41353664,  2.23632115, -0.62730371,  8.36644525,
    #    -3.06243539,  7.67223533, -0.52998804, -1.52531707, -1.6720276 ,
    #     1.90942851,  0.91702488, -2.87504327,  1.16907694, -0.90701965,
    #     6.8550732 , -3.11420156,  9.01382321, -0.20200651,  0.27550289])
    
    #sub_mps[5] = -2
    #sub_mps[5] = -4
    #sub_mps[13] = 4
    
    #sub_mps[4] = 0
    #sub_mps[5] = -0.3
    #sub_mps[10] = 0
    
    #sub_mps[[8,9]] = [-0.5,-28]
#    shift = 25
    # sub_mps = np.array([ 3.09849539, -1.42216837,  0.49229459, -2.3151882+shift ,  1.1601244 ,
    #    -0.84186075,  0.02376837,  0.41865553, -1.01352075, -2.20848693-shift,
    #    -0.32747158, -1.36456941,  0.41532375,  0.72145421, -0.6569219 ,
    #     0.17191365,  0.74302502, -1.07247888,  1.39859451, -0.67806507,
    #     0.1872425-shift ,  3.71160024,  3.5079484 ,  2.28888796,  0.24868646])
    
#    sub_mps = {key: model_param[-1,i,:] for i,key in enumerate(sim_names) if key in keys_iin}
#    for key in sub_mps:
#        sub_mps[key][13] = 0
#    sub_mps[13] = 0
#    sub_mps[0] = 1.5
    
    with Pool() as proc_pool:
        proc_pool = None
        diff_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, sim_funcs=sim_fs, data=datas,\
                           pool=proc_pool)
        #diff_fn(sub_mps, exp_params=exp_parameters, 
                        #keys=keys_keep)
        for i in [5]:#range(44):
            sub_mps = np.copy(model_param[-1,i,:])
#            sub_mps[18] = 1.7
#            sub_mps[21] = -1
#            sub_mps[0] = 1
            res = diff_fn(sub_mps, exp_params=exp_parameters)
            for key in res:
                plt.figure(str(key))
                plt.scatter(datas[key][:,0], datas[key][:,1])
                plt.plot(datas[key][:,0], res[key], label=str(i))
        #print(res)
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
