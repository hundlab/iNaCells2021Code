#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:16:54 2020

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
from atrial_model.run_sims import calc_results, SimRunner
from atrial_model.iNa.define_sims import keys_all, exp_parameters, data, sim_fs, datas
from atrial_model.iNa.model_setup import model, mp_locs, sub_mps, sub_mp_bounds, model_params_initial, model_param_names
from atrial_model.iNa.stat_model_3 import key_frame

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy import integrate
import pickle


keys_keep = []

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# # #iv curve
keys_iin = [
# ('1323431_1',	'Dataset B'), ('1323431_3',	'Dataset A 2'),
# ('1323431_3',	'Dataset A 20'), ('1323431_3',	'Dataset A 5'),
# ('1323431_4',	'Dataset B Control'),
# #('7971163_1', 'Dataset'),
 
('8928874_7',	'Dataset C day 1'), ('8928874_7',	'Dataset C day 3'),
('8928874_7',	'Dataset C day 5'), ('8928874_7',	'Dataset C fresh'),
('21647304_1',	'Dataset B Adults'), ###('21647304_1', 'Dataset B Pediatrics'),
('12890054_3', 'Dataset C Control'), ('12890054_3',	'Dataset D Control'),
('12890054_5', 'Dataset C Control'), ###('12890054_5',	'Dataset D Control'),
('23341576_2', 'Dataset C Control')
]
keys_keep += keys_iin


# ##activation normalized to driving force
# keys_iin = [
#             ('1323431_2',	'Dataset'),\
#             ('8928874_7',	'Dataset D fresh'), ('8928874_7',	'Dataset D day 1'),\
#             ('8928874_7',	'Dataset D day 3'), ('8928874_7',	'Dataset D day 5'),\
#             ('21647304_3',	'Dataset A Adults'), ('21647304_3',	'Dataset A Pediatrics')
# ]
# keys_keep += keys_iin




# # I2/I1 Recovery
# keys_iin = [('1323431_8', 'Dataset A -140'), ('1323431_8',	'Dataset A -120'),\
#               ('1323431_8',	'Dataset A -100'),\
#             ('21647304_3',	'Dataset C Adults'), ('21647304_3',	'Dataset C Pediatrics'),\
#             ('8928874_9', 'Dataset fresh'), ('8928874_9', 'Dataset day 1'),\
#             ('8928874_9', 'Dataset day 3'), ('8928874_9', 'Dataset day 5')
# ]
# keys_keep += keys_iin


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
# keys_iin = [
#                   ('7971163_4', 'Dataset 256ms'), ('7971163_4', 'Dataset 512ms'),\

#               ('8928874_8',	'Dataset C fresh'), ('8928874_8',	'Dataset C day 1'),\
#               ('8928874_8',	'Dataset C day 3'), ('8928874_8',	'Dataset C day 5'),
# ('21647304_3',	'Dataset B Adults'), ('21647304_3',	'Dataset B Pediatrics')
#               ]
# # #
# keys_keep += keys_iin


# #inactivation normalized to first
# keys_iin = [('7971163_5',	'Dataset A -65'), ('7971163_5',	'Dataset A -75'),\
#             ('7971163_5',	'Dataset A -85'), ('7971163_5',	'Dataset A -95'),\
#             ('7971163_5',	'Dataset A -105')
#             ]
# keys_keep += keys_iin


#idealized current traces
#keys_iin = [#('21647304_2', 'Dataset B Adults'), ('21647304_2', 'Dataset B Pediactric'),
              #('8928874_8',	'Dataset D fresh'),# ('8928874_8',	'Dataset D day 1'),\
              #('8928874_8',	'Dataset D day 3'),# ('8928874_8',	'Dataset D day 5'),
#              ('7971163_3',	'Dataset C')
#]
#keys_keep += keys_iin





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


## Temperature dependant effects ##
test_voltages = np.arange(-100,30,1)
voltages = np.array([[-140, v] for v in test_voltages])
durs = np.ones((len(test_voltages), 2))*10
solver = partial(integrate.solve_ivp, method='BDF')
simfs_temp = {}
for temperature in [0, 10, 20]:
    sim_param = {'TEMP': temperature+290.15, 'naO': 140.0, 'naI':7.0}
    sim_f = SimRunner(model, voltages, durs, sim_param, peakCurr, None, 
                dt=0.005, solver=solver)
    simfs_temp[str(temperature)] = sim_f

## Variability ##
num_draws = 50
voltages = np.array([[-140, v] for v in test_voltages])
durs = np.ones((len(test_voltages), 2))*10
solver = partial(integrate.solve_ivp, method='BDF')
sim_param = {'TEMP': 310.15, 'naO': 140.0, 'naI':7.0}
simfs_rand = {i: SimRunner(model, voltages, durs, sim_param, peakCurr, None, 
                dt=0.005, solver=solver) for i in range(num_draws)}


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
atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = True #diff
atrial_model.run_sims_functions.plot3 = False #tau

if __name__ == '__main__':
    class ObjContainer():
        pass
    
    filename = 'MAP_OHaraRudy_wMark_INa_0803_1107'
    
    #filename = 'MAP_OHaraRudy_wMark_INa_0801_1358'

    base_dir = atrial_model.fit_data_dir+'/'
    with open(base_dir+'/'+filename+'.pickle','rb') as file:
        map_result = pickle.load(file)
    keys_frame = key_frame(keys_keep_grps, exp_parameters)
    sub_mps_overall = {}
    b_temp = map_result['MAP']['b_temp']
    model_param_intercept = map_result['MAP']['model_param_intercept']
    b_temp[[2,8,19]] = 0
    for key in keys_frame.index:
        # if key in map_result['keys']:
        #     loc = map_result['keys'].index(key)
        #     sub_mps[key] = map_result['MAP']['model_param'][loc]
        # else:
            TEMP = keys_frame.loc[key, 'temp ( K )'] -290.15
            sub_mps_overall[key] = model_param_intercept +TEMP*b_temp
    sub_mps_indiv = {}      
    for key in keys_frame.index:
        loc = map_result['keys'].index(key)
        sub_mps_indiv[key] = map_result['MAP']['model_param'][loc]

            
    sub_mps_temp = {key: model_param_intercept +float(key)*b_temp for key in simfs_temp}
    
    sub_mps_rand = {key: np.random.normal(
                        loc=model_param_intercept +20*b_temp,
                        scale=map_result['MAP']['model_param_sigma'])
                    for key in simfs_rand}
    

    import pandas as pd
    results_table = pd.DataFrame(index = model_param_names)
    results_table['model_param_intercept'] = map_result['MAP']['model_param_intercept']
    results_table['b_temp'] = map_result['MAP']['b_temp']
    results_table['model_param_sigma'] = map_result['MAP']['model_param_sigma']

    # b_temp = map_result['MAP']['b_temp']
    # model_param_intercept = map_result['MAP']['model_param_intercept']
    # b_temp[[2,8,19]] = 0
    # sub_mps = map_result['MAP']['model_param_intercept'] + 20*map_result['MAP']['b_temp']

    with Pool() as proc_pool:
        proc_pool = None
        results_data = calc_results(sub_mps_overall, exp_params=exp_parameters, 
                            model_parameters_full=model_params_initial,\
                            mp_locs=mp_locs, sim_funcs=sim_fs, data=datas,\
                            pool=proc_pool)
            
        results_data_indiv = calc_results(sub_mps_indiv, exp_params=exp_parameters, 
                            model_parameters_full=model_params_initial,\
                            mp_locs=mp_locs, sim_funcs=sim_fs, data=datas,\
                            pool=proc_pool)
            
        temperature_data = calc_results(sub_mps_temp, exp_params=None, 
                            model_parameters_full=model_params_initial,\
                            mp_locs=mp_locs, sim_funcs=simfs_temp, data=None,\
                            pool=proc_pool)
            
        # random_data = calc_results(sub_mps_rand, exp_params=None, 
        #                     model_parameters_full=model_params_initial,\
        #                     mp_locs=mp_locs, sim_funcs=simfs_rand, data=None,\
        #                     pool=proc_pool)

import matplotlib
import matplotlib.cm as cm
import colorcet

norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
cmap = colorcet.m_linear_kry_0_97_c73
color_map = cm.ScalarMappable(norm=norm, cmap=cmap)

font = {'family':'sans-serif','sans-serif':['Arial'], 'size': 10}
plt.rc('font',**font)
plt.rcParams['lines.linewidth']=4
plt.rcParams['axes.spines.right']=False
plt.rcParams['axes.spines.top']=False
legend_properties = {'weight':'bold'}

lines = []
labels = []
fig = plt.figure('iv curve')
ax_iv = fig.add_subplot(1,1,1)
for key, vals_sim in results_data.items():
    sub_dat = datas[key]
    temp_color = color_map.to_rgba(keys_frame.loc[key, 'temp ( K )']-290.15)
    line_artist = ax_iv.plot(sub_dat[:,0], vals_sim, label=key, c=temp_color)
    ax_iv.scatter(sub_dat[:,0], sub_dat[:,1], c=temp_color)
    lines += line_artist
    labels.append(key)
#plt.colorbar(color_map)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_axis_off()
colorbar = fig.colorbar(color_map)
colorbar.set_ticks([0, 5, 10, 15, 20])
colorbar.set_ticklabels(['17', '22', '27', '32', '37'])
lgnd = plt.legend(lines, labels, prop=legend_properties, frameon=False)


fig = plt.figure('iv curve individual')
ax = fig.add_subplot(1,1,1, sharey=ax_iv)
for key, vals_sim in results_data_indiv.items():
    sub_dat = datas[key]
    temp_color = color_map.to_rgba(keys_frame.loc[key, 'temp ( K )']-290.15)
    ax.plot(sub_dat[:,0], vals_sim, label=key, c=temp_color)
    ax.scatter(sub_dat[:,0], sub_dat[:,1], c=temp_color)
#plt.colorbar(color_map)
#plt.legend(prop=legend_properties, frameon=False)

fig = plt.figure('Temperature Effects')
ax = fig.add_subplot(1,1,1)
for key, vals_sim in temperature_data.items():    
    temp_color = color_map.to_rgba(float(key))
    ax.plot(test_voltages, vals_sim, label=key, c=temp_color)
#plt.colorbar(color_map)

fig = plt.figure('Population')
mean_iv = np.zeros(len(test_voltages))
ax2 = fig.add_subplot(1,1,1, sharey=ax)
for key, vals_sim in random_data.items():
    temp_color = list(color_map.to_rgba(20))
    temp_color[3] = 0.5
    ax2.plot(test_voltages, vals_sim, c=temp_color)
    mean_iv += vals_sim
mean_iv /= num_draws
ax2.plot(test_voltages, mean_iv, c='black')

