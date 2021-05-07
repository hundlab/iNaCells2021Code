#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:49:06 2021

@author: grat05
"""



import sys
sys.path.append('../../../')

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import arviz as az

import atrial_model
from atrial_model.iNa.define_sims import exp_parameters
from atrial_model.iNa.model_setup import model_param_names
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_results
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all
from atrial_model.iNa.model_setup import model_params_initial, mp_locs, sub_mps, model


from multiprocessing import Pool
from functools import partial
import os


from SaveSAP import savePlots,setAxisSizePlots
sizes = {'logp': (3.5, 3.5), 'model_param_intercept': (3.5, 3.5), 'b_temp': (3.5, 3.5),
         'paper_eff Sakakibara et al': (3.5, 3.5), 'paper_eff Cai et al': (3.5,3.5),
         'paper_eff Feng et al': (3.5, 3.5), 'paper_eff Schneider et al': (3.5, 3.5),
         'paper_eff Lalevée et al': (3.5, 3.5), 'paper_eff Wettwer et al': (3.5, 3.5),
         'paper_eff_sd': (3.5, 3.5), 'model_param_sd': (3.5, 3.5), 
         'model_params_legend': (2, 6), 'error_sd': (3.5, 3.5), 'sim_groups_legend': (2, 6), 
         'GNaFactor': (3.5, 2), 'baselineFactor': (3.5, 2), 'mss_tauFactor': (3.5, 2),
         'mss_shiftFactor': (3.5, 2), 'tm_maxFactor': (3.5, 2), 'tm_tau1Factor': (3.5, 2),
         'tm_shiftFactor': (3.5, 2), 'tm_tau2Factor': (3.5, 2), 'hss_tauFactor': (3.5, 2),
         'hss_shiftFactor': (3.5, 2), 'thf_maxFactor': (3.5, 2), 'thf_shiftFactor': (3.5, 2),
         'thf_tau1Factor': (3.5, 2), 'thf_tau2Factor': (3.5, 2), 'ths_maxFactor': (3.5, 2),
         'ths_shiftFactor': (3.5, 2), 'ths_tau1Factor': (3.5, 2), 'ths_tau2Factor': (3.5, 2),
         'Ahf_multFactor': (3.5, 2), 'jss_tauFactor': (3.5, 2), 'jss_shiftFactor': (3.5, 2),
         'tj_maxFactor': (3.5, 2), 'tj_shiftFactor': (3.5, 2), 'tj_tau2Factor': (3.5, 2),
         'tj_tau1Factor': (3.5, 2),
         'model_param_corr': (6,6)}
#setAxisSizePlots(sizes)
#savePlots('R:/Hund/DanielGratz/atrial_model/plots/latest/plots/', ftype='svg')
#setAxisSizePlots([(3.5,3.5)]*40)
#setAxisSizePlots((3,3))

atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau

burn_till =0#500#2000#500#800#40000#34_000#2500#31_000 #35000
max_loc = 2688
chain = 0#7
#burn_till = 60000
stack = False

if __name__ == '__main__':
    class ObjContainer():
        pass
    #filename = 'mcmc_OHaraRudy_wMark_INa_0824_1344'
    
    #filename = 'mcmc_OHaraRudy_wMark_INa_0919_1832_sc'
    filename = 'mcmc_OHaraRudy_wMark_INa_0924_1205'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0831_1043_sc'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0829_1748'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0829_1334'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0827_1055'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0826_0958'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0821_1132'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0702_1656'
    
    filename = 'mcmc_OHaraRudy_wMark_INa_1012_1149'
    
    filename = 'mcmc_OHaraRudy_wMark_INa_1202_1906'
    filename = 'mcmc_OHaraRudy_wMark_INa_1204_1201'
    filename = 'mcmc_OHaraRudy_wMark_INa_1205_1323'
    filename = 'mcmc_OHaraRudy_wMark_INa_1213_1353'
    filename = 'mcmc_OHaraRudy_wMark_INa_1216_1109'
    filename = 'mcmc_OHaraRudy_wMark_INa_1222_1754'
    filename = 'mcmc_OHaraRudy_wMark_INa_0109_1802'
    
#    filename = 'mcmc_OHaraRudy_wMark_INa_0121_1201'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0121_1450'
    filename = 'mcmc_OHaraRudy_wMark_INa_0121_1531'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0122_1447'
    filename = 'mcmc_OHaraRudy_wMark_INa_0122_1607'
    filename = 'mcmc_OHaraRudy_wMark_INa_0125_1328'
    filename = 'mcmc_OHaraRudy_wMark_INa_0125_1346'
    filename = 'mcmc_OHaraRudy_wMark_INa_0127_1333'
    filename = 'mcmc_OHaraRudy_wMark_INa_0127_1525'
    filename = 'mcmc_OHaraRudy_wMark_INa_0128_1623'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0129_1549'
    filename = 'mcmc_OHaraRudy_wMark_INa_0129_1601'
    filename = 'mcmc_OHaraRudy_wMark_INa_0215_0722'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0319_1706'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0322_1334'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0322_1603'
# #    filename = 'mcmc_OHaraRudy_wMark_INa_0323_0955'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0323_1628'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0324_1010'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0324_1609'
# #    filename = 'test'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0325_1044'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0325_1300'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0325_1518'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0325_2128'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0326_1753'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0326_1721'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0326_2028'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0326_2030'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0329_0817'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0326_2030'
    
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0329_1005'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0329_1730'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0330_0906'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0330_1020'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0330_1130'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0330_1212'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0330_1428'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0331_0817'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0331_1057'
#    # filename = 'mcmc_OHaraRudy_wMark_INa_0402_1513'
#     #filename = 'mcmc_OHaraRudy_wMark_INa_0407_1328'
#     filename = 'mcmc_OHaraRudy_wMark_INa_0408_1723'

    #filename = 'mcmc_OHaraRudy_wMark_INa_0215_0722'
   
    # filename = 'mcmc_OHaraRudy_wMark_INa_0106_1257'
    # filename = 'mcmc_OHaraRudy_wMark_INa_0106_1547'
    # filename = 'mcmc_OHaraRudy_wMark_INa_0107_1145'
    # filename = 'mcmc_OHaraRudy_wMark_INa_0108_0941'
    # filename = 'mcmc_OHaraRudy_wMark_INa_0108_1108'

    
    # filename = 'mcmc_OHaraRudy_wMark_INa_1223_1730'
    # filename = 'mcmc_OHaraRudy_wMark_INa_1228_1411'
    # filename = 'mcmc_OHaraRudy_wMark_INa_1230_1217'
    # filename = 'mcmc_OHaraRudy_wMark_INa_0101_1124'
    # filename = 'mcmc_OHaraRudy_wMark_INa_0104_1052'
    # filename = 'mcmc_OHaraRudy_wMark_INa_0105_1517'


    #filename = 'mcmc_OHaraRudy_wMark_INa_1229_1140'

#    filename = 'mcmc_OHaraRudy_wMark_INa_1226_1624'

    
    #filename = 'mcmc_OHaraRudy_wMark_INa_0627_1152'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0626_0808'
    #filename = 'mcmc_OHaraRudy_wMark_INa_0606_0047'
    #filename = 'mcmc_Koval_0601_1835'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0603_1051'
    #filename = 'mcmc_OHara_0528_1805'
#    filename = 'mcmc_OHaraRudy_wMark_INa_0528_1833'
    #filename = 'mcmc_Koval_0526_1728'
    #filename = 'mcmc_Koval_0519_1830'

    base_dir = atrial_model.fit_data_dir+'/'
    with open(base_dir+'/'+filename+'.pickle','rb') as file:
        db_full = pickle.load(file)
    db = db_full['trace']
    db_post = db.warmup_posterior#posterior#
    # with open(base_dir+'/'+filename+'_metadata.pickle','rb') as file:
    #     model_metadata = pickle.load(file)
    # with open(base_dir+model_metadata.trace_pickel_file,'rb') as file:
    #     db = pickle.load(file)
    # if db['_state_']['sampler']['status'] == 'paused':
    #     current_iter = db['_state_']['sampler']['_current_iter']
    #     current_iter -= db['_state_']['sampler']['_burn']
    #     for key in db.keys():
    #         if key != '_state_':
    #             db[key][chain] = db[key][chain][:current_iter]
    # if stack:
    #     for key in db.keys():
    #         if key != '_state_' and key != 'AdaptiveSDMetropolis_model_param_adaptive_scale_factor'\
    #         and key != 'biophys_res':
    #             stacked = [db[key][chain] for chain in db[key]]
    #             db[key] = [np.concatenate(stacked)]
                    

    key_frame = db_full['key_frame']
    sim_groups = key_frame['Sim Group']
    group_names = key_frame['Sim Group'].unique()
    sim_names = key_frame.index
    
    pmid2idx = {}
    curr_idx = 0
    for key in key_frame.index:
        pubmed_id = int(key[0].split('_')[0])
        if not pubmed_id in pmid2idx:
            pmid2idx[pubmed_id] = curr_idx
            curr_idx += 1
    # group_names = []
    # sim_groups = []
    # sim_names = []
    # for key_group in db_full['keys_all']:
    #     group_names.append(exp_parameters.loc[key_group[0], 'Sim Group'])
    #     for key in key_group:
    #         sim_names.append(key)
    #         sim_groups.append(group_names[-1])
    # bounds = np.array(db_full['param_bounds'])[db_full['mp_locs'], :]


    model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
    model_param_index = np.tile(model_param_index, (len(key_frame),1))
    paper_idx = {}
    curr_idx = 0
    sim_idx = []
    for key in key_frame.index:
        pubmed_id = int(key[0].split('_')[0])
        if not pubmed_id in paper_idx:
            paper_idx[pubmed_id] = curr_idx
            curr_idx += 1
        sim_idx.append(paper_idx[pubmed_id])
    sim_paper_idx = np.array(sim_idx)
    model_param_intercept = db_post['model_param_intercept'][0]
    b_temp = db_post['b_temp'][0]
    temperature_arr = np.array(key_frame['temp ( K )'], dtype=float) -290.15
    paper_eff = db_post['paper_eff'][0]
    mean = np.array(model_param_intercept)[:,model_param_index] +\
        np.array(b_temp)[:,model_param_index]*temperature_arr[...,None] +\
        np.array(paper_eff)[:,sim_paper_idx,:]
    model_param_sd = db_post['model_param_sd'][0]

    from SaveSAP import paultcolors
    c_scheme = 'muted'

    legend_labels = {
        '8928874': 'Feng et al',
        '21647304': 'Cai et al',
        '12890054': 'Lalevée et al',
        '23341576': 'Wettwer et al',
        '1323431': 'Sakakibara et al',
        '7971163': 'Schneider et al'
        }

    mp_sets = dict(
        ss_tau = [2,8,19],
        t_tau = [5,7,12,13,16,17,23,24],
        t_max = [1,4,10,14,21],
        shifts = [3,6,9,11,15,20,22],
        misc = [0,18]
    )

    for set_name, set_locs in mp_sets.items():
        fig = plt.figure(set_name)
        ax = fig.add_subplot()

        trace_data = db_post['model_param_sd'][chain, max_loc]
        ax.scatter(model_param_names[set_locs], trace_data[set_locs])
        
        trace_data = db_post['paper_eff_sd'][chain, max_loc]
        ax.scatter(model_param_names[set_locs], trace_data[set_locs])

        for label in ax.get_xticklabels():
            label.set_rotation(-45)
            label.set_horizontalalignment('left')
            
    fig = plt.figure('error_sd')
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[3, 1], height_ratios=[1,3])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    ax = [fig.add_subplot(spec[1,0])]
    ax.append(fig.add_subplot(spec[0,1]))
    trace_data = db_post['error_sd'][chain, max_loc]
    smaller = [0,1,2,3,5]
    ax[0].scatter(group_names[smaller], trace_data[smaller], color=paultcolors[c_scheme][2])
    ax[1].scatter(group_names[4], trace_data[4], color=paultcolors[c_scheme][2])
    ax[1].xaxis.set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax_o = fig.add_subplot(spec[0,0], sharey=ax[1])
    ax_o.xaxis.set_visible(False)
    ax_o.spines['bottom'].set_visible(False)
    ax_o = fig.add_subplot(spec[1,1], sharex=ax[1])
    ax_o.yaxis.set_visible(False)
    ax_o.spines['left'].set_visible(False)
    for label in ax_o.get_xticklabels():
            label.set_rotation(-45)
            label.set_horizontalalignment('left')
    

    for label in ax[0].get_xticklabels():
            label.set_rotation(-45)
            label.set_horizontalalignment('left')
        
        
        
        
    trace = 'model_param_corr'
    trace_data = db_post[trace][chain]
    fig = plt.figure(trace)
    ax = np.empty((len(mp_locs),len(mp_locs)), dtype=object)
    avgtrace = trace_data[max_loc]
    for i in range(len(mp_locs)):
        for j in range(len(mp_locs)):
            sharex = ax[i-1,j] if i-1 > 0 else None
            sharey = ax[i,j-1] if j-1 > 0 else None
            ax[i,j] = fig.add_subplot(*ax.shape,
            i*ax.shape[0]+j+1, 
            #sharex=sharex, 
            sharey=sharey)
            ax[i,j].xaxis.set_visible(False)
            ax[i,j].spines['bottom'].set_visible(False)
            ax[i,j].yaxis.set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            ax[i,j].set_ylim(top=1, bottom=-1)
            if i >= j:
                img = ax[i,j].imshow([[avgtrace[i,j]]], vmin=-1, vmax=1, cmap='bwr')
                #ax[i,j].plot(trace_data[burn_till:, i,j
                
    fig = plt.figure('model_corr_legend')
    ax = fig.add_subplot(1,1,1)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.colorbar(img, drawedges=False)
#setAxisSizePlots([(1,3.5), (3.5,3.5), (2,3.5), (3.5,3.5), (1,3.5), (3.5,3.5)])