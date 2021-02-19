#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:47:10 2020

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


plot_trace = False
plot_sim = False
plot_regressions = False
plot_pymc_diag = False

#from SaveSAP import savePlots,setAxisSizePlots
#sizes = {'deviance': (4, 4), 'deviance_zoomed': (4, 4), 'model_param_mean': (4, 4), 'b_temp': (4, 4), 'model_param_tau': (4, 4), 'model_params_legend': (2, 6), 'error_tau': (4, 4), 'sim_groups_legend': (2, 6), 'GNaFactor': (6, 2), 'baselineFactor': (6, 2), 'mss_tauFactor': (6, 2), 'mss_shiftFactor': (6, 2), 'tm_maxFactor': (6, 2), 'tm_tau1Factor': (6, 2), 'tm_shiftFactor': (6, 2), 'tm_tau2Factor': (6, 2), 'hss_tauFactor': (6, 2), 'hss_shiftFactor': (6, 2), 'thf_maxFactor': (6, 2), 'thf_shiftFactor': (6, 2), 'thf_tau1Factor': (6, 2), 'thf_tau2Factor': (6, 2), 'ths_maxFactor': (6, 2), 'ths_shiftFactor': (6, 2), 'ths_tau1Factor': (6, 2), 'ths_tau2Factor': (6, 2), 'Ahf_multFactor': (6, 2), 'jss_tauFactor': (6, 2), 'jss_shiftFactor': (6, 2), 'tj_maxFactor': (6, 2), 'tj_shiftFactor': (6, 2), 'tj_tau2Factor': (6, 2), 'tj_tau1Factor': (6, 2)}
#setAxisSizePlots(sizes)
#savePlots('R:/Hund/DanielGratz/atrial_model/plots/latest/OHaraRudy_wMark/', ftype='svg')
#setAxisSizePlots([(4,4)]*40)
#setAxisSizePlots((3,3))

atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau

burn_till = 1000#500#800#40000#34_000#2500#31_000 #35000
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
#    filename = 'test'


    
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
    db_post = db.warmup_posterior
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

    if plot_trace:
        
        
        
        c_by_f_type = [0,1,2,3,4,5,6,5,2,3,4,6,5,5,4,6,5,5,6,2,3,4,6,5,5]
        c_by_s_type = [0,1,2,2,3,3,3,3,4,4,5,5,5,5,6,6,6,6,9,7,7,8,8,8,8]

        stroke_by_s_type = ['-',
                            '-',
                            '-', '--',
                            '-', '--', '-.', ':',
                            '-', '--',
                            '-', '--', '-.', ':',
                            '-', '--', '-.', ':',
                            '-',
                            '-', '--',
                            '-', '--', '-.', ':']
        no_stroke = ['-']*len(stroke_by_s_type)
        
        c_by_mp = [paultcolors[c_scheme][i] for i in c_by_s_type]
        strokes = no_stroke#stroke_by_s_type
    
#         trace = 'deviance'
#         trace_data = db[trace][chain]
#         if stack:
#             trace_data_all = trace_data
#         else:
#             trace_data_all = [db[trace][ch] for ch in db[trace].keys()]
#             trace_data_all = np.concatenate(trace_data_all)
#         fig = plt.figure(trace)
#         ax = [fig.add_subplot(1,1,1)]
# #        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
#         ax[0].plot(trace_data_all, c=c_by_mp[0])
# #        ax[1].hist(trace_data[burn_till:], orientation='horizontal', color=c_by_mp[0])
        
#         fig = plt.figure(trace+'_zoomed')
#         ax = [fig.add_subplot(1,1,1)]
# #        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
#         ax[0].plot(trace_data[burn_till:], c=c_by_mp[0])

        trace = 'model_param_intercept'
        trace_data = db_post[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        ax[1].yaxis.set_visible(False)
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        sd = np.array(np.std(trace_data[burn_till:,:], axis=0))
        sorted_params = np.arange(len(sd))[np.argsort(-sd)]
        for i in sorted_params:
            ax[0].plot(trace_data[burn_till:,i], label=model_param_names[i], 
                       c=c_by_mp[i], linestyle=strokes[i])
#            _,_,hist = ax[1].hist(trace_data[:,i], orientation='horizontal', label=model_param_names[i])
            density = stats.gaussian_kde(trace_data[burn_till:,i])
            lower = np.min(trace_data[burn_till:,i])
            upper = np.max(trace_data[burn_till:,i])
            values = np.linspace(lower, upper, num=100)
            ax[1].plot(density(values), values, label=model_param_names[i], c=c_by_mp[i])
#            color = hist[0].get_facecolor()
#            ax[0].axhline(bounds[i, 0]+i/100, c=color, label=model_param_names[i]+'_lower')
#            ax[0].axhline(bounds[i, 1]+i/100, c=color, label=model_param_names[i]+'_upper')
        handles, labels = ax[0].get_legend_handles_labels()
#        ax[1].legend(handles, labels, frameon=False)
        
        trace = 'b_temp'
        trace_data = db_post[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        ax[1].yaxis.set_visible(False)
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        sd = np.array(np.std(trace_data[burn_till:,:], axis=0))
        sorted_params = np.arange(len(sd))[np.argsort(-sd)]
        for i in sorted_params:
            ax[0].plot(trace_data[burn_till:,i], label=model_param_names[i],
                       c=c_by_mp[i], linestyle=strokes[i])
            density = stats.gaussian_kde(trace_data[burn_till:,i])
            lower = np.min(trace_data[burn_till:,i])
            upper = np.max(trace_data[burn_till:,i])
            values = np.linspace(lower, upper, num=100)
            ax[1].plot(density(values), values, label=model_param_names[i], c=c_by_mp[i])
            #ax[1].hist(trace_data[:,i], orientation='horizontal', label=model_param_names[i])
        handles, labels = ax[0].get_legend_handles_labels()
#        ax[1].legend(handles, labels, frameon=False)
        
        trace = 'paper_eff'
        trace_data = db_post[trace][chain]
        for pmid, paper_idx in pmid2idx.items():
            fig = plt.figure(trace+' '+legend_labels[str(pmid)])
            ax = [fig.add_subplot(1,2,1)]
            ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
            ax[1].yaxis.set_visible(False)
            fig.subplots_adjust(hspace=0.1, wspace=0.05)
            sd = np.array(np.std(trace_data[burn_till:,paper_idx,:], axis=0))
            sorted_params = np.arange(len(sd))[np.argsort(-sd)]
            for i in sorted_params:
                ax[0].plot(trace_data[burn_till:,paper_idx,i], label=model_param_names[i], 
                           c=c_by_mp[i], linestyle=strokes[i])
    #            _,_,hist = ax[1].hist(trace_data[:,i], orientation='horizontal', label=model_param_names[i])
                density = stats.gaussian_kde(trace_data[burn_till:,paper_idx,i])
                lower = np.min(trace_data[burn_till:,paper_idx,i])
                upper = np.max(trace_data[burn_till:,paper_idx,i])
                values = np.linspace(lower, upper, num=100)
                ax[1].plot(density(values), values, label=model_param_names[i], c=c_by_mp[i])
    #            color = hist[0].get_facecolor()
    #            ax[0].axhline(bounds[i, 0]+i/100, c=color, label=model_param_names[i]+'_lower')
    #            ax[0].axhline(bounds[i, 1]+i/100, c=color, label=model_param_names[i]+'_upper')
            handles, labels = ax[0].get_legend_handles_labels()
    #        ax[1].legend(handles, labels, frameon=False)

        trace = 'paper_eff_sd'
        trace_data = db_post[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        ax[1].yaxis.set_visible(False)
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        sd = np.array(np.std(trace_data[burn_till:,:], axis=0))
        sorted_params = np.arange(len(sd))[np.argsort(-sd)]
        for i in sorted_params:
            ax[0].plot(trace_data[burn_till:,i], label=model_param_names[i], 
                       c=c_by_mp[i], linestyle=strokes[i])
#            _,_,hist = ax[1].hist(trace_data[:,i], orientation='horizontal', label=model_param_names[i])
            density = stats.gaussian_kde(trace_data[burn_till:,i])
            lower = np.min(trace_data[burn_till:,i])
            upper = np.max(trace_data[burn_till:,i])
            values = np.linspace(lower, upper, num=100)
            ax[1].plot(density(values), values, label=model_param_names[i], c=c_by_mp[i])
#            color = hist[0].get_facecolor()
#            ax[0].axhline(bounds[i, 0]+i/100, c=color, label=model_param_names[i]+'_lower')
#            ax[0].axhline(bounds[i, 1]+i/100, c=color, label=model_param_names[i]+'_upper')
        handles, labels = ax[0].get_legend_handles_labels()

        trace = 'model_param_sd'
        trace_data = db_post[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(1,2,1)]
        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        ax[1].yaxis.set_visible(False)
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        sd = np.array(np.std(trace_data[burn_till:,:], axis=0))
        sorted_params = np.arange(len(sd))[np.argsort(-sd)]
        for i in sorted_params:
            ax[0].plot(trace_data[burn_till:,i], label=model_param_names[i],
                        c=c_by_mp[i], linestyle=strokes[i])
            density = stats.gaussian_kde(trace_data[burn_till:,i])
            lower = np.min(trace_data[burn_till:,i])
            upper = np.max(trace_data[burn_till:,i])
            values = np.linspace(lower, upper, num=100)
            ax[1].plot(density(values), values, label=model_param_names[i], c=c_by_mp[i])
#            ax[1].hist(trace_sigma, orientation='horizontal', label=model_param_names[i])
        handles, labels = ax[0].get_legend_handles_labels()
#        ax[1].legend(handles, labels, frameon=False)

        fig = plt.figure('model_params_legend')
        order = [labels.index(mp_name) for mp_name in model_param_names]
        handles = [handles[i] for i in order]
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(handles, model_param_names, frameon=False)

        trace = 'error_sd'
        trace_data = db_post[trace][chain]
        fig = plt.figure(trace)
        ax = [fig.add_subplot(2,2,1)]
        ax.append(fig.add_subplot(2,2,2, sharey=ax[0]))
        ax.append(fig.add_subplot(2,2,3, sharex=ax[0]))
        ax.append(fig.add_subplot(2,2,4, sharex=ax[1], sharey=ax[2]))
        ax[0].xaxis.set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[1].xaxis.set_visible(False)
        ax[1].xaxis.set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].yaxis.set_visible(False)
        ax[3].yaxis.set_visible(False)
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        for i in range(trace_data.shape[1]):
            if group_names[i] == 'iv curve':
                ax0 = 0
                ax1 = 1
            else:
                ax0 = 2
                ax1 = 3
            
            color = paultcolors[c_scheme][i]
            ax[ax0].plot(trace_data[burn_till:,i], label=group_names[i], c=color)
            density = stats.gaussian_kde(trace_data[burn_till:,i])
            lower = np.min(trace_data[burn_till:,i])
            upper = np.max(trace_data[burn_till:,i])
            values = np.linspace(lower, upper, num=100)
            ax[ax1].plot(density(values), values, label=group_names[i], c=color)
            #ax[1].hist(trace_sigma, orientation='horizontal', label=group_names[i])
        handles, labels = ax[0].get_legend_handles_labels()
        handles1, labels1 = ax[2].get_legend_handles_labels()
        handles += handles1
        labels += labels1
        
        fig = plt.figure('sim_groups_legend')
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(handles, labels, frameon=False)
        #ax[1].legend(handles, labels, frameon=False)
        
        trace = 'model_param'
        trace_data = db_post[trace][chain]
        for param_i in range(trace_data.shape[2]):
            fig = plt.figure(model_param_names[param_i])
            ax = [fig.add_subplot()]
    #        ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
            sd = np.std(trace_data[burn_till:,:,param_i], axis=0)
            sorted_sims = np.arange(len(sd))[np.argsort(-sd)]
            for sim_i in sorted_sims:
                label = sim_groups[sim_i]
                group_i = list(group_names).index(label)
                ax[0].plot(trace_data[burn_till:,sim_i,param_i], c=paultcolors[c_scheme][group_i], label=label)#, label=model_param_names[sim_i])
    #            ax[1].hist(1/np.sqrt(trace_data[:,i]), orientation='horizontal', label=group_names[i])
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            #ax[0].legend(by_label.values(), by_label.keys(), frameon=False)
        
        
        trace = 'model_param_corr'
        trace_data = db.warmup_posterior[trace][chain]
        fig = plt.figure(trace)
        ax = np.empty((len(mp_locs),len(mp_locs)), dtype=object)
        avgtrace = np.mean(trace_data,axis=0)
        for i in range(len(mp_locs)):
            for j in range(len(mp_locs)):
                sharex = ax[i-1,j] if i-1 > 0 else None
                sharey = ax[i,j-1] if j-1 > 0 else None
                ax[i,j] = fig.add_subplot(*ax.shape,
                i*ax.shape[0]+j+1)#, 
                #sharex=sharex, 
                #sharey=sharey)
                ax[i,j].xaxis.set_visible(False)
                ax[i,j].spines['bottom'].set_visible(False)
                ax[i,j].yaxis.set_visible(False)
                ax[i,j].spines['left'].set_visible(False)
                if i <= j:
                    ax[i,j].imshow([[avgtrace[i,j]]], vmin=-1, vmax=1, cmap='bwr')
                    #ax[i,j].plot(trace_data[burn_till:, i,j])
                else:
                    density = stats.gaussian_kde(trace_data[burn_till:,i,j])
                    lower = np.min(trace_data[burn_till:,i,j])
                    upper = np.max(trace_data[burn_till:,i,j])
                    values = np.linspace(lower, upper, num=100)
                    ax[i,j].plot(density(values), values)
                    ax[i,j].axhline(c='black')
        #for i in range(len(mp_locs)):
        #    ax[i,0].yaxis.set_visible(True)
        #    ax[i,0].spines['left'].set_visible(True)
        #for j in range(len(mp_locs)):
        #    ax[-1,j].xaxis.set_visible(True)
        #    ax[-1,j].spines['bottom'].set_visible(True)
        
        # trace = 'biophys_res'
        # trace_data = db[trace][0]
        # fig = plt.figure(trace)
        # ax = [fig.add_subplot(1,2,1)]
        # ax[0].plot(trace_data)
        # ax.append(fig.add_subplot(1,2,2, sharey=ax[0]))
        # for i in range(trace_data.shape[1]):
        #     ax[1].hist(trace_data[:,i], orientation='horizontal')
        
        
    if plot_sim:
        try:
            with open(base_dir+'/plot_mcmc_cache.pickle','rb') as file:
                    cache = pickle.load(file)
        except FileNotFoundError:
            cache = {'overall':{
                'model_param': {},
                'results': {}},
            'individual':{
                'model_param': {},
                'results': {}}
            }
        # b_temp = np.zeros_like(mp_locs, dtype=float)
        # b_temp[[1,4,10,14,21]] = -0.7/10
        # b_temp[[3,6,9,11,15,20,22]] = 0.4
        # b_temp[[3]] = -0.4
        b_temp = np.median(db_post['b_temp'][chain][burn_till:], axis=0)
        # for i in range(db['b_temp'][chain].shape[1]):
        #     trace = db['b_temp'][chain][burn_till:, i]
        #     f_sig = np.sum(trace > 0)/len(trace)
        #     if not (f_sig < 0.05 or f_sig > 0.95):
        #         b_temp[i] = 0
        # b_temp[[2]] = 0
        # b_temp[[10]] = -0.2
        # b_temp[0] = 0.2
        intercept = np.median(db_post['model_param_intercept'][chain][burn_till:], axis=0)
        num_sims = len(key_frame)#sum(map(len, model_metadata.keys_all))
        model_params = {}
        fit_keys = key_frame.index #[key for keys in model_metadata.keys_all for key in keys]
        defined_keys = [key for keys in keys_all for key in keys]
        good_keys = [key for key in fit_keys if key in defined_keys]
        sim_fs_good = {key: sim_fs[key] for key in good_keys}
        for key in good_keys:
            temperature = exp_parameters.loc[key, 'temp ( K )'] -290.15
            b_temp_eff = b_temp * temperature
            sub_mps = intercept + b_temp_eff
            sub_mps[18] = 1.7
            model_params[key] = sub_mps
        model_params = {key: mp for key, mp in model_params.items() if key in good_keys}
        
        model_param_mean = np.median(db_post['model_param'][chain][burn_till:], axis=0)
#        model_param_mean[:,2] = 0
        model_param_sim_mean = {key: model_param_mean[k]
                                for k, key in enumerate(fit_keys)
                                if key in good_keys}
        
        use_cache = False
        for key, mp in model_params.items():
            if key in cache['overall']['model_param']:
                if not np.array_equal(cache['overall']['model_param'][key], mp):
                    use_cache = False
                    break
            else:
                use_cache = False
                break
        if use_cache:
            for key, mp in model_param_sim_mean.items():
                if key in cache['individual']['model_param']:
                    if not np.array_equal(cache['individual']['model_param'][key], mp):
                        use_cache = False
                        break
                else:
                    use_cache = False
                    break
        
        if use_cache:
            res_overall = cache['overall']['results']
            res_indiv = cache['individual']['results']
        else:
            with Pool() as proc_pool:
    #            proc_pool = None
                res_overall = calc_results(model_params, sim_funcs=sim_fs_good,\
                                          model_parameters_full=model_params_initial,\
                                mp_locs=mp_locs, data=datas,error_fill=0,\
                                pool=proc_pool)
                    
    
                res_indiv = calc_results(model_param_sim_mean, sim_funcs=sim_fs_good,\
                                          model_parameters_full=model_params_initial,\
                                mp_locs=mp_locs, data=datas,error_fill=0,\
                                pool=proc_pool)
            cache['overall'] = {'model_param': model_params,
                                'results': res_overall}
            cache['overall'] = {'model_param': model_param_sim_mean,
                                'results': res_indiv}
            with open(base_dir+'/plot_mcmc_cache.pickle','wb') as file:
                pickle.dump(cache, file)
        
        text_leg_lab = True

        handles, labels = [], []
        
        for key in good_keys:
            figname = exp_parameters.loc[key, 'Sim Group']
            figname = figname if not pd.isna(figname) else 'Missing Label'
            if text_leg_lab:
                paper_key = key[0].split('_')[0]
                color = list(legend_labels.keys()).index(paper_key)
                color = paultcolors[c_scheme][color]
                label = legend_labels[paper_key]
            else:
                label = key
                color = None
                
            if figname == 'iv curve':
                plot_data = datas[key].copy()
                plot_data[:,1] = np.sign(plot_data[:,1])*np.square(plot_data[:,1])
                
                sim_overall = np.sign(res_overall[key])*np.square(res_overall[key])
                sim_individ = np.sign(res_indiv[key])*np.square(res_indiv[key])
            else:
                plot_data = datas[key]
                sim_overall = res_overall[key]
                sim_individ = res_indiv[key]
                
            fig = plt.figure(figname + " overall fit")
            ax = fig.get_axes()
            if len(ax) == 0:
                fig.add_subplot()
                ax = fig.get_axes()
            ax = ax[0]
            ax.plot(plot_data[:,0], sim_overall, label=label, color=color)
            ax.scatter(plot_data[:,0], plot_data[:,1], color=color)
            if text_leg_lab:
                handles_labels = ax.get_legend_handles_labels()
                handles += handles_labels[0]
                labels += handles_labels[1]
            else:
                ax.legend(frameon=False)
            
            
            fig = plt.figure(figname + " individual fit")
            ax = fig.get_axes()
            if len(ax) == 0:
                fig.add_subplot()
                ax = fig.get_axes()
            ax = ax[0]
            ax.plot(plot_data[:,0], sim_individ, label=label, color=color)
            ax.scatter(plot_data[:,0], plot_data[:,1], color=color)
            
        if text_leg_lab:
            idx = [labels.index(name) for name in legend_labels.values()]
            handles = [handles[i] for i in idx]
            labels = [labels[i] for i in idx]
            fig = plt.figure('sim_fits_legend')
            ax = fig.add_subplot(1,1,1)
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.legend(handles, labels, frameon=False)
            
            
    
    if plot_regressions:
        group_is = np.array([group_names.index(sim_grp) for sim_grp in sim_groups])
        temperatures = [exp_parameters.loc[key, 'temp ( K )'] -290.15 for key in sim_names]
        intercept = np.median(db['model_param_mean'][chain][burn_till:], axis=0)
        b_temp = np.median(db['b_temp'][chain][burn_till:], axis=0)
        model_params = np.median(db['model_param'][chain][burn_till:], axis=0)
        reg_temps = np.arange(min(temperatures),max(temperatures)+1)
        reg_res = intercept + reg_temps[...,None] *b_temp
        for i, name in enumerate(model_param_names):
            plt.figure(name)
            plt.scatter(temperatures, model_params[:,i], c=np.array(paultcolors[c_scheme])[group_is])
            plt.plot(reg_temps, reg_res[:,i])
    
    if plot_pymc_diag:
        try:
            import pymc
            from pymc import Matplot
            from pymc import diagnostics
#            db = pymc.database.pickle.load(trace_file)
            geweke_scores = diagnostics.geweke(db['model_param_tau'][0][:,2])
            Matplot.geweke_plot(geweke_scores, name="Gweke Scores")
            
        except ImportError:
            print("pymc not installed")
    
