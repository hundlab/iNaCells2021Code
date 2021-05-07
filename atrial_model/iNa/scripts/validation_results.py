#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:51:52 2020

@author: grat05
"""

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

import atrial_model
from atrial_model.data_loader import load_data_parameters, all_data
from convert_to_hdf5 import load_or_convert

import numpy as np
import copy
import matplotlib.pyplot as plt

np.seterr(all='ignore')

# load data from papers
exp_parameters, data = load_data_parameters('/params old/INa_old.xlsx','validation', data=all_data)

# load simulation data
datadir = atrial_model.fit_data_dir+'/'

foldername = "data022221-1402"

#traces, measures = load_or_convert(datadir, foldername)

#dvdt = np.array([measures[i][('vOld', 'maxderiv')].iloc[-1] for i in range(len(measures))])

# single_cell_data = {
#     "Simulated": dvdt,
#     "30356673_2": data[('30356673_2',	'Dataset B dVdt max Ctl')][:,1],
#     '26121051_5': data[('26121051_5',	'Dataset Ad Control')][0,1],
#     }
# fig = plt.figure("Single Cell Validation")
# ax = fig.add_subplot()
# boxplot = ax.boxplot(list(single_cell_data.values()), showmeans=True, labels=list(single_cell_data.keys()))
# for comp in ['caps', 'whiskers']:
#     for graphic_loc in range(4,6):
#         boxplot[comp][graphic_loc].set_visible(False)
# for comp in ['boxes', 'medians']:
#     boxplot[comp][2].set_visible(False)

from SaveSAP import paultcolors
c_scheme = 'muted'
colors = paultcolors[c_scheme]

legend_labels = {
    '30356673': 'Molina et al',
    '26121051': 'Poulet et al',
    '2112042': 'Le Grand et al',
    '23341576': 'Wettwer et al'
    }

fig = plt.figure("Validation")
ax = fig.add_subplot()


foldername = "data031821-1409"
traces, measures = load_or_convert(datadir, foldername)
cl = np.array([measures[i][('vOld', 'cl')] for i in range(len(measures))])
cl = np.round(np.mean(cl,axis=0), 0)
dvdt = np.array([measures[i][('vOld', 'maxderiv')] for i in range(len(measures))])
last_beat = list(np.diff(cl, axis=0).nonzero()[0][1:])
#last_beat.append(len(cl)-1)
last_beat = np.array(last_beat)
sim_dvdt = dvdt[:, last_beat]
sim_mean_dvdt = np.mean(sim_dvdt, axis = 0)
sim_sd_dvdt = np.std(sim_dvdt, axis = 0)

dat_name = 'Simulated Single Cells'
ax.fill_between(cl[last_beat], sim_mean_dvdt-sim_sd_dvdt, sim_mean_dvdt+sim_sd_dvdt, alpha=0.5, color=colors[0])
ax.plot(cl[last_beat], sim_mean_dvdt, label=dat_name, c=colors[0], zorder=1)

dat_name = ('30356673_2',	'Dataset B dVdt max Ctl')
leg_name = legend_labels[dat_name[0].split("_")[0]]
exp_dat = data[dat_name][:,1]
ax.errorbar([1000], np.mean(exp_dat), np.std(exp_dat), c=colors[2], zorder=1)
ax.scatter([1000], np.mean(exp_dat), label=leg_name, c=colors[2], marker='v', zorder=1)

dat_name = ('26121051_5',	'Dataset Ad Control')
leg_name = legend_labels[dat_name[0].split("_")[0]]
exp_dat = data[dat_name][0,1]
ax.scatter([1001], exp_dat, label=leg_name, c=colors[3], marker='s')



# dat_name = ('2112042_2', 'Dataset Control A')
# leg_name = legend_labels[dat_name[0].split("_")[0]] + " A"
# exp_dat = data[dat_name].T
# plt.scatter(*exp_dat, label=leg_name)



dat_name = ('23341576_1',	'Dataset Ba Control')
leg_name = legend_labels[dat_name[0].split("_")[0]]
exp_dat = data[dat_name][[3,10],1].T # 19, 28
ax.scatter([999, 501], exp_dat, label=leg_name, c=colors[5], marker='*') #, 333, 250



dat_name = 'Simulated Fiber'
foldername = "data031921-0813"
traces, measures = load_or_convert(datadir, foldername)
cl = np.round(np.diff(measures[0][((0,0) , 'vOld', 'maxt')]), 0)
dvdt = np.array([measures[i]
                 .xs(('vOld', 'maxderiv'), level=('Variable', 'Property'), axis='columns')
                 .iloc[:, 2:9].mean(axis='columns') 
                 for i in range(len(measures))])

last_beat = list(np.diff(cl, axis=0).nonzero()[0])
last_beat.append(len(cl)-1)
last_beat = np.array(last_beat)

sim_dvdt = dvdt[:, last_beat]
sim_mean_dvdt = np.mean(sim_dvdt, axis = 0)
sim_sd_dvdt = np.std(sim_dvdt, axis = 0)

ax.fill_between(cl[last_beat], sim_mean_dvdt-sim_sd_dvdt, sim_mean_dvdt+sim_sd_dvdt, alpha=0.5, color=colors[1])
ax.plot(cl[last_beat], sim_mean_dvdt, label=dat_name, c=colors[1], zorder=1)

dat_name = ('2112042_2', 'Dataset Control B')
leg_name = legend_labels[dat_name[0].split("_")[0]] + " B"
exp_dat = data[dat_name].T
plt.scatter(*exp_dat, label=leg_name, c=colors[4], marker='P')
ax.legend()

foldername = "data042121-1141"
traces, measures = load_or_convert(datadir, foldername)
fig_full = plt.figure("Single Cell Traces")
ax_full = fig_full.add_subplot()
fig_sub = plt.figure("Single Cell Traces Sub")
ax_sub = fig_sub.add_subplot()
ax_sub.xaxis.set_visible(False)
ax_sub.spines['bottom'].set_visible(False)
ax_sub.yaxis.set_visible(False)
ax_sub.spines['left'].set_visible(False)
for i in range(20):
    trace = traces[i]
    use = (trace['t'] > 498990) & (trace['t'] < 499500)
    t_min = 499_000
    ax_full.plot(trace['t'][use] -t_min ,traces[i]['vOld'][use], alpha=0.5, color=colors[0])
    use = use & (trace['t'] < 499030)
    ax_sub.plot(trace['t'][use] -t_min ,traces[i]['vOld'][use], alpha=0.5, color=colors[0])
    

foldername = "data042121-1235"
traces, measures = load_or_convert(datadir, foldername)
fig_full = plt.figure("Fiber Traces")
ax_full = fig_full.add_subplot()
fig_sub = plt.figure("Fiber Traces Sub")
ax_sub = fig_sub.add_subplot()
ax_sub = fig_sub.add_subplot()
ax_sub.xaxis.set_visible(False)
ax_sub.spines['bottom'].set_visible(False)
ax_sub.yaxis.set_visible(False)
ax_sub.spines['left'].set_visible(False)
for i in range(10):
    trace = traces[i]
    use = (trace[(0,80),'t'] < 19_500)
    t_min = 19_000
    ax_full.plot(trace[(0,30),'t'][use]-t_min ,traces[i][(0,30),'vOld'][use], alpha=0.5, color=colors[0], label='30')
    ax_full.plot(trace[(0,80),'t'][use]-t_min ,traces[i][(0,80),'vOld'][use], alpha=0.5, color=colors[1], label='80')
    use = use & (trace[(0,80),'t'] < 19_030)
    ax_sub.plot(trace[(0,30),'t'][use]-t_min ,traces[i][(0,30),'vOld'][use], alpha=0.5, color=colors[0])
    ax_sub.plot(trace[(0,80),'t'][use]-t_min ,traces[i][(0,80),'vOld'][use], alpha=0.5, color=colors[1])
handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles[0:2], labels[0:2], frameon=False)


#from SaveSAP import savePlots,setAxisSizePlots
#setAxisSizePlots([(6,3), (2.5,2.5), (1.5,1.5), (2.5,2.5), (1.5,1.5)])
#savePlots('R:/Hund/DanielGratz/atrial_model/plots/latest/plots/', ftype='svg')
