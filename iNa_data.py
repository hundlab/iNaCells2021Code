#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:00:12 2020

@author: grat05
"""
#data = load_all_data()
iv_curves = {'1323431_1':[0], '1323431_3':[0,1,2], '1323431_4':[0,1,2], '7971163_1':[0], '27694909_8':[0,1,2,3], '21647304_1':[0,1], '12890054_3':[0,1,2,3], '12890054_5':[0,1,2,3], '20488304_2':[5,6], '8928874_7':[0,1,2,3], '23341576_2':[2,3,4,5]}

reversal_pot = {'1323431_3':[3]}

#TTX drug
current_depletion = { '1323431_4':[3]}

iv_curve_datas = {}
iv_curve_names = {}
for pmid in iv_curves:
    subplot_datas = []
    subplot_names = []
    plt.figure()
    plt.title(pmid)
    plt.grid(True, which='major')
    plt.axhline()
    plt.axvline()
    for subplot in iv_curves[pmid]:
        subplot_data = all_data[pmid]['datasetColl'][subplot]['data']
        subplot_name = all_data[pmid]['datasetColl'][subplot]['name']
        subplot_data = np.array([item['value'] for item in subplot_data])
        subplot_datas.append(subplot_data)
        subplot_names.append(subplot_name)
        plt.scatter(subplot_data[:,0],subplot_data[:,1],label=subplot_name)
    plt.legend()
    iv_curve_datas[pmid] = subplot_datas
    iv_curve_names[pmid] = subplot_names

