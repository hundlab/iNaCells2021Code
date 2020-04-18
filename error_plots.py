#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:00:46 2020

@author: grat05
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt


#full optimization
res = pickle.load(open('./optimize_ohara_0417_0041.pkl','rb'))


points = np.empty((len(res.all_res),2))
for i,sim_res in enumerate(res.all_res):
    points[i,0] = np.linalg.norm(res.x-sim_res[1])
    points[i,1] = 0.5*np.sum(np.square(sim_res[0]))

points = points[np.where(points[:,-1] < 100000)[0]]

#plt.figure()
#plt.scatter(points[:,0],points[:,1])
    
pair_points = np.empty((4000,len(res.x)+1))
for i,sim_res in enumerate(res.all_res):
    pair_points[i,:-1] = sim_res[1]
    pair_points[i,-1] = 0.5*np.sum(np.square(sim_res[0]))
    
sub_pps = pair_points[np.where(pair_points[:,-1] < 100000)[0]]

plt.figure()
plt.scatter(sub_pps[:,10],sub_pps[:,11],c=sub_pps[:,-1], cmap='seismic_r', vmin=10, vmax=40)