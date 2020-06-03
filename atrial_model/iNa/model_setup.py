#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:13:10 2020

@author: grat05
"""
from .models import OHaraRudy_INa, Koval_ina, OHaraRudy_Gratz_INa, OHaraRudy_wMark_INa
from ..parse_cmd_args import args

import numpy as np
import inspect


try: run_fits
except NameError: 
    run_fits = {'Activation':   True,
                'Inactivation': True,
                'Recovery':     True,
                'Tau Act':      True,
                'Late':         True,
                }

#model_name = "OHaraGratz"
#model_name = "OHaraRudy_wMark_INa"
#model_name = "OHara"
#model_name = "Koval"

mp_locs = []

if args.model_name == "OHara":
    model = OHaraRudy_INa#models.iNa.OharaRudy_INa#OHaraRudy_INa
    retOptions  = model().retOptions
    dt = 0.05
    
    model_params_initial = np.zeros(model.num_params)
    
    #fits_results_joint = pickle.load(open('./fits_res_joint_ohara_0306.pkl','rb'))
    #res = fits_results_joint['group']
    
    model_params_initial = np.zeros(model.num_params)#np.array([ 0.        ,  0.        ,  1.49431475, -1.84448536, -1.21581823,
    #        0.04750437,  0.09809738,  0.        ,  0.        ,  0.        ,
    #        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #        0.        ,  0.        ,  1.487     , -1.788     , -0.254     ,
    #       -3.423     ,  4.661     ,  0.        ,  0.        ,  0.        ,
    #        0.        ,  0.        ,  0.        ,  0.        ])#np.zeros(model().num_params)
    #model_params_initial[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = [ 0.505, -0.533, -0.286 , 2.558 , 0.92 ,  0.333 ,14.476 , 2.169 , 1.876, -0.487
    # , 3.163 , 3.814 , 2.584 ,-0.772 ,-0.129 , 0.528, -0.868 , 0.319 ,-3.578 , 0.149]
        
    #best fit at one point
    #model_params = np.array([ 0.        ,  0.        ,  0, -1.84448536, -1.21581823,
          #   0.04750437,  0.09809738,  0.78      ,  0.725     , -0.031     ,
          # -2.144     ,  0.84      ,  2.018     ,  0.276     ,  7.073     ,
          #   4.641     , -1.427     ,  1.513     , -1.345     , -0.26      ,
          # -3.422     ,  6.159     ,  0.        ,  0.        ,  0.        ,
          #   0.        ,  0.        ,  0.        ,  0.        ])
    #a terrible fit
    #model_params = np.array([ 0.06721265, -0.08104161, -0.9442514 ,  0.93791262,  0.24704036,
           # -0.54269396, -1.62755681,  0.41342681, -0.07028288, -1.84152805,
           # -0.35309849, -0.23240665, -0.3949827 , -0.77066988,  0.74966051,
           # -0.74232144, -0.6377425 ,  0.34105554])
    #model_params = np.random.normal(loc=0, scale=0.5, size=model.num_params)
    #model_params = np.random.uniform(low=-5, high=5, size=model.num_params)
    #model_params[[7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 30]] = np.array([-0.16394553,  1.42641913,  1.28038251,  1.29642736, -1.16718484,
    #       -4.44375614,  1.30915332, -4.0239836 , 13.42958559,  2.20538925,
    #        1.69031741, -1.93559858])
    #model_params[[7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 30]] = np.array([ 1.81552155e-01, -5.05609163e+00,  2.82459638e+01,  2.20175018e-01,
    #       -3.55661129e-03, -6.74500269e+01,  1.66096238e-02, -3.33020753e+00,
    #        1.62150070e+01,  7.80529744e-01,  5.35441384e-01, -5.99818295e-01])
    #model_params[[5,6]] = [np.log(1/100), 16]
    
    #model_params[2:7] = 1.49431475, -1.84448536, -1.21581823,  0.04750437,  0.09809738
    #model_params[2:7] = 0
    #model_params[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = [ 0.505, -0.533, -0.286 , 2.558 , 0.92 ,  0.333 ,14.476 , 2.169 , 1.876, -0.487
    # , 3.163 , 3.814 , 2.584 ,-0.772 ,-0.129 , 0.528, -0.868 , 0.319 ,-3.578 , 0.149]
    #model_params[7] = 2
    # model_params[7] = np.log(2)
    # model_params[8] = np.log(1/10 * 1/1.2)
    # model_params[9] = np.log(1/1.2)
    # model_params[10] = np.log(1/1.2)
    # model_params[11] = np.log(1/1.2)
    # model_params[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = [ 0.505, -0.533, -0.286 , 2.558 , 0.92 ,  0.333 ,14.476 , 2.169 , 1.876, -0.487
    #  , 3.163 , 3.814 , 2.584 ,-0.772 ,-0.129 , 0.528, -0.868 , 0.319 ,-3.578 , 0.149]
    #model_params[16] = -2
    
    if run_fits['Recovery']:
        mp_locs += [7] + list(range(17,22))# + [31,32] # [7]
        
    if run_fits['Inactivation']:
        mp_locs += list(range(7,12)) + [16,30] #7,17
        
    if run_fits['Activation']:
        mp_locs += list(range(2,7)) + [29]#list(range(2,5))

elif args.model_name == "Koval":
    model = Koval_ina
    retOptions  = model().retOptions
    dt = 0.05
    
    model_params_initial = np.zeros(model.num_params)
    mp_locs = np.arange(1,20)
    
elif args.model_name == "OHaraGratz":
    model = OHaraRudy_Gratz_INa#models.iNa.OharaRudy_INa#OHaraRudy_INa
    retOptions  = model().retOptions
    dt = 0.05
    
    model_params_initial = np.zeros(model.num_params)

    if run_fits['Recovery']:
        mp_locs += [7] + list(range(13,16)) + [25,26] # [7]
        
    if run_fits['Inactivation']:
        mp_locs += list(range(7,13)) + [24,27] #7,17
        
    if run_fits['Activation']:
        mp_locs += list(range(2,7)) + [23]#list(range(2,5))
        
elif args.model_name == "OHaraRudy_wMark_INa":
    model = OHaraRudy_wMark_INa#models.iNa.OharaRudy_INa#OHaraRudy_INa
    retOptions  = model().retOptions
    dt = 0.05
    
    model_params_initial = np.zeros(model.num_params)

    mp_locs = np.arange(model.num_params)


mp_locs = np.array(list(set(mp_locs)))
sub_mps = model_params_initial[mp_locs]
sub_mp_bounds = np.array(model.param_bounds)[mp_locs]

model_param_names = np.array(inspect.getfullargspec(model.__init__).args)[1:][mp_locs]
print(model_param_names)

