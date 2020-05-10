#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:43:47 2020

@author: dgratz
"""

import pymc
import datetime
import numpy as np

from scripts import out_dir
import iNa_fit_functions
from iNa_fit_functions import calc_results, SimResults
from multiprocessing import Pool
from functools import partial
import os

from iNa_sims import sim_fs, datas, keys_all
from stat_model import make_model
from iNa_model_setup import model_name as biophys_model_name
from iNa_model_setup import model_params_initial, mp_locs, sub_mps, model

#from './optimize_Koval_0423_0326.pkl'
model_params_initial[mp_locs] = np.array(
    [-0.80680888,  0.63417512, -0.69291108, -2.04633128, -0.01836348,
        0.35378153,  0.64030428, -0.8010144 ,  0.72033717, -1.68578422,
        5.87859494, -1.00653083, -1.67532066,  0.84144004,  0.88200433,
       -2.70056045, -2.26745786,  2.2395883 , -0.48703343])


if __name__ == '__main__':
    
    iNa_fit_functions.plot1 = False #sim
    iNa_fit_functions.plot2 = False #diff
    iNa_fit_functions.plot3 = False #tau

    model_name = './mcmc_'
    model_name +=  biophys_model_name
    model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    model_name += '.pickle'
    model_name = model_name.format(cdate=datetime.datetime.now())
    db_path = out_dir+'/'+model_name

    print("Running Pool with", os.cpu_count(), "processes")
    with Pool() as proc_pool:
        
        calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, data=datas,error_fill=0,\
                        pool=proc_pool)
        run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs)

        
        made_model = make_model(run_biophysical, keys_all, datas, model_params_initial, mp_locs, model)
        S = pymc.MCMC(made_model, db='pickle', dbname=db_path)
        S.sample(iter=100, burn=0, thin=1, tune_throughout=True, save_interval=100)#, burn_till_tuned=True)
        S.db.close()
        print("Pickle File Written to:")
        print(model_name)
        #pymc.Matplot.plot(S)
