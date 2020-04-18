#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:43:47 2020

@author: dgratz
"""

import pymc
import datetime

from scripts import out_dir
import iNa_fit_functions
from iNa_fit_functions import calc_results
from multiprocessing import Pool
from functools import partial
import os

from iNa_sims import sim_fs, datas, keys_all
from iNa_sims import model_params_initial, model, mp_locs, sub_mps
from stat_model import make_model

if __name__ == '__main__':
    
    iNa_fit_functions.plot1 = False #sim
    iNa_fit_functions.plot2 = False #diff
    iNa_fit_functions.plot3 = False #tau
    
    model_name = './ina_ohara_\
{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}\
.pickle'
    model_name = model_name.format(cdate=datetime.datetime.now())
    db_path = out_dir+'/'+model_name

    print("Running Pool with", os.cpu_count(), "processes")
    with Pool() as proc_pool:
        
        calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                        mp_locs=mp_locs, sim_funcs=sim_fs, data=datas,error_fill=0,\
                        pool=proc_pool)
            
        
        made_model = make_model(calc_fn, keys_all, datas, sub_mps, model)
        S = pymc.MCMC(made_model, db='pickle', dbname=db_path)
        S.sample(iter=1000, burn=300, thin=1)
        S.db.close()
        print("Pickle File Written to:")
        print(model_name)
        #pymc.Matplot.plot(S)
