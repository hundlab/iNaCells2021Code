#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:35:44 2020

@author: grat05
"""

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

import atrial_model
from convert_sim_data import data2DataFrame, data2HDF, readHDF

def load_or_convert(datadir, foldername):
    out_filename = foldername+'.h5'
    
    try:
        traces, measures = readHDF(datadir+out_filename)
    except OSError:
        traces, measures = data2DataFrame(datadir+foldername)
        data2HDF(datadir+out_filename,
         traces_by_cell=traces,
         measured_by_cell=measures)
    return traces, measures



