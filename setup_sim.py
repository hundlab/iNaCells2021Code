#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:45:03 2020

@author: grat05
"""

import numpy as np
import pandas as pd


def setup_sim_vc(data, exp_parameters, hold_dur, data_len=None):
    step_names = [('potential 1 (mV)', 'duration (ms)'),
                  ('holding potential (mV)', 'IPI (ms)'),
                  ('potential 2 (mV)', 'duration 2 (ms)'),
                  ('holding potential (mV)', 'IPI 2 (ms)'),
                  ('potential 3 (mV)', 'duration 3 (ms)')]

    if data_len is None:
        base_params = np.ones_like(data[:,0])
    else:
        base_params = np.ones(data_len)
    voltages = []
    durs = []
    try:
        holding_pot = exp_parameters['holding potential (mV)']
        voltages = [base_params*holding_pot]
        durs = [base_params*hold_dur]
    except KeyError:
        ValueError("No holding potential")
    
    for i, step_name in enumerate(step_names):
        try:
            voltage_name, dur_name = step_name
            voltage = exp_parameters[voltage_name]
            dur = exp_parameters[dur_name]
            
            if voltage == 'x':
                voltage = data[:,0]
            elif pd.isna(voltage):
                raise KeyError
                
            if dur == 'x':
                dur = data[:,0]
            elif pd.isna(dur):
                raise KeyError
                
            voltages.append(base_params*voltage)
            durs.append(base_params*dur)
        except KeyError:
            pass
    
    voltages = np.array(voltages)
    durs = np.clip(np.array(durs), a_min=0, a_max=None)

    return voltages.T, durs.T

def setup_sim(model, data, exp_parameters, hold_dur=1, num_repeats=5, data_len=None, sim_args={}): #ret = [True]*3
    voltages, durs = setup_sim_vc(data, exp_parameters, hold_dur, data_len=data_len,)

    sim_param = {}
    sim_param['naO'] = exp_parameters['[Na]o (mM)']
    sim_param['naI'] = exp_parameters['[Na]I (mM)']
    sim_param['TEMP'] = exp_parameters['temp ( K )']

    try:
        repeat_frq = exp_parameters['Repeat (Hz)']
        if repeat_frq == 'x':
            repeat_dur = 1000/data[:,0]
        else:
            if pd.isna(repeat_frq):
                raise KeyError
            repeat_dur = 1000/repeat_frq
            
        rep_dur = np.sum(durs, axis=1)
        rest_dur = repeat_dur - rep_dur
        if np.min(rest_dur) < 0:
            raise ValueError("Repeat Frequency smaller than protocol duration")
        hold_potential = voltages[:,0]
        voltages = np.concatenate((voltages, hold_potential[...,None]), axis=1)
        voltages = np.tile(voltages, (1,num_repeats))
        durs = np.concatenate((durs, rest_dur[...,None]), axis=1)
        durs = np.tile(durs, (1,num_repeats))
    except KeyError:
        pass

    f_call = partial(run_sim, model=model, voltages=voltages, durs=durs,\
                     sim_param=sim_param, **sim_args)
    return voltages, durs, f_call


def get_exp_y(data, exp_parameters):
    curr_real = data[:,1]
    capacitance = exp_parameters['capacitance (pF)']
    if not pd.isna(capacitance):
        curr_real = curr_real*1000/capacitance
    return curr_real