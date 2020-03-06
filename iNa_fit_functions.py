#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:37:41 2020

@author: grat05
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import partial

plot1 = True  #sim
plot2 = True   #diff
plot3 = True  #tau


def isList(thing):
    return isinstance(thing, (list, tuple, np.ndarray))

def flatten_durs(durs, sub_sim_pos):
    return [val[sub_sim_pos] if isList(val) else val\
                 for val in durs]

def plotEach(times, current, **kwargs):
    plt.plot(times, current)

def peakCurr(times, current, sub_sim_pos, durs, durn=None, **kwargs):
    flat_durs = flatten_durs(durs, sub_sim_pos)
    return current[getPeak(times, current, durn, flat_durs)[0]]

def biLinear(t, A1, B1, A2, B2):
    first = -t/B1 + A1
    second = -t/B2 + A2
    return np.maximum(first, second)

def triExp(t,A1,tau1,A2,tau2,A3,tau3,A0,sign=-1):
#    if np.sign(A1) != np.sign(A2):
#        A1 = -A1
    return A1*np.exp(-t/tau1)+A2*np.exp(sign*t/tau2)+A3*np.exp(sign*t/tau3)+A0


def biExp(t,A1,tau1,A2,tau2,A0,sign=-1):
#    if np.sign(A1) != np.sign(A2):
#        A1 = -A1
    return A1*np.exp(-t/tau1)+A2*np.exp(sign*t/tau2)+A0

def monoExp(t,A1,tau1,A0,sign=-1):
    val = A1*np.exp(sign*t/tau1)+A0
    test = np.sum(np.isinf(val))
    if test > 0:
        print(test)
    return val

def diff(x, pred, func, times):
    return pred-func(times, *x)


def calcExpTauInact(times, current, func, x0, sub_sim_pos, durs, calc_dur = 1, keep=None, bounds=(-np.inf, np.inf), **kwargs):
    if keep is None:
        keep = np.ones_like(x0, dtype=bool)
    flat_durs = flatten_durs(durs, sub_sim_pos)
    startt = np.sum(flat_durs[:calc_dur])
    stopt  = startt + flat_durs[calc_dur]
    stimMask = (startt < times) & (times < stopt)
    min_loc = np.argmax(np.abs(current[stimMask]))
    min_loc = np.where(stimMask)[0][min_loc]
    stimMask = stimMask & (times[min_loc] < times)
#    coefs, _ = optimize.curve_fit(func, times[min_loc:]-times[min_loc], current[min_loc:],\
#                                  p0=(current[min_loc]*3/4,1,current[min_loc]*1/4,1,0))
    diff_fn = partial(diff, pred=current[stimMask], func=func, \
                      times=times[stimMask]-times[min_loc])
    res = optimize.least_squares(diff_fn, x0, bounds=bounds)
    if plot3:
        plt.figure()
        plt.axvline(0,c='black')
        plt.title(str(res.x))
        plt.plot(times-times[min_loc], current)
        plt.plot(times[min_loc:]-times[min_loc], func(times[min_loc:]-times[min_loc],*res.x))
    return res.x[keep]

def calcExpTauAct(times, current, func, x0, sub_sim_pos, durs, calc_dur = (0,1), keep=None, **kwargs):
    if keep is None:
        keep = np.ones_like(x0, dtype=bool)

    flat_durs = flatten_durs(durs, sub_sim_pos)
    startt = np.sum(flat_durs[:calc_dur[0]])
    stopt  = startt + np.sum(flat_durs[calc_dur[0]:calc_dur[1]]) + flat_durs[calc_dur[1]]
    stimMask = (startt < times) & (times < stopt)
    min_loc = np.argmax(np.abs(current[stimMask]))
    min_loc = np.where(stimMask)[0][min_loc]
    stimMask[min_loc+1:] = False
    adj_times = times[stimMask]-times[stimMask][0]
#    coefs, _ = optimize.curve_fit(func, times[min_loc:]-times[min_loc], current[min_loc:],\
#                                  p0=(current[min_loc]*3/4,1,current[min_loc]*1/4,1,0))
#    x0[-1] = current[min_loc]
    diff_fn = partial(diff, pred=current[stimMask], func=func, \
                      times=adj_times)
    res = optimize.least_squares(diff_fn, x0)
    if plot3:
        plt.figure()
        plt.axvline(times[min_loc]-times[stimMask][0],c='black')
        plt.plot(times-times[stimMask][0], current)
        plt.plot(adj_times, func(adj_times,*res.x))
    return res.x[keep]#(current[min_loc]-current[stimMask][0])/(times[min_loc]-times[stimMask][0])#


def getPeak(times, current, dur_loc, flat_durs):
    if isinstance(dur_loc, tuple):
        begint = np.sum(flat_durs[:dur_loc[0]])
        endt  = begint + np.sum(flat_durs[dur_loc[0]:dur_loc[1]]) + flat_durs[dur_loc[1]]
    elif dur_loc is None:
        begint = 0
        endt = times[-1]
    else:
        begint = sum(flat_durs[:dur_loc])
        endt = begint + flat_durs[dur_loc]
    locs = np.where((times>begint)&(times<endt))[0]
    peak_loc = np.argmax(np.abs(current[locs]))
    return locs[peak_loc], begint, endt

#I2/I1
def normalize2prepulse(times, current, sub_sim_pos, durs, pulse1=1, pulse2=3, **kwargs):
    flat_durs = flatten_durs(durs, sub_sim_pos)
    peak = []
    peak.append(current[getPeak(times, current, pulse1, flat_durs)[0]])
    peak.append(current[getPeak(times, current, pulse2, flat_durs)[0]])

#    plotEach(times, current, **kwargs)
    return peak[1]/peak[0]

def normalized2val(times, current, sub_sim_pos, durs, val, durn, **kwargs):
    flat_durs = flatten_durs(durs, sub_sim_pos)

    peak_loc,_,_ = getPeak(times, current, durn, flat_durs)
    peak = current[peak_loc]

#    plotEach(times, current, **kwargs)
#    print(np.round(peak,2), peak_loc)
    return peak/val

def run_sim(model_parameters, model, voltages, durs, sim_param, process, dt=0.005, ret = [True]*3):
    out = []
    try:
        counterDur = max(map(len, filter(isList,durs)))
    except ValueError:
        counterDur = 0
    try:
        counterVm = max(map(len, filter(isList,voltages)))
    except ValueError:
        counterVm = 0
    counter = counterDur+counterVm
    if counter == 0:
        counter = 1

    for x in range(counter):
        model_parameters_temp = list(model_parameters)+[sim_param['TEMP']]
        run_model = model(*model_parameters_temp)
        run = True
        t = 0
        i = 0
        tnext = durs[i]
        vM = voltages[i]
        iNa = []
        times = []
        vMs = []
        while run:
            while t >= tnext:
                i += 1
                if i >= len(durs):
                    run = False
                    break
                if isList(durs[i]):
                    tnext += durs[i][x]
                else:
                    tnext += durs[i]
                if isList(voltages[i]):
                    vM = voltages[i][x]
                else:
                    vM = voltages[i]
            iNa.append(run_model.updateIna(vM, dt, naO=sim_param['naO'], naI=sim_param['naI'], ret=ret))
            times.append(t)
            vMs.append(vM)
            t += dt
        times = np.array(times)
        iNa = np.array(iNa)
        vMs = np.array(vMs)
        out.append(process(times=times,current=iNa,vMs=vMs,sub_sim_pos=x,durs=durs))
        if plot1:
            plt.figure()
            plt.subplot(311)
            plt.plot(times, iNa)
            plt.subplot(312)
            plt.plot(times, vMs)
            plt.subplot(313)
            plt.plot(times, run_model.recArray)
#        if plot1 or plot3:
#            plt.show()
    return np.array(out)

def setup_sim(model, data, exp_parameters, process, dt=0.005, ret = [True]*3, hold_dur=1):
    voltages = []
    durs = []
    sim_param = {}
    sim_param['naO'] = exp_parameters['[Na]o (mM)']
    sim_param['naI'] = exp_parameters['[Na]I (mM)']
    sim_param['TEMP'] = exp_parameters['temp ( K )']
    try:
        #holding
        holding_pot = exp_parameters['holding potential (mV)']
        voltages.append(holding_pot)
        durs.append(hold_dur)

        #pulse 1
        duration = exp_parameters['duration (ms)']
        if duration == 'x':
            durs.append(data[:,0])
        else:
            durs.append(duration)
        potential = exp_parameters['potential 1 (mV)']
        if potential == 'x':
            voltages.append(data[:,0])
        else:
            voltages.append(potential)

        #IPI 1
        IPI = exp_parameters['IPI (ms)']
        if IPI == 'x':
            durs.append(data[:,0])
        else:
            durs.append(IPI)
        voltages.append(holding_pot)

        #pulse 2
        duration = exp_parameters['duration 2 (ms)']
        if duration == 'x':
            durs.append(data[:,0])
        else:
            durs.append(duration)
        potential = exp_parameters['potential 2 (mV)']
        if potential == 'x':
            voltages.append(data[:,0])
        else:
            voltages.append(potential)

        #IPI 2
        IPI = exp_parameters['IPI 2 (ms)']
        if IPI == 'x':
            durs.append(data[:,0])
        else:
            durs.append(IPI)
        voltages.append(holding_pot)

         #pulse 3
        duration = exp_parameters['duration 3 (ms)']
        if duration == 'x':
            durs.append(data[:,0])
        else:
            durs.append(duration)
        potential = exp_parameters['potential 3 (mV)']
        if potential == 'x':
            voltages.append(data[:,0])
        else:
            voltages.append(potential)

    except KeyError:
        pass
    durs = list(filter(lambda x: isList(x) or ~np.isnan(x), durs))
    voltages  = list(filter(lambda x: isList(x) or ~np.isnan(x), voltages))
    f_call = partial(run_sim, model=model, voltages=voltages, durs=durs,\
                     sim_param=sim_param, process=process, dt=dt, ret=ret)
    return voltages, durs, f_call


def get_exp_y(data, exp_parameters):
    curr_real = data[:,1]
    capacitance = exp_parameters['capacitance (pF)']
    if not np.isnan(capacitance):
        curr_real = curr_real*1000/capacitance
    return curr_real

def calc_diff(model_parameters_part, model_parameters_full, sim_func, data, mp_locs=None, l=1, pool=None):
    if isList(sim_func):
        return calc_diff_multiple(model_parameters_part, model_parameters_full, sim_func, data, mp_locs, l, pool)
    else:
        return calc_diff_single(model_parameters_part, model_parameters_full, sim_func, data, mp_locs, l)


def calc_diff_single(model_parameters_part, model_parameters_full, sim_func, data, mp_locs=None, l=1):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    model_parameters_full[mp_locs] = model_parameters_part
    vals_sim = sim_func(model_parameters_full)
    if plot2:
        plt.figure("Overall")
        plt.plot(data[:,0], vals_sim)
        plt.scatter(data[:,0], data[:,1])
    p_loss = 1/(model_parameters_full+1)-0.5
    error = np.concatenate((vals_sim-data[:,1], l*p_loss))
    with np.printoptions(precision=3):
        print(model_parameters_full)
        print(np.sum(error**2),np.sum(p_loss**2))
    return error

def calc_diff_multiple(model_parameters_part, model_parameters_full, sim_func, data, mp_locs=None, l=1, pool=None):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    if plot2:
        plt.figure("Overall")
    model_parameters_full[mp_locs] = model_parameters_part
    error = []
    if not pool is None:
        vals_sims_res = []
        for i in range(len(sim_func)):
            vals_sims_res.append(pool.apply_async(sim_func[i], (model_parameters_full,)))
        vals_sims = [res.get() for res in vals_sims_res]

    for i in range(len(sim_func)):
        sub_dat = data[i]
        if not pool is None:
            vals_sim = vals_sims[i]
        else:
            sim_f_sing = sim_func[i]
            vals_sim = sim_f_sing(model_parameters_full)
        error += list(sub_dat[:,1] - vals_sim)
        if plot2:
            plt.figure("Overall")
            plt.plot(sub_dat[:,0], vals_sim)
            plt.scatter(sub_dat[:,0], sub_dat[:,1])
    error = np.array(error)
    p_loss = 1/(model_parameters_full+1)-0.5
    error = np.concatenate((error, l*p_loss))
    with np.printoptions(precision=3):
        print(model_parameters_full)
        print(np.sum(error**2),np.sum((l*p_loss)**2))
    return error

#from functools import partial
##voltages = np.arange(-100,100, 10)
#voltagesAll = [-120, -20, -140, -20]
#durs = np.arange(0,2,0.1)
#dursAll = [100, 100, durs, 10]
#normedCurrs = run_sim(np.ones(22), Koval_ina, voltagesAll, dursAll, {'TEMP':310, 'naO': 140.0, 'naI':8.355}, normalize2prepulse)
#partial(calcExpTauInact,func=monoExp,x0=np.ones(3))