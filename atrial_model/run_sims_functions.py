#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:37:41 2020

@author: grat05
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
from functools import partial
from sklearn.preprocessing import minmax_scale

plot1 = True  #sim
plot2 = True  #diff
plot3 = True  #tau


monoExp_params = [-1,1,0]
biExp_params = np.array([-1,1,-1,1000,0])

def isList(thing):
    return isinstance(thing, (list, tuple, np.ndarray))

def plotEach(times, current, **kwargs):
    plt.plot(times, current)
    
def timeAndVoltage(times, current, **kwargs):
    return times, current

def peakCurr(times, current, sub_sim_pos, durs, durn=None, **kwargs):
    flat_durs = durs[sub_sim_pos,:]
    return current[getPeak(times, current, durn, flat_durs)[0]]

def biLinear(t, A1, B1, A2, B2):
    first = -t/B1 + A1
    second = -t/B2 + A2
    return np.maximum(first, second)

def triExp(t,A1,tau1,A2,tau2,A3,tau3,A0,sign=-1):
#    if np.sign(A1) != np.sign(A2):
#        A1 = -A1
    return A1*np.exp(-t/tau1)+A2*np.exp(sign*t/tau2)+A3*np.exp(sign*t/tau3)+A0


def biExp(t,tau1,tau2,A):#t,A1,tau1,A2,tau2,A0,sign=-1):
#    if np.sign(A1) != np.sign(A2):
#        A1 = -A1
#A1*np.exp(-t/tau1)+A2*np.exp(sign*t/tau2)+A0
    A0=1
    A1 = A/(A+1)
    A2 = 1-A1
    return -A1*np.exp(-t/tau1) -A2*np.exp(-t/tau2) + A0

def monoExp(t,A1,tau1,A0,sign=-1):
    val = A1*np.exp(sign*t/tau1)+A0
    return val

def diff(x, pred, func, times, ssq=False):
    error = pred-func(times, *x)
    if ssq:
        return 0.5*np.sum(np.square(error))
    else:
        return error

def lstsq_wrap(fun, x0, bounds=None, **kwargs):
    if bounds is None:
        bounds = (-np.inf,np.inf)
    else:
        #it had best be convertable to a numpy array
        bounds = np.array(bounds).T
    options = None
    if 'ssq' in kwargs:
        options = {'ssq': kwargs['ssq']}
    try:
        res = optimize.least_squares(fun, x0, bounds=bounds, kwargs=options)
        res.resid = res.fun
        res.fun = res.cost
        return res
    except ValueError:
        return optimize.OptimizeResult(x=x0, success=False, status=-1, fun=np.inf)


def getNormalizedCurrentSection(times, current, sub_sim_pos, durs, calc_dur = 1, **kwargs):
    flat_durs = durs[sub_sim_pos,:]
    startt = np.sum(flat_durs[:calc_dur])
    stopt  = startt + flat_durs[calc_dur]
    stimMask = (startt <= times) & (times <= stopt)
    
    times_sub = times[stimMask]
    currents_sub = current[stimMask]
    times_normed = times_sub - times_sub[0]
    min_abs_loc = np.argmin(np.abs(currents_sub))
    min_curr = currents_sub[min_abs_loc]
    currents_normed = currents_sub - min_curr
    max_abs_loc = np.argmax(np.abs(currents_normed))
    currents_normed = currents_normed/currents_normed[max_abs_loc]
    currents_normed *= -np.sign(currents_normed[max_abs_loc])
    
    return currents_normed
    

def calcExpTauInact(times, current, func, x0, sub_sim_pos, durs, calc_dur = 1, keep=None, bounds=(-np.inf, np.inf), **kwargs):
    if keep is None:
        keep = np.ones_like(x0, dtype=bool)
    flat_durs = durs[sub_sim_pos,:]
    startt = np.sum(flat_durs[:calc_dur])
    stopt  = startt + flat_durs[calc_dur]
    stimMask = (startt < times) & (times < stopt)
    peak_loc = np.argmax(np.abs(current[stimMask]))
    baseline_loc = np.argmin(np.abs(current[stimMask]))
    peak_val = current[stimMask][peak_loc]
    baseline_val = current[stimMask][baseline_loc]
    cut_val = peak_val - (peak_val - baseline_val)*.1
    try:
        cut_loc = np.where(np.abs(current[stimMask][peak_loc:]) < np.abs(cut_val))[0][0]
        cut_loc = np.where(stimMask)[0][peak_loc+cut_loc]
    except:
        raise ValueError("Current Peak too small to calculate tau")
    
#    firstdv = np.diff(current)/np.diff(times)
#    print(cut_loc-np.argmax(firstdv))
    
    min_loc = cut_loc#np.argmax(np.abs(current[stimMask]))
#    min_loc = np.where(stimMask)[0][min_loc]
    stimMask = stimMask & (times[min_loc] < times)
    ncurrent = minmax_scale(current[stimMask], (0,1))
    ntimes = times[stimMask]-times[stimMask][0]
#    coefs, _ = optimize.curve_fit(func, times[min_loc:]-times[min_loc], current[min_loc:],\
#                                  p0=(current[min_loc]*3/4,1,current[min_loc]*1/4,1,0))
    diff_fn = partial(diff, pred=ncurrent, func=func, \
                      times=ntimes, ssq=True)
    bounds=([0, 0, 0],[7, 40, 50])
#    res2 = optimize.least_squares(diff_fn, x0,bounds=bounds)#bounds=([0, 0, 0, 2, -np.inf],[np.inf, 5, np.inf, 30, np.inf])
 
    bounds = np.array(bounds)
    minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False}}
    res = optimize.dual_annealing(diff_fn, bounds=bounds.T,\
                                          local_search_options=minimizer_kwargs)
#    print(res.x - res2.x)
#    minimizer_kwargs = {"method": "BFGS", "options": {"maxiter":100}}

#    diff_fn = partial(diff, pred=current[stimMask], func=func, \
#                      times=times[stimMask]-times[min_loc], ssq=True)
#    minimizer_kwargs = {"method": lstsq_wrap}#"BFGS"}
#    res = optimize.basinhopping(diff_fn, x0, minimizer_kwargs=minimizer_kwargs, niter=10)
#    res = optimize.minimize(diff_fn, x0, **minimizer_kwargs)
    tau_f,tau_s = res.x[:2]#res.x[1], res.x[3]
    tau_f,tau_s = min(tau_f,tau_s), max(tau_f,tau_s)
 
    if plot3:
        try:
            print(res.success, res.optimality, res.cost, tau_f,tau_s)
        except AttributeError:
            print(res)
        plt.figure()
        plt.axvline(0,c='black')
        plt.title(str(res.x))
        plt.plot(ntimes, ncurrent)
        plt.plot(ntimes, func(ntimes,*res.x))
#    print(1/np.mean(np.abs(res.jac[:,0])), 1/np.mean(np.abs(res.jac[:,1])), np.abs(res.x[-1]))
#    check = int(1/np.mean(np.abs(res.jac[:,0]))>100) + int(1/np.mean(np.abs(res.jac[:,1]))>1000) + int(np.abs(res.x[-1]) > 20)
#    if check >= 2:
#        tau_s = 0

    res.x[:2] = tau_f,tau_s
    #res.x[1], res.x[3] = tau_f,tau_s
    return res.x[keep]

def calcExpTauAct(times, current, func, x0, sub_sim_pos, durs, calc_dur = (0,1), keep=None, **kwargs):
    if keep is None:
        keep = np.ones_like(x0, dtype=bool)

    flat_durs = durs[sub_sim_pos,:]
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
    diff_fn = partial(diff, pred=np.cbrt(current[stimMask]), func=func, \
                      times=adj_times)
    res = optimize.least_squares(diff_fn, x0)
    if plot3:
        plt.figure()
        plt.title(str(res.x))
        plt.axvline(times[min_loc]-times[stimMask][0],c='black')
        plt.plot(times-times[stimMask][0], current)
        plt.plot(adj_times, func(adj_times,*res.x)**3)
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
    flat_durs = durs[sub_sim_pos,:]
    peak = []
    peak.append(current[getPeak(times, current, pulse1, flat_durs)[0]])
    peak.append(current[getPeak(times, current, pulse2, flat_durs)[0]])

#    plotEach(times, current, **kwargs)
    return peak[1]/peak[0]

def normalized2val(times, current, sub_sim_pos, durs, val, durn, **kwargs):
    flat_durs = durs[sub_sim_pos,:]

    peak_loc,_,_ = getPeak(times, current, durn, flat_durs)
    peak = current[peak_loc]

#    plotEach(times, current, **kwargs)
#    print(np.round(peak,2), peak_loc)
    return peak/val

def integrateDur(times, current, dur_loc, durs, sub_sim_pos, begin_offset=0, **kwargs):
    flat_durs = durs[sub_sim_pos,:]
    if isinstance(dur_loc, tuple):
        begint = np.sum(flat_durs[:dur_loc[0]]) + begin_offset
        endt  = begint + np.sum(flat_durs[dur_loc[0]:dur_loc[1]]) + flat_durs[dur_loc[1]]
    elif dur_loc is None:
        begint = begin_offset
        endt = times[-1]
    else:
        begint = sum(flat_durs[:dur_loc]) + begin_offset
        endt = begint + flat_durs[dur_loc]
    locs = np.where((times>begint)&(times<endt))[0]
    integ = integrate.simps(y=current[locs], x=times[locs])
    return integ

def medianValFromEnd(times, current, dur_loc, durs, sub_sim_pos, window=5, **kwargs):
    flat_durs = durs[sub_sim_pos,:]
    if dur_loc is None:
        endt = times[-1]
        begint = endt - window
    else:
        endt = sum(flat_durs[:dur_loc]) + flat_durs[dur_loc]
        begint =  endt - window
    locs = np.where((times>=begint)&(times<=endt))[0]
    return np.median(current[locs])

def multipleProcess(processes, **kwargs):
    res = [process(**kwargs) for process in processes]
    return res
