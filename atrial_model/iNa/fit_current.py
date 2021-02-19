#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:06:25 2020

@author: grat05
"""


from functools import partial
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import linalg
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt

import sys
sys.path.append('../../../')

def calcExpTauInact(times, current, sub_sim_pos, durs, calc_dur = (0,1), keep=None, **kwargs):
    if keep is None:
        keep = [1,2]

    flat_durs = durs[sub_sim_pos,:]
    startt = np.sum(flat_durs[:calc_dur[0]])
    stopt  = startt + np.sum(flat_durs[calc_dur[0]:calc_dur[1]]) + flat_durs[calc_dur[1]]
    stimMask = (startt < times) & (times < stopt)
    stimMask = (startt < times) & (times < stopt)

    times_sub = times[stimMask]
    currents_sub = current[stimMask]
    times_normed = times_sub - times_sub[0]
    min_abs_loc = np.argmin(np.abs(currents_sub))
    min_curr = currents_sub[min_abs_loc]
    currents_normed = currents_sub - min_curr
    max_abs_loc = np.argmax(np.abs(currents_normed))
    currents_normed = currents_normed/currents_normed[max_abs_loc]
    currents_normed *= -np.sign(currents_normed[max_abs_loc])
    
    
    #currents_normed = minmax_scale(currents_normed, feature_range=(-1,0))
    # plt.figure()
    # plt.plot(times_normed, currents_sub)
    # plt.plot(times_normed, currents_normed)
    # return
    

    try:
        tau_m = fit_act(times_normed, currents_normed)
        tau_fs = fit_inact(times_normed, currents_normed, tau_m)
        fit = fit_ina(times_normed, currents_normed, tau_m, tau_fs)
    except RuntimeError as e:
        raise e

    return fit[keep]

    
#     peak_loc = np.argmax(np.abs(current[stimMask]))
#     baseline_loc = np.argmin(np.abs(current[stimMask]))
#     peak_val = current[stimMask][peak_loc]
#     baseline_val = current[stimMask][baseline_loc]
#     cut_val = peak_val - (peak_val - baseline_val)*.1
#     try:
#         cut_loc = np.where(np.abs(current[stimMask][peak_loc:]) < np.abs(cut_val))[0][0]
#         cut_loc = np.where(stimMask)[0][peak_loc+cut_loc]
#     except:
#         raise ValueError("Current Peak too small to calculate tau")
    
# #    firstdv = np.diff(current)/np.diff(times)
# #    print(cut_loc-np.argmax(firstdv))
    
#     min_loc = cut_loc#np.argmax(np.abs(current[stimMask]))
# #    min_loc = np.where(stimMask)[0][min_loc]
#     stimMask = stimMask & (times[min_loc] < times)
#     ncurrent = minmax_scale(current[stimMask], (0,1))
#     ntimes = times[stimMask]-times[stimMask][0]
# #    coefs, _ = optimize.curve_fit(func, times[min_loc:]-times[min_loc], current[min_loc:],\
# #                                  p0=(current[min_loc]*3/4,1,current[min_loc]*1/4,1,0))
#     diff_fn = partial(diff, pred=ncurrent, func=func, \
#                       times=ntimes, ssq=True)
#     bounds=([0, 0, 0],[7, 40, 50])
# #    res2 = optimize.least_squares(diff_fn, x0,bounds=bounds)#bounds=([0, 0, 0, 2, -np.inf],[np.inf, 5, np.inf, 30, np.inf])
 
#     bounds = np.array(bounds)
#     minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False}}
#     res = optimize.dual_annealing(diff_fn, bounds=bounds.T,\
#                                           local_search_options=minimizer_kwargs)
# #    print(res.x - res2.x)
# #    minimizer_kwargs = {"method": "BFGS", "options": {"maxiter":100}}

# #    diff_fn = partial(diff, pred=current[stimMask], func=func, \
# #                      times=times[stimMask]-times[min_loc], ssq=True)
# #    minimizer_kwargs = {"method": lstsq_wrap}#"BFGS"}
# #    res = optimize.basinhopping(diff_fn, x0, minimizer_kwargs=minimizer_kwargs, niter=10)
# #    res = optimize.minimize(diff_fn, x0, **minimizer_kwargs)
#     tau_f,tau_s = res.x[:2]#res.x[1], res.x[3]
#     tau_f,tau_s = min(tau_f,tau_s), max(tau_f,tau_s)
 
#     if plot3:
#         try:
#             print(res.success, res.optimality, res.cost, tau_f,tau_s)
#         except AttributeError:
#             print(res)
#         plt.figure()
#         plt.axvline(0,c='black')
#         plt.title(str(res.x))
#         plt.plot(ntimes, ncurrent)
#         plt.plot(ntimes, func(ntimes,*res.x))
# #    print(1/np.mean(np.abs(res.jac[:,0])), 1/np.mean(np.abs(res.jac[:,1])), np.abs(res.x[-1]))
# #    check = int(1/np.mean(np.abs(res.jac[:,0]))>100) + int(1/np.mean(np.abs(res.jac[:,1]))>1000) + int(np.abs(res.x[-1]) > 20)
# #    if check >= 2:
# #        tau_s = 0

#     res.x[:2] = tau_f,tau_s
#     #res.x[1], res.x[3] = tau_f,tau_s
#     return res.x[keep]

def calcExpTauAct(times, current, sub_sim_pos, durs, calc_dur = (0,1), keep=None, **kwargs):
    if keep is None:
        keep = [0]

    flat_durs = durs[sub_sim_pos,:]
    startt = np.sum(flat_durs[:calc_dur[0]])
    stopt  = startt + np.sum(flat_durs[calc_dur[0]:calc_dur[1]]) + flat_durs[calc_dur[1]]
    stimMask = (startt < times) & (times < stopt)
    stimMask = (startt < times) & (times < stopt)

    times_sub = times[stimMask]
    currents_sub = current[stimMask]
    times_normed = times_sub - times_sub[0]
    min_abs_loc = np.argmin(np.abs(currents_sub))
    min_curr = currents_sub[min_abs_loc]
    currents_normed = currents_sub - min_curr
    max_abs_loc = np.argmax(np.abs(currents_normed))
    currents_normed = currents_normed/currents_normed[max_abs_loc]
    currents_normed *= -np.sign(currents_normed[max_abs_loc])
    
    
    #currents_normed = minmax_scale(currents_normed, feature_range=(-1,0))
    # plt.figure()
    # plt.plot(times_normed, currents_sub)
    # plt.plot(times_normed, currents_normed)
    # return
    

    try:
        tau_m = fit_act(times_normed, currents_normed)
        tau_fs = fit_inact(times_normed, currents_normed, tau_m)
        fit = fit_ina(times_normed, currents_normed, tau_m, tau_fs)
    except RuntimeError as e:
        raise e

    return fit[keep]

#     if keep is None:
#         keep = np.ones_like(x0, dtype=bool)

#     flat_durs = durs[sub_sim_pos,:]
#     startt = np.sum(flat_durs[:calc_dur[0]])
#     stopt  = startt + np.sum(flat_durs[calc_dur[0]:calc_dur[1]]) + flat_durs[calc_dur[1]]
#     stimMask = (startt < times) & (times < stopt)
#     min_loc = np.argmax(np.abs(current[stimMask]))
#     min_loc = np.where(stimMask)[0][min_loc]
#     stimMask[min_loc+1:] = False
#     adj_times = times[stimMask]-times[stimMask][0]
# #    coefs, _ = optimize.curve_fit(func, times[min_loc:]-times[min_loc], current[min_loc:],\
# #                                  p0=(current[min_loc]*3/4,1,current[min_loc]*1/4,1,0))
# #    x0[-1] = current[min_loc]
#     diff_fn = partial(diff, pred=np.cbrt(current[stimMask]), func=func, \
#                       times=adj_times)
#     res = optimize.least_squares(diff_fn, x0)
#     if plot3:
#         plt.figure()
#         plt.title(str(res.x))
#         plt.axvline(times[min_loc]-times[stimMask][0],c='black')
#         plt.plot(times-times[stimMask][0], current)
#         plt.plot(adj_times, func(adj_times,*res.x)**3)
#     return res.x[keep]#(current[min_loc]-current[stimMask][0])/(times[min_loc]-times[stimMask][0])#

## curve functions

def simpleCurve(t, tau_m, tau_f):
    res = np.power(1-np.exp(-t/tau_m),3) * (-np.exp(-t/tau_f))
    res = minmax_scale(res, feature_range=(-1,0))
    return res


def iNaCurve(t, tau_m, tau_f, tau_s, fs_prop):
    A1 = fs_prop
    A2 = 1 - fs_prop
    res = np.power(1-np.exp(-t/tau_m),3) * (-A1*np.exp(-t/tau_f) -A2*np.exp(-t/tau_s))
    res = minmax_scale(res, feature_range=(-1,0))
    return res

## overall fit functions

def fit_act(t, vals, bounds = np.array([[0,3],[0,20]])):
    res = vals
    peak_loc = np.argmax(np.abs(res))
    if res[peak_loc] > 0:
        res *= -1
    #plt.plot(t[:peak_loc], np.log(np.cbrt(res[:peak_loc])+3))
    try:
        fit_m, _ = optimize.curve_fit(simpleCurve, t[:peak_loc], res[:peak_loc], bounds=bounds.T)
    except Exception as e:
        print(e)
        return 0.1
#    print(fit_m)
    return fit_m[0]

def fit_inact(t, vals, tau_m, bounds = np.array([[0,5],[0,10],[0,20],[0,1]])):
    res = vals
    peak_loc = np.argmax(np.abs(res))
    if res[peak_loc] > 0:
        res *= -1
    res = res/(1-np.exp(-t/tau_m))**3
    try:
        fs_res =calc_opt(t[peak_loc:], res[peak_loc:])
    except RuntimeError as e:
        print(e)
        fs_res =calc_opt(t[peak_loc:], vals[peak_loc:])
    try:
        fs_res2 = calc_opt_int(t[peak_loc:], res[peak_loc:])
    except RuntimeError as e:
        print(e)
        fs_res2 = calc_opt_int(t[peak_loc:], vals[peak_loc:])
    
    u_sol = fs_res.x
    rates1 = 0.5*(-u_sol[0]+np.array([1,-1])*(u_sol[0]**2 -4*u_sol[1])**0.5)
    rates1 = np.sort(rates1)
#    coefs_res1 = calc_opt_coefs(t[peak_loc:], res[peak_loc:], rates1)
#    coefs1 = coefs_res1.x
#    sol1 = coefs1[0]*np.exp(t[peak_loc:]*rates1[0])+coefs1[1]*np.exp(t[peak_loc:]*rates1[1])
#    print(-1/rates1, coefs_res1.cost)
#    plt.plot(t[peak_loc:], sol1, label='fit1')
    
    u_sol = fs_res2.x
    rates2 = 0.5*(-u_sol[0]+np.array([1,-1])*(u_sol[0]**2 -4*u_sol[1])**0.5)
    rates2 = np.sort(rates2)
#    coefs_res2 = calc_opt_coefs(t[peak_loc:], res[peak_loc:], rates2)
#    coefs2 = coefs_res2.x
#    sol2 = coefs2[0]*np.exp(t[peak_loc:]*rates2[0])+coefs2[1]*np.exp(t[peak_loc:]*rates2[1])
#    print(-1/rates2, coefs_res2.cost)
#    plt.plot(t[peak_loc:], sol2, label='fit2')
#    plt.legend()
    taus = best_taus(-1/rates2, -1/rates1, bounds[1:3].T)
 #   print(taus)
    return taus

def fit_ina(t, vals, tau_m, tau_fs, bounds = np.array([[0,5],[0,10],[0,20],[0,1]])):
    res = vals
    peak_loc = np.argmax(np.abs(res))
    coefs_best = calc_opt_coefs(t[peak_loc:], res[peak_loc:], -1/tau_fs)
    p0 = np.array([tau_m, tau_fs[0], tau_fs[1], coefs_best.x[0]/np.sum(coefs_best.x)])
    
    fit_all, _ = optimize.curve_fit(iNaCurve, t, res, p0=p0, bounds=bounds.T)
    
    if fit_all[1] > fit_all[2]:
        fit_all[1:3] = fit_all[2:0:-1]
        fit_all[3] = 1-fit_all[3]
    #print(fit_all)
    return fit_all

## supporting fit funcitons

def calc_opt_int(t, vals):
    if not np.isfinite(vals).all():
        raise RuntimeError("calc_opt: current vals not finite")
        
    def calc_fs_res(t, vals):
        vals_diff1 = np.gradient(vals, t, edge_order=2)
        int_vals = integrate.cumtrapz(vals, t, initial=vals[0])
        A = np.array([vals, int_vals, np.ones_like(int_vals)]).T
        b = -vals_diff1
        bounds = np.array([[0, np.inf],[0, np.inf], [-np.inf, np.inf]]).T
        fs_res = optimize.lsq_linear(A, b, bounds=bounds)
        return fs_res
    fs_res = calc_fs_res(t, vals)
    peak_err = np.argmax(fs_res.fun)
    
    fs_res = calc_fs_res(t[peak_err:], vals[peak_err:])
    
    return fs_res

def calc_opt(t, vals):
    if not np.isfinite(vals).all():
        raise RuntimeError("calc_opt: current vals not finite")
    
    vals_diff1 = np.gradient(vals, t, edge_order=2)
    vals_diff2 = np.gradient(vals_diff1, t, edge_order=2)

    A = np.array([vals_diff1, vals, -np.ones_like(vals)]).T
    b = -vals_diff2
 
    try:
        fs_res = optimize.lsq_linear(A, b)
    except linalg.LinAlgError as e:
        print(e, np.min(b), np.max(b), np.min(A), np.max(A))
        fs_res = optimize.least_squares(lambda x: A.dot(x), [1,1,1])
    return fs_res

def calc_opt_coefs(t, vals, rates):
    A = np.array([np.exp(rates[0]*t), np.exp(rates[1]*t)]).T
    b = vals
    bounds=np.array([[-np.inf,0],[-np.inf,0]])
    coefs_res = optimize.lsq_linear(A,b,bounds=bounds.T)
    return coefs_res

def best_taus(taus1, taus2, bounds): 
    in_bounds = (bounds[0] <= taus1) & (taus1 <= bounds[1])
    taus1 = taus1[in_bounds]
    if len(taus1) == 2:
        return taus1
    in_bounds = (bounds[0] <= taus2) & (taus2 <= bounds[1])
    taus2 = taus2[in_bounds]
    if len(taus2) == 2:
        return taus2
    best = np.ones(2)
    for i in range(len(taus1)):
        best[i] = taus1[i]
    return np.sort(best)
