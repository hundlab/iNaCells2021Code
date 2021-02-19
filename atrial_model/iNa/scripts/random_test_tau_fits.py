#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:39:02 2020

@author: grat05
"""

from functools import partial
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import linalg
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
import warnings

import sys
sys.path.append('../../../')

#from iNa_models import Koval_ina, OHaraRudy_INa
import atrial_model
from atrial_model import run_sims
from atrial_model.run_sims_functions import timeAndVoltage, lstsq_wrap, diff
from atrial_model.iNa.models import OHaraRudy_wMark_INa, OHaraRudy_INa
from atrial_model.iNa.fit_current import iNaCurve, fit_act, fit_inact, fit_ina


atrial_model.run_sims_functions.plot1 = False #sim
atrial_model.run_sims_functions.plot2 = False #diff
atrial_model.run_sims_functions.plot3 = False #tau


# true_params = np.random.uniform(low=[0.1,0.5,10,0.6], high=[1,1,20,1], size=4)
# res = iNaCurve(t, *true_params)
# peak_loc = np.argmin(res)
# truth = bi_exp(t, *true_params[1:])
# print(true_params, 0.5*np.sum(np.square(res[peak_loc:]-truth[peak_loc:])))
# plt.plot(t, res, label='original')
# plt.plot(t, truth, label='truth')
# fs_res =calc_opt(t[peak_loc:], res[peak_loc:])
# fs_res2 = calc_opt_int(t[peak_loc:], res[peak_loc:])

# u_sol = fs_res.x
# rate_fs = 0.5*(-u_sol[0]+np.array([1,-1])*(u_sol[0]**2 -4*u_sol[1])**0.5)
# coefs_res = calc_opt_coefs(t[peak_loc:], res[peak_loc:], rate_fs)
# sol1 = coefs_res.x[0]*np.exp(t[peak_loc:]*rate_fs[0])+coefs_res.x[1]*np.exp(t[peak_loc:]*rate_fs[1])
# print(-1/rate_fs, coefs_res.cost)
# plt.plot(t[peak_loc:], sol1, label='fit1')

# u_sol = fs_res2.x
# rate_fs = 0.5*(-u_sol[0]+np.array([1,-1])*(u_sol[0]**2 -4*u_sol[1])**0.5)
# coefs_res = calc_opt_coefs(t[peak_loc:], res[peak_loc:], rate_fs)
# sol2 = coefs_res.x[0]*np.exp(t[peak_loc:]*rate_fs[0])+coefs_res.x[1]*np.exp(t[peak_loc:]*rate_fs[1])
# print(-1/rate_fs, coefs_res.cost)
# plt.plot(t[peak_loc:], sol2, label='fit2')
# plt.legend()

# def fit_inact(t, vals):
#     def calc_opt(t, vals):
#         vals_diff1 = np.gradient(vals, t, edge_order=2)
#         vals_diff2 = np.gradient(vals_diff1, t, edge_order=2)
    
#         A = np.array([vals_diff1, vals, -np.ones_like(vals)]).T
#         b = -vals_diff2
#         fs_res = optimize.lsq_linear(A, b)
#         return fs_res
#     def bi_exp(t, tau_f, tau_s, fs_prop, c):
#         A1 = fs_prop
#         A2 = 1 - fs_prop
#         res = -A1*np.exp(-t/tau_f) -A2*np.exp(-t/tau_s) +c
#         res = minmax_scale(res, feature_range=(-1,0))
#         return res
#     try:
#         peak_loc = np.argmax(np.abs(vals))
#         loc_low = np.arange(peak_loc, len(t), 1, dtype=int)
#         t = t - t[peak_loc]
#         fs_res = calc_opt(t[loc_low], vals[loc_low])
#         i = 0
#         while i < 10 and np.abs(np.median(fs_res.fun)) > 1e-7:
#             low, high = np.percentile(fs_res.fun, [5,95])
#             loc_low = loc_low[np.where((low < fs_res.fun) & (fs_res.fun < high))[0]]
#             fs_res = calc_opt(t[loc_low], vals[loc_low])
#             i += 1
#     #    print(i)
#         fs_res = calc_opt(t[loc_low[0]:loc_low[-1]], vals[loc_low[0]:loc_low[-1]])
#         u_sol = fs_res.x
    
#     #    plt.plot(t, vals)
#     #    plt.scatter(t[loc_low[0]:loc_low[-1]], vals[loc_low[0]:loc_low[-1]])
#     #    plt.plot(t, vals_diff1)
#     #    plt.plot(t, vals_diff2)
#     #    return res
#     #    plt.plot(t, res.fun)
        
#         rate_fs = 0.5*(-u_sol[0]+np.array([1,-1])*(u_sol[0]**2 -4*u_sol[1])**0.5)
#         rate_f, rate_s = min(rate_fs), max(rate_fs)
#         # A = -(np.exp(rate_f*t[loc_low])-np.exp(rate_s*t[loc_low]))
#         # b = vals[loc_low] + np.exp(rate_s*t[loc_low])
#         # prop_res = optimize.lsq_linear(A[..., None], b, bounds=[0,1])
#         # prop = prop_res.x[0]

#         tau_f, tau_s = -1/rate_f, -1/rate_s
#         tau_f = tau_f if tau_f > 0 else 1
#         tau_s = tau_s if tau_s > 0 else 1
        
#         bounds = np.array([[0,1], [-np.inf,np.inf]])
#         fit, _ = optimize.curve_fit(lambda t, fs_prop, c: bi_exp(t, tau_f, tau_s, fs_prop, c), t[peak_loc:], vals[peak_loc:], bounds=bounds.T)
#         fs_prop = fit[0]
#         plt.figure()
#         plt.plot(t[peak_loc:], vals[peak_loc:])
#         plt.plot(t[peak_loc:], bi_exp(t[peak_loc:], tau_f, tau_s, fs_prop, fit[1]))
        
#         return tau_f, tau_s, fs_prop
#     except IndexError:
#         print("fit biExp failed")
#         return 1,1

# def fit_act(t, vals):
#     peak_loc = np.argmax(np.abs(vals))
#     if peak_loc == 0:
#         print("fit_act: No activation found")
#         return 1
#     def mono_exp(t, tau, A, c):
#         return A*np.exp(-t/tau)+c
#     bounds = np.array([[0,np.inf],[0,np.inf],[-np.inf,np.inf]])
#     fit, _ = optimize.curve_fit(mono_exp, t[:peak_loc], np.cbrt(vals[:peak_loc]), bounds=bounds.T)
#     # plt.figure()
#     # plt.plot(t[:peak_loc], np.cbrt(vals[:peak_loc]))
#     # plt.plot(t[:peak_loc], mono_exp(t[:peak_loc], *fit))
# #    plt.figure()
# #    plt.plot(t, vals)
# #    plt.plot(t[peak_loc:], vals[peak_loc:]/(1-np.exp(-t[peak_loc:]/fit[0]))**3)
#     return fit[0]

# def iNaCurve_single(t, tau_m, tau_f):
#     res = np.power(1-np.exp(-t/tau_m),3) * (-np.exp(-t/tau_f))
#     res = minmax_scale(res, feature_range=(-1,0))
#     return res

# def jac(t, tau_m, tau_f, tau_s, fs_prop):
#     A1 = fs_prop
#     A2 = 1 - fs_prop
    
#     res = np.zeros(shape=(len(t), 4))
#     res[:,0] = (-A1*np.exp(-t/tau_f) -A2*np.exp(-t/tau_s)) * 3*np.square(1-np.exp(-t/tau_m))*(-np.exp(-t/tau_m)*t/(tau_m**2))
#     res[:,1] = np.power(1-np.exp(-t/tau_m),3) * (-A1*np.exp(-t/tau_f)*t/(tau_f**2))
#     res[:,2] = np.power(1-np.exp(-t/tau_m),3) * (-A2*np.exp(-t/tau_s)*t/(tau_s**2))
#     res[:,3] = np.power(1-np.exp(-t/tau_m),3) * (-np.exp(-t/tau_f) +np.exp(-t/tau_s))
    
#     return res

def test_fit(model, model_params, v_step_to):
    print("__________")
    voltages = np.array([[-140,v_step_to]])
    durs = np.array([[20,400]])
    
    model_kwargs = dict(TEMP=310, naO=5, naI=5)
    model_instance = model(*model_params, **model_kwargs)
    tau, *rest = model_instance.calc_constants(v_step_to)
    actual_vals = np.array([tau.tm, tau.thf, tau.ths, model_instance.Ahf_mult])
    if actual_vals[1] > actual_vals[2]:
            actual_vals[1:3] = actual_vals[2:0:-1]
            actual_vals[3] = 1-actual_vals[3]

    times_normed = np.arange(0, 20, 0.1)
    currents_normed = iNaCurve(times_normed, *actual_vals)
    
    
    sim = run_sims.SimRunner(model, 
                              voltages, durs,
                              model_kwargs,
                              timeAndVoltage,
                              post_process=None, 
                              solver=partial(integrate.solve_ivp, method='BDF'))#, t_eval=times_normed))
    res = sim.run_sim(model_params)
    times, currents = res.get_output()
    
    times_sub = times[times > 20]
    currents_sub = currents[times > 20]
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
    

    
    bounds = np.array([[0,np.inf],[0,np.inf],[0,np.inf],[0,1]])
    tau_f_est, tau_s_est = 0,0
    try:
        tau_m = fit_act(times_normed, currents_normed)
        tau_fs = fit_inact(times_normed, currents_normed, tau_m)
        fit = fit_ina(times_normed, currents_normed, tau_m, tau_fs)
        
        # peak_loc = np.argmax(np.abs(currents_normed))
        # tau_m = fit_act(times_normed, currents_normed)
        # #vals[peak_loc:]/(1-np.exp(-t[peak_loc:]/fit[0]))**3
        # tau_f, tau_s, fs_prop = fit_inact(times_normed, currents_normed)
        # bounds = np.array([[0,np.inf],[0,np.inf],[0,1]])
        # p0 = np.array([tau_f, tau_s, fs_prop])
        # try:
        #     fit, _ = optimize.curve_fit(lambda t, tau_f, tau_s, fs_prop: iNaCurve( t, tau_m, tau_f, tau_s, fs_prop)
        #                                 , times_normed, currents_normed, bounds=bounds.T, p0=p0)
        #     fit = [tau_m, *fit]
        # except Exception as e:
        #     print(p0)
        #     raise(e)
        
        # peak_loc = np.argmin(currents_normed)
        # tau_f_est, tau_s_est = calc_biExp(times_normed[peak_loc+100:], currents_normed[peak_loc+100:])
        # if tau_f_est > tau_s_est:
        #     tau_f_est, tau_s_est = tau_s_est, tau_f_est
        # p0 = np.array([1, tau_f_est, tau_s_est, 0.5])
        # bounds = np.array([[0,np.inf],[0,np.inf],[0,np.inf],[0,1]])
        # try:
        #     fit, _ = optimize.curve_fit(iNaCurve, times_normed, currents_normed, bounds=bounds.T, p0=p0)
        # except(ValueError):
        #     print("calc_biExp Failed")
        #     fit, _ = optimize.curve_fit(iNaCurve, times_normed, currents_normed, bounds=bounds.T)
        
        # bounds = np.array([[0,np.inf],[0,np.inf]])
        # fit_single, _ = optimize.curve_fit(iNaCurve_single, times_normed, currents_normed, bounds=bounds.T)
        # bounds = np.array([[0,np.inf],[0,np.inf],[0,np.inf],[0,1]])
        # p0 = np.array([1, fit_single[1], fit_single[1], 0.5])
        # fit_sym, _ = optimize.curve_fit(iNaCurve, times_normed, currents_normed, bounds=bounds.T, p0=p0)
        # p0 = np.array([1, fit_single[1], 1, 0.9])
        # fit_right, _ = optimize.curve_fit(iNaCurve, times_normed, currents_normed, bounds=bounds.T, p0=p0)
        # p0 = np.array([1, 1, fit_single[1], 0.1])
        # fit_left, _ = optimize.curve_fit(iNaCurve, times_normed, currents_normed, bounds=bounds.T, p0=p0)
        # fit_default, _ = optimize.curve_fit(iNaCurve, times_normed, currents_normed, bounds=bounds.T)
        # fit = None
        # min_err = np.inf
        # for test_fit in [fit_sym, fit_right, fit_left, fit_default]:
        #     err = 0.5*np.sum(np.square(currents_normed - iNaCurve(times_normed, *test_fit)))
        #     if err < min_err:
        #         err = min_err
        #         fit = test_fit
        
        #, p0=np.array([1,1,10,0.9]
        # bounds =   np.array([[0,np.inf],[0,np.inf]])      
        # fit, _ = optimize.curve_fit(iNaCurve_single, times_normed, currents_normed, bounds=bounds.T)
        # fit_vals = iNaCurve_single(times_normed, *fit)
        # peak_loc = np.argmin(currents_normed)
        # fs_prop_est = np.sum(np.sign(currents_normed-fit_vals)[peak_loc:]==-1) 
        # fs_prop_est = actual_vals[3]#/= len(currents_normed[peak_loc:])
        # # tau_m_est = fit[0]
        # # #p0 = np.array([fit[1], fit[1]])
        # # fit, _ = optimize.curve_fit(lambda t, tau_f, tau_s: iNaCurve(t, tau_m_est, tau_f, tau_s, fs_prop_est),
        # #                             times_normed, currents_normed, bounds=bounds.T)
        # bounds = np.array([[0,np.inf]])
        # fit, _ = optimize.curve_fit(lambda t, arg: iNaCurve(t, actual_vals[0], actual_vals[1], actual_vals[2], arg),
        #                             times_normed, currents_normed, bounds=bounds.T)
        # fit = np.array([actual_vals[0], actual_vals[1], actual_vals[2], fit[0]])

        # p0 = np.array([1,1,1,1])
        # func = partial(iNaCurve, )
        # fit, _ = optimize.curve_fit(iNaCurve, times_normed, currents_normed, bounds=bounds.T)
        
        # bounds = np.array([[0,1000],[0,1000],[0,1000],[0,1]])
        # diff_fn = partial(diff, pred=currents_normed, func=iNaCurve, \
        #               times=times_normed, ssq=True)
        # minimizer_kwargs = {"method": lstsq_wrap, "options":{"ssq": False}}
        # res = optimize.dual_annealing(diff_fn, bounds=bounds,\
        #                                   local_search_options=minimizer_kwargs, no_local_search=True)
 #       fit = res.x
        if fit[1] > fit[2]:
            fit[1:3] = fit[2:0:-1]
            fit[3] = 1-fit[3]
            
    except RuntimeError as e:
        raise e
        # plt.figure()
        # plt.plot(times_normed, currents_normed)
        # plt.plot(times_normed, currents_sub)
        # print(actual_vals)
        return 1
    
    
    plt.figure()
    plt.plot(times_normed, currents_normed)
    plt.plot(times_normed, iNaCurve(times_normed, *fit))
    print(actual_vals)
    #print([tau_f, tau_s])
    print(fit)
    
    # print(actual_vals[3])
    # #print(fs_prop)
    # print(fit[3])
    
    # print(actual_vals[0])
    # # print(tau_m)
    # print(fit[0])
    print('---')
    curve_err = 0.5*np.sum(np.square(currents_normed - iNaCurve(times_normed, *fit)))
    actual_val_err = 0.5*np.sum(np.square(currents_normed - iNaCurve(times_normed, *actual_vals)))
    param_err = 0.5*np.sum(np.square(actual_vals - fit))
    
    if param_err > 10:
        print(np.abs(curve_err - actual_val_err))
        if curve_err > actual_val_err:
            # print("Curve error", curve_err)
            # print("Param Err", param_err)
            print(actual_vals)
            print(fit)
            print(tau_f_est, tau_s_est)
            
            # plt.figure()
            # plt.plot(times_normed, currents_normed)
            # plt.plot(times_normed, iNaCurve(times_normed, *fit))
            # plt.plot(times_normed, iNaCurve(times_normed, *actual_vals))
            return 1
    return 0

n_fail = 0
n = 10
for i in range(n):
    model_params = np.random.normal(loc=0, scale=2, size=24)
    n_fail += test_fit(OHaraRudy_wMark_INa, model_params, 20)
print(n_fail/n)
    