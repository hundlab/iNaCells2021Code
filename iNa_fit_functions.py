#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:37:41 2020

@author: grat05
"""
import numpy as np
import pandas as pd
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

def standardSolver(flat_durs, flat_voltages, run_model, dt):
    run = True
    t = 0
    i = 0
    tnext = flat_durs[i]
    vM = flat_voltages[i]
    iNa = []
    times = []
    vMs = []
    while run:
        while t >= tnext:
            i += 1
            if i >= len(flat_durs):
                run = False
                break
            tnext += flat_durs[i]
            vM = flat_voltages[i]
        iNa.append(run_model.update(vM, dt)) #, ret=ret
        times.append(t)
        vMs.append(vM)
        t += dt
    times = np.array(times)
    iNa = np.array(iNa)
    vMs = np.array(vMs)
    return times, iNa, vMs

class ModelWrapper():
    def __init__(self, flat_durs, flat_voltages, run_model):
        self.call_count = 0
        self.run_model = run_model
        self.times = np.array([sum(flat_durs[:i]) for i in range(len(flat_durs)+1)])
        self.flat_voltages = np.array(flat_voltages)
        self.t_end = self.times[-1]
    def getvOld(self, t):
        loc = np.searchsorted(self.times, t)-1
        loc = np.clip(loc, 0, len(self.flat_voltages)-1)
        vOld = self.flat_voltages[loc]
        return vOld
    def __call__(self, t, vals):
        if self.call_count > 20000 and self.call_count/t > 20:
            print("Error too many calls", self.call_count)
            print(t/self.call_count)
#            raise ValueError
        self.call_count += 1
        vOld = self.getvOld(t)
        ddt_vals =  self.run_model.ddtcalc(vals, vOld)
        return ddt_vals
    def jac(self, t, vals):
        vOld = self.getvOld(t)
        return self.run_model.jac(vOld)

def scipySolver(flat_durs, flat_voltages, run_model, solver, dt=None):
    max_step = np.min(flat_durs[flat_durs > 0])/2
    wrap_run_model = ModelWrapper(flat_durs, flat_voltages, run_model)
    
    if run_model.jac(-80) is not None:
        jac = wrap_run_model.jac
    else:
        jac = None
    try:
        res = solver(wrap_run_model, (0,wrap_run_model.t_end), run_model.state_vals,
                     first_step=dt, max_step = max_step, jac=jac, vectorized=True)
    except Exception as e:
        print("Model State:")
        print(wrap_run_model.run_model.lastVal)
        raise e
    if not res.success:
        message = res.message
        message += 'Solve IVP Failure\nModel State\n'
        message += str(wrap_run_model.run_model.lastVal)
        raise ValueError(message)

#    print(res)
    times = res.t
    vMs = wrap_run_model.getvOld(times)
    iNa = run_model.calcCurrent(res.y, vMs)
    return times, iNa, vMs

def solveAndProcesses(durs, voltages, run_model, solver, dt, process, sub_sim_pos):
    flat_durs = durs[sub_sim_pos,:]
    flat_voltages = voltages[sub_sim_pos,:]
    
    if solver is None:
        times, iNa, vMs = standardSolver(flat_durs, flat_voltages, run_model, dt)
    else:
        times, iNa, vMs = scipySolver(flat_durs, flat_voltages, run_model, solver,
                                      dt)
    times = np.array(times)
    iNa = np.array(iNa)
    vMs = np.array(vMs)
    
    if plot1:
        plt.figure()
        plt.subplot(311)
        plt.plot(times, iNa)
        plt.subplot(312)
        plt.plot(times, vMs)
        plt.subplot(313)
        lines = plt.plot(times, run_model.recArray)
        plt.legend(lines, list(run_model.recArrayNames))
        
    processed = process(times=times,current=iNa,vMs=vMs,sub_sim_pos=sub_sim_pos,durs=durs)


    return np.array(processed)

# def scipySolver(flat_durs, flat_voltages, run_model, solver, dt=None):
#     max_step = np.min(flat_durs[flat_durs > 0])/2
#     wrap_run_model = ModelWrapper(flat_durs, flat_voltages, run_model, multiple=True)
    
#     res = solver(wrap_run_model, (0,wrap_run_model.times[-1]), run_model.state_vals,
#                  first_step=dt, max_step = max_step)
# #    print(res)
#     times = res.t
#     vMs = wrap_run_model.getvOld(times)
#     iNa = run_model.calcCurrent(res.y, vMs)
#     return times, iNa, vMs

def run_sim(model_parameters, model, voltages, durs, sim_param, process, post_process,
            dt=0.005, solver=None, retOptions=None, pool=None):#ret = [True]*3
    out = []
    
    model_args = list(model_parameters)
    model_kwargs = {'TEMP': sim_param['TEMP'], 'naO': sim_param['naO'], 'naI': sim_param['naI']}
    for sub_sim_pos in range(voltages.shape[0]):
        run_model = model(*model_args, **model_kwargs)
        if not retOptions is None:
            run_model.retOptions = retOptions

        if pool is not None:
            processed = pool.apply_async(solveAndProcesses, 
                                                   (durs, voltages, \
                                                    run_model, solver, dt, \
                                                    process, sub_sim_pos))
        else:
            processed = solveAndProcesses(durs, voltages, run_model, \
                                          solver, dt,  process, sub_sim_pos)
        
        out.append(processed)
    
    yield
    if pool is not None:
        out2 = []
        for processed in out:
            out2.append(processed.get())
        out = out2
    out = np.array(out)
    if not post_process is None:
        out =  post_process(np.array(out))
    yield np.squeeze(out)

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

class SimResults():
    def __init__(self, calc_fn, sim_funcs, **kwargs):
        self.calc_fn = calc_fn
        self.sim_funcs = sim_funcs
        self.keywords = kwargs
#        self.cache = {}
        self.call_counter = 0
    def __call__(self, model_parameters_list, keys):
        model_parameters_dict = {key: np.array(model_parameters, dtype=float) 
                                 for key, model_parameters in 
                                 zip(keys,model_parameters_list)}
#        if not self.cache_args is None:
#            print(np.abs(self.cache_args- model_parameters))
        model_params_to_run = {}
        simfs_to_run = {}
        for key, model_parameters in model_parameters_dict.items():
            model_params_to_run[key] = model_parameters
            simfs_to_run[key] = self.sim_funcs[key]
#            prev_cache = np.array([np.array(mps) for c_key, mps in self.cache if c_key == key])
            
        new_cache = self.calc_fn(model_params_to_run,
                                      sim_funcs=simfs_to_run,
                                      **self.keywords)

        self.call_counter += 1
        print("Num calls: ", self.call_counter)

        res = []
        for key in keys:
            res += list(new_cache[key])
        res = np.array(res)
        return res

#sim_funcs is now a dict (pmid_fig, name) -> sim_func
def calc_results(model_parameters_part, model_parameters_full, sim_funcs,\
                       data, mp_locs=None, pool=None,\
                       exp_params=None,error_fill=np.inf):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    
    vals_sims = {}
    sims_iters = {}
    for key, sim_func in sim_funcs.items():
#        print(key)
        if isinstance(model_parameters_part, dict):
            model_parameters_full[mp_locs] = model_parameters_part[key]
        else:
            model_parameters_full[mp_locs] = model_parameters_part
        sims_iter = sim_func(model_parameters_full, pool=pool)
        sims_iters[key] = sims_iter
        next(sims_iter)
    for key in sim_funcs:
        sims_iter = sims_iters[key]
        try:
            vals_sim = next(sims_iter)
            vals_sims[key] = vals_sim
            if vals_sim is None:
                raise ValueError("sims_iter returned none")
        except Exception as e:
            print(e)
#            raise e
            sub_dat = data[key]
            vals_sims[key] = error_fill*np.ones(sub_dat.shape[0])
    
#    with np.printoptions(precision=3):
#        print(model_parameters_part)
    return vals_sims


def calc_diff(model_parameters_part, model_parameters_full, sim_func, data, mp_locs=None, l=1, ssq=False, **kwargs):
    if isList(sim_func) or isinstance(sim_func, dict):
        return calc_diff_multiple(model_parameters_part, model_parameters_full, sim_func, data, mp_locs, l, ssq, **kwargs)
    else:
        return calc_diff_single(model_parameters_part, model_parameters_full, sim_func, data, mp_locs, l, ssq, **kwargs)


def calc_diff_single(model_parameters_part, model_parameters_full, sim_func,\
                     data, mp_locs=None, l=1, ssq=False, results=None):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    model_parameters_full[mp_locs] = model_parameters_part
    sims_iter = sim_func(model_parameters_full)
    vals_sim = next(next(sims_iter))
    if plot2:
        plt.figure("Overall")
        plt.plot(data[:,0], vals_sim)
        plt.scatter(data[:,0], data[:,1])
    p_loss = 1/(model_parameters_full+1)-0.5
    error = np.concatenate((vals_sim-data[:,1]), l*p_loss)
    with np.printoptions(precision=3):
        print(model_parameters_part)
        print(0.5*np.sum(error**2),np.sum(p_loss**2))
    if not results is None:
        results.append((error,model_parameters_part))
    if ssq:
        return 0.5*np.sum(np.square(error))
    else:
        return error

def calc_diff_multiple(model_parameters_part, model_parameters_full, sim_func,\
                       data, mp_locs=None, l=1, ssq=False, pool=None,\
                       exp_params=None, keys=None, results=None,error_fill=np.inf):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    model_parameters_full[mp_locs] = model_parameters_part
    with np.printoptions(precision=3):
        print(model_parameters_part)
    error = []
    sims_iters = {}
    for key, sim_f_sing in sim_func.items():
        sims_iter = sim_f_sing(model_parameters_full, pool=pool)
        sims_iters[key] = sims_iter
        next(sims_iter)
    for key in sim_func:
        sub_dat = data[key]
        sims_iter = sims_iters[key]
        try:
            vals_sim = next(sims_iter)
            error += list((sub_dat[:,1] - vals_sim))
            if plot2:
                if exp_params is None:
                    plt.figure("Overall")
                else:
                    figname = exp_params.loc[key, 'Sim Group']
                    figname = figname if not pd.isna(figname) else 'Missing Label'
                    plt.figure(figname)
                plt.plot(sub_dat[:,0], vals_sim, label=key)
                plt.scatter(sub_dat[:,0], sub_dat[:,1])
                plt.legend()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
#            raise e
            error += [error_fill]*sub_dat.shape[0]
    # if not pool is None:
    #     vals_sims_res = []
    #     for i in range(len(sim_func)):
            
    #         vals_sims_res.append(pool.apply_async(sim_func[i], (model_parameters_full,)))
    #     vals_sims = [res.get() for res in vals_sims_res]
 

    # for i in range(len(sim_func)):
    #     sub_dat = data[i]
    #     if not pool is None:
    #         vals_sim = vals_sims[i]
    #     else:
    #         sim_f_sing = sim_func[i]
    #         vals_sim = sim_f_sing(model_parameters_full)
    #     error += list((sub_dat[:,1] - vals_sim))
    #     if plot2:
    #         if exp_params is None or keys is None:
    #             plt.figure("Overall")
    #         else:
    #             plt.figure(exp_params.loc[keys[i], 'Sim Group'])
    #         plt.plot(sub_dat[:,0], vals_sim)
    #         plt.scatter(sub_dat[:,0], sub_dat[:,1])
    error = np.array(error)
    p_loss = 1/(model_parameters_full+1)-0.5
    error = np.concatenate((error, l*p_loss))
    with np.printoptions(precision=3):
#        print(model_parameters_part)
        print(0.5*np.sum(error**2),np.sum((l*p_loss)**2))
    if not results is None:
        results.append((error,model_parameters_part))
    if ssq:
        return 0.5*np.sum(np.square(error))
    else:
        return error

#from functools import partial
##voltages = np.arange(-100,100, 10)
#voltagesAll = [-120, -20, -140, -20]
#durs = np.arange(0,2,0.1)
#dursAll = [100, 100, durs, 10]
#normedCurrs = run_sim(np.ones(22), Koval_ina, voltagesAll, dursAll, {'TEMP':310, 'naO': 140.0, 'naI':8.355}, normalize2prepulse)
#partial(calcExpTauInact,func=monoExp,x0=np.ones(3))
