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
plot2 = True  #diff
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


from sklearn.preprocessing import minmax_scale
def calcExpTauInact(times, current, func, x0, sub_sim_pos, durs, calc_dur = 1, keep=None, bounds=(-np.inf, np.inf), **kwargs):
    if keep is None:
        keep = np.ones_like(x0, dtype=bool)
    flat_durs = flatten_durs(durs, sub_sim_pos)
    startt = np.sum(flat_durs[:calc_dur])
    stopt  = startt + flat_durs[calc_dur]
    stimMask = (startt < times) & (times < stopt)
    peak_loc = np.argmax(np.abs(current[stimMask]))
    baseline_loc = np.argmin(np.abs(current[stimMask]))
    peak_val = current[stimMask][peak_loc]
    baseline_val = current[stimMask][baseline_loc]
    cut_val = peak_val - (peak_val - baseline_val)*.1
    cut_loc = np.where(np.abs(current[stimMask][peak_loc:]) < np.abs(cut_val))[0][0]
    cut_loc = np.where(stimMask)[0][peak_loc+cut_loc]
    
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
        vOld = self.getvOld(t)
        ddt_vals =  self.run_model.ddtcalc(vals, vOld)
        return ddt_vals

def scipySolver(flat_durs, flat_voltages, run_model, solver, dt=None):
    max_step = np.min(flat_durs[flat_durs > 0])/2
    wrap_run_model = ModelWrapper(flat_durs, flat_voltages, run_model)
    
    res = solver(wrap_run_model, (0,wrap_run_model.t_end), run_model.state_vals,
                 first_step=dt, max_step = max_step)#, vectorized=True)
#    print(res)
    times = res.t
    vMs = wrap_run_model.getvOld(times)
    iNa = run_model.calcCurrent(res.y, vMs)
    return times, iNa, vMs

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

def run_sim(model_parameters, model, voltages, durs, sim_param, process,\
            dt=0.005, solver=None, retOptions=None):#ret = [True]*3
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
        model_args = list(model_parameters)
        model_kwargs = {'TEMP': sim_param['TEMP'], 'naO': sim_param['naO'], 'naI': sim_param['naI']}
        run_model = model(*model_args, **model_kwargs)
        if not retOptions is None:
            run_model.retOptions = retOptions
            
        flat_durs = np.clip(flatten_durs(durs, x), a_min=0, a_max=None)
        flat_voltages = flatten_durs(voltages, x)

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
        try:
            processed = process(times=times,current=iNa,vMs=vMs,sub_sim_pos=x,durs=durs)
        except:
            processed = np.nan
        out.append(processed)
        
    #        if plot1 or plot3:
    #            plt.show()
    return np.array(out)

def setup_sim(model, data, exp_parameters, process, dt=0.005, hold_dur=1, sim_args={}): #ret = [True]*3
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
                     sim_param=sim_param, process=process, dt=dt,\
                     **sim_args)
    return voltages, durs, f_call


def get_exp_y(data, exp_parameters):
    curr_real = data[:,1]
    capacitance = exp_parameters['capacitance (pF)']
    if not np.isnan(capacitance):
        curr_real = curr_real*1000/capacitance
    return curr_real

def calc_diff(model_parameters_part, model_parameters_full, sim_func, data, mp_locs=None, l=1, ssq=False, **kwargs):
    if isList(sim_func):
        return calc_diff_multiple(model_parameters_part, model_parameters_full, sim_func, data, mp_locs, l, ssq, **kwargs)
    else:
        return calc_diff_single(model_parameters_part, model_parameters_full, sim_func, data, mp_locs, l, ssq, **kwargs)


def calc_diff_single(model_parameters_part, model_parameters_full, sim_func,\
                     data, mp_locs=None, l=1, ssq=False, results=None):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    model_parameters_full[mp_locs] = model_parameters_part
    vals_sim = sim_func(model_parameters_full)
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
                       exp_params=None, keys=None, results=None):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    model_parameters_full[mp_locs] = model_parameters_part
    with np.printoptions(precision=3):
        print(model_parameters_part)
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
        error += list((sub_dat[:,1] - vals_sim))
        if plot2:
            if exp_params is None or keys is None:
                plt.figure("Overall")
            else:
                plt.figure(exp_params.loc[keys[i], 'Sim Group'])
            plt.plot(sub_dat[:,0], vals_sim)
            plt.scatter(sub_dat[:,0], sub_dat[:,1])
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