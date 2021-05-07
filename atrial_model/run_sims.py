#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:46:28 2020

@author: grat05
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import copy

from . import run_sims_functions
from .run_sims_functions import isList

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
        if self.call_count > 100000 and self.call_count/t > 80:
            print("Error too many calls", self.call_count/t, self.call_count)
            self.call_count = 1
            raise ValueError
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
                     first_step=dt, max_step = max_step, jac=jac)#, vectorized=True)
    except Exception as e:
        print("Model State:")
        print(wrap_run_model.run_model.lastVal)
        raise e
    if not res.success:
        message = res.message
        message += 'Solve IVP Failure\nModel State\n'
        message += str(wrap_run_model.run_model.lastVal)
        raise ValueError(message)

#    print(len(res.t)/res.t[-1])
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
    
    if run_sims_functions.plot1:
        fig = plt.figure()
        ax = [fig.add_subplot(3,1,1)]
        ax.append(fig.add_subplot(3,1,2, sharex=ax[0]))
        ax.append(fig.add_subplot(3,1,3, sharex=ax[0]))
        ax[0].plot(times, iNa)
        ax[1].plot(times, vMs)
        lines = ax[2].plot(times, run_model.recArray)
        ax[2].legend(lines, list(run_model.recArrayNames))
        
    processed = process(times=times,current=iNa,vMs=vMs,sub_sim_pos=sub_sim_pos,durs=durs)


    return np.array(processed)

class SimRunner():
    class SimRes():
        def __init__(self, parent, pooled=False):
            self.parent = parent
            
            self.out = []
            self.pooled = pooled
            self.exception = None
            
        def get_output(self):
            return self.parent.get_output(self)
            
    def __init__(self, model, voltages, durs, sim_param, process, post_process,
            dt=0.005, solver=None, retOptions=None):
        self.model = model
        self.voltages = voltages
        self.durs = durs
        self.sim_param = sim_param
        self.process = process
        self.post_process = post_process
        self.dt = dt
        self.solver = solver
        self.retOptions = retOptions
        
    def run_sim(self, model_parameters, pool=None):
        res = self.SimRes(self, pooled=not pool is None)

        model_args = list(model_parameters)
        model_kwargs = dict(TEMP=self.sim_param['TEMP'],
                            naO=self.sim_param['naO'],
                            naI=self.sim_param['naI'])
        try:
            for sub_sim_pos in range(self.voltages.shape[0]):
                run_model = self.model(*model_args, **model_kwargs)
                if not self.retOptions is None:
                    run_model.retOptions.update(self.retOptions)
                
                solveAProc_args = (self.durs, self.voltages, \
                                   run_model, self.solver, self.dt, \
                                   self.process, sub_sim_pos)
                if pool is not None:
                    processed = pool.apply_async(solveAndProcesses, solveAProc_args)
                else:
                    processed = solveAndProcesses(*solveAProc_args)
                res.out.append(processed)

        except Exception as e:
            res.exception = e
            raise e
        
        return res
    
    def get_output(self, res):
        if not res.exception is None:
            raise res.exception
        out = res.out
        
        if res.pooled:
            out2 = []
            for processed in out:
                out2.append(processed.get())
            out = out2
        out = np.array(out)
        if not self.post_process is None:
            out =  self.post_process(np.array(out))
        return np.squeeze(out)
        

# def run_sim(model_parameters, model, voltages, durs, sim_param, process, post_process,
#             dt=0.005, solver=None, retOptions=None, pool=None):
#     out = []
    
#     model_args = list(model_parameters)
#     model_kwargs = {'TEMP': sim_param['TEMP'], 'naO': sim_param['naO'], 'naI': sim_param['naI']}
#     for sub_sim_pos in range(voltages.shape[0]):
#         run_model = model(*model_args, **model_kwargs)
#         if not retOptions is None:
#             run_model.retOptions = retOptions

#         if pool is not None:
#             processed = pool.apply_async(solveAndProcesses, 
#                                                    (durs, voltages, \
#                                                     run_model, solver, dt, \
#                                                     process, sub_sim_pos))
#         else:
#             processed = solveAndProcesses(durs, voltages, run_model, \
#                                           solver, dt,  process, sub_sim_pos)
        
#         out.append(processed)
    
#     yield
#     if pool is not None:
#         out2 = []
#         for processed in out:
#             out2.append(processed.get())
#         out = out2
#     out = np.array(out)
#     if not post_process is None:
#         out =  post_process(np.array(out))
#     yield np.squeeze(out)



class SimResults():
    def __init__(self, calc_fn, sim_funcs, disp_print=True, **kwargs):
        self.calc_fn = calc_fn
        self.sim_funcs = sim_funcs
        self.keywords = kwargs
        self.cache_len = 5
        # [key][i < cache_len] = (params, results)
        self.cache = {}
        self.call_counter = 0
        self.disp_print = disp_print
    def __call__(self, model_parameters_list, keys, flatten=True):
        model_parameters_dict = {key: np.array(model_parameters, dtype=float) 
                                 for key, model_parameters in 
                                 zip(keys,model_parameters_list)}
        misses, results = self.lookup_cache(model_parameters_dict)

        model_params_to_run = {}
        simfs_to_run = {}
        for key in misses:
            model_params_to_run[key] = model_parameters_dict[key]
            simfs_to_run[key] = self.sim_funcs[key]
#            prev_cache = np.array([np.array(mps) for c_key, mps in self.cache if c_key == key])
            
        new_results = self.calc_fn(model_params_to_run,
                                      sim_funcs=simfs_to_run,
                                      **self.keywords)
        
        self.update_cache(model_parameters_dict, new_results)

        self.call_counter += 1
        if self.disp_print:
            print("Num calls: ", self.call_counter, "Num Run: ", len(new_results))
            if len(new_results) == 1:
                print(list(new_results.keys()))

        results.update(new_results)
        res = []
        for key in keys:
            if flatten:
                res += list(results[key])
            else:
                res.append(results[key])
        if flatten:
            res = np.array(res)
        else:
            res = copy.deepcopy(res)
        return res
    
    def update_cache(self, model_parameters, results):
        for key, res in results.items():
            if not key in self.cache:
                self.cache[key] = deque(maxlen=self.cache_len)
            self.cache[key].append((model_parameters[key], res))
            
    def lookup_cache(self, model_parameters):
        misses = []
        results = {}
        for key, model_p_new in model_parameters.items():
            try:
                cached = self.cache[key]
                found = False
                for model_p_cache, res_cache in cached:
                    if np.array_equal(model_p_new, model_p_cache):
                        results[key] = res_cache
                        found = True
                        break
                if not found:
                    misses.append(key)
            except KeyError:
                misses.append(key)
                
        return misses, results

#sim_funcs is now a dict (pmid_fig, name) -> sim_func
def calc_results(model_parameters_part, model_parameters_full, sim_funcs,\
                       data, mp_locs=None, pool=None,\
                       exp_params=None,error_fill=np.inf):
    if mp_locs is None:
        mp_locs = np.ones_like(model_parameters_full, dtype=bool)
    
    vals_sims = {}
    sims_res = {}
    for key, sim_func in sim_funcs.items():
#        print(key)
        if isinstance(model_parameters_part, dict):
            model_parameters_full[mp_locs] = model_parameters_part[key]
        else:
            model_parameters_full[mp_locs] = model_parameters_part
        sims_res[key] = sim_func.run_sim(model_parameters_full, pool=pool)
    for key, sim_res in sims_res.items():
        try:
            vals_sim = sim_res.get_output()
            vals_sims[key] = vals_sim
            if vals_sim is None:
                raise ValueError("simulation returned none")
        except Exception as e:
            print(e)
#            if raise_err:
#                raise e
            try:
                sub_dat = data[key]
                vals_sims[key] = error_fill*np.ones(sub_dat.shape[0])
            except Exception:
                 vals_sims[key] = error_fill*np.ones(1)
    return vals_sims

def calc_diff(model_parameters_part, model_parameters_full, sim_func,\
                       data, mp_locs=None, ssq=False, pool=None,\
                       exp_params=None, keys=None, results=None,error_fill=np.inf,\
                       prnt_err=True):
    
    vals_sims = calc_results(model_parameters_part, model_parameters_full, sim_func,\
                       data, mp_locs=mp_locs, pool=pool,\
                       exp_params=exp_params,error_fill=np.nan)
      
    error = []
    for key, vals_sim in vals_sims.items():
        sub_dat = data[key]
        error += list((sub_dat[:,1] - vals_sim))
    error = np.array(error)
    error[np.isnan(error)] = error_fill
    
#    with np.printoptions(precision=3):
#        print(model_parameters_part)
    if prnt_err:
        print(0.5*np.sum(np.square(error)))
    if not results is None:
        results.append((error,model_parameters_part))
    if run_sims_functions.plot2:
        plot_results(vals_sims, data, exp_params)
    if ssq:
        return 0.5*np.sum(np.square(error))
    else:
        return error
    
    #    

def plot_results(vals_sims, data, exp_params=None):
    for key, vals_sim in vals_sims.items():
        sub_dat = data[key]
        if exp_params is None:
            plt.figure("Overall")
        else:
            figname = exp_params.loc[key, 'Sim Group']
            figname = figname if not pd.isna(figname) else 'Missing Label'
            plt.figure(figname)
        plt.plot(sub_dat[:,0], vals_sim, label=key)
        plt.scatter(sub_dat[:,0], sub_dat[:,1])
        plt.legend()
