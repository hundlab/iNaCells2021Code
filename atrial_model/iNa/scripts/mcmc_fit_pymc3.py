#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:05:59 2020

@author: grat05
"""

import pymc3 as pm
import arviz as az
import datetime
import numpy as np
import pickle
from threading import Timer
from multiprocessing import Manager
from functools import partial
import os
import warnings
from copy import deepcopy

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

from atrial_model.parse_cmd_args import args
import atrial_model.run_sims_functions
from atrial_model.run_sims import calc_results, SimResults
from atrial_model.iNa.define_sims import sim_fs, datas, keys_all, exp_parameters
from atrial_model.iNa.stat_model_3 import StatModel, key_frame
from atrial_model.iNa.model_setup import model_params_initial, mp_locs, sub_mps, model
from atrial_model.pymc3step import MultiMetropolis, MultivariateNormalProposal,\
    RowwiseMultivariateNormalProposal, NormalProposal
from atrial_model.pymc3stepMetrop import DEMetropolisJoint, DEMetropolisZ

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_last(db):
    last_draw_idx = db.indexes['draw'][-1]
    last = [db.sel(dict(chain=chain_idx, draw=last_draw_idx))
                for chain_idx in db.indexes['chain']]
    last =  [{key: np.squeeze(np.array(val)) 
                for key, val in chain.items()}
                    for chain in last]
    return deepcopy(last)

def add_transformed(start, stat_model):
    start2 = []
    for chain in start:
        t_chain = deepcopy(chain)
        start2.append(t_chain)
        for key, val in  chain.items():
            try:
                var = stat_model[key]
                if hasattr(var, 'transformation'):
                    transformed_name = var.name+'_'+\
                        var.transformation.name+'__'
                    transformed_val = var.transformation.forward_val(val.copy())
                    t_chain[transformed_name] = transformed_val
            except KeyError:
                print(key, " is missing")
    return start2

if __name__ == '__main__':
    
    atrial_model.run_sims_functions.plot1 = False #sim
    atrial_model.run_sims_functions.plot2 = False #diff
    atrial_model.run_sims_functions.plot3 = False #tau

    
    model_name = './mcmc_'
    model_name +=  args.model_name
    model_name += '_{cdate.month:02d}{cdate.day:02d}_{cdate.hour:02d}{cdate.minute:02d}'
    model_name = model_name.format(cdate=datetime.datetime.now())
    model_name += '{}.{}'
    db_path = args.out_dir+'/'+model_name
    trace_db_path = db_path.format('_trace', 'nc')
    main_db_path = db_path.format('', 'pickle')
    
    model_db = {}
#    model_db['model_params_initial'] = model_params_initial
    model_db['mp_locs'] = mp_locs
    model_db['param_bounds'] = model.param_bounds
    model_db['bio_model_name'] = args.model_name
    model_db['trace_obj_path'] = trace_db_path
    
    traceobj_db = {}
    traceobj_db['main_db_path'] = main_db_path
    
    n_chains = 1
    n_cores = 1

    with Manager() as manager:
#    print("Running Pool with", os.cpu_count(), "processes")
        with manager.Pool() as proc_pool:
    #        proc_pool = None
            
            calc_fn = partial(calc_results, model_parameters_full=model_params_initial,\
                            mp_locs=mp_locs, data=datas,error_fill=0,\
                            pool=proc_pool)
            run_biophysical = SimResults(calc_fn=calc_fn, sim_funcs=sim_fs, disp_print=False)
            key_frame = key_frame(keys_all, exp_parameters)
            
            with StatModel(run_biophysical, key_frame, datas,
                                    mp_locs, model) as stat_model:
                
                cancel_sim = manager.Value('cancel_sim', False)
            
                logps = {}
                for obs in stat_model.observed_RVs:
                    logps[obs.name] = obs.logp_elemwise
            
                logps_trace = {name: [] for name in logps}
                total_logp_trace = []
                diverge = []
                
                def stop_sim_callback(trace, draw, **kwargs):

                    total_logp = 0
                    for name in logps:
                        part_logp = logps[name](draw.point)
                        logps_trace[name].append((draw.chain, part_logp))
                        total_logp += np.sum(part_logp)
                    total_logp_trace.append((draw.chain, total_logp))

                    # if len(diverge) != len(draw.stats):
                    #     diverge = [int(stat['diverging']) for stat in draw.stats]
                    # else:
                    #     diverge = [diverge[i] + int(draw.stats[i]['diverging'])
                    #                for i in range(len(diverge))]
                    try:
                        print([int(stat['diverging']) for stat in draw.stats], end=' ')
                    except Exception:
                        pass

                    end = ' '
                    if draw.chain == n_chains-1:
                        end = '\n'
                    print(total_logp, end=end, flush=True)
                    
                    if cancel_sim.get() and draw.chain == n_chains-1:
                        print("Sampling Interrupted")
                        raise KeyboardInterrupt
    #                else:
    #                    print("Sampling")
                
                def stop_sim(pymc_model):
                    cancel_sim.set(True)
                    print("Sampling Canceled")
                

                old_trace = None
                start = None
                proposal_cov = None
                scalings = None
                if not args.previous_run is None:
                    model_db['previous'] = {}
                    model_db['previous']['file'] = args.previous_run
                    model_db['previous']['is_trace'] = not args.previous_run_manual
                    with open(args.previous_run,'rb') as file:
                        db_full = pickle.load(file)
                    if args.previous_run_manual:
                        db = db_full['trace']
                        proposal_cov = db_full['proposal_cov']
                        scalings = db_full['scalings']
                        try:
                            start = get_last(db.posterior)#warmup_
                        except AttributeError:
                            start = get_last(db.posterior)
                        start = add_transformed(start, stat_model)
                    else:
                        old_trace = db_full['trace']
                    del db
                
                #set model params to random points
                if False:
                    for chain in start:
                        for key, val in chain.items():
                            if 'mp ' in key:
                                val[:] = np.random.uniform(low=-2,high=2,size=len(mp_locs))
                
##                step1 = pm.Metropolis(vars=stat_model.model_param)
#                sd = np.load(args.out_dir+'/model_param.npy')
#                proposal_dist = NormalProposal(sd)
 #               #proposal_dist = MultivariateNormalProposal(np.identity(len(mp_locs)), 
  #              #                                          len(key_frame))
   #             #proposal_dist = RowwiseMultivariateNormalProposal(
    #            #    np.tile(np.identity(len(mp_locs)), (len(key_frame),1,1)))
#                step1 = MultiMetropolis(vars=stat_model.model_param, 
#                                        proposal_dist=proposal_dist)#S=sd)

                step = []
                
                groups = {}
                for key in key_frame.index:
                    group_name = key_frame.loc[key, 'Sim Group']
                    group = set(key_frame.index[key_frame['Sim Group'] == group_name])
                    key_paper = key[0].split('_')[0]
                    paper = {o_key 
                             for o_key in key_frame.index 
                             if o_key[0].split('_')[0] == key_paper}
                    group_all = list(map(lambda x: "mp "+str(x),
                             group.union(paper) - {key}))
                        
                    groups["mp "+str(key)] = group_all
                proposal_dist = MultivariateNormalProposal#(0.1*np.identity(len(mp_locs)), 1)
                step1 = MultiMetropolis(vars=stat_model.model_param, 
                                        proposal_dist=proposal_dist,
                                        S=proposal_cov,
                                        scaling=scalings,
                                        tune_interval=150,
                                        key_groups=groups,
                                        alt_step_interval=-1,
                                        tune=False)
#                step1 = DEMetropolisJoint(vars=stat_model.model_param, tune=None)#"lambda"
                step.append(step1)
                
                # step2 = pm.NUTS(vars=stat_model.error_sd, k=0.95, t0=40, step_scale=0.05)
                # step.append(step2)
                
                # step2 = pm.NUTS(vars=stat_model.model_param_sd, k=0.95, t0=40, step_scale=0.05)
                # step.append(step2)
                
                # step2 = pm.NUTS(vars=stat_model.paper_eff_sd, k=0.95, t0=40, step_scale=0.05)
                # step.append(step2)
                
#                step4 = pm.NUTS(vars=[stat_model.error_sd], step_scale=0.02)
#                step.append(step4)

                # step2 = DEMetropolisZ(vars=stat_model.error_sd, history=db.posterior)
                # step.append(step2)
                
                # step3 = DEMetropolisZ(vars=stat_model.model_param_sd, history=db.posterior)
                # step.append(step3)
                
                # step3 = DEMetropolisZ(vars=stat_model.paper_eff, history=db.posterior)
                # step.append(step3)
                
                # step4 = DEMetropolisZ(vars=[stat_model.b_temp, 
                #                        stat_model.model_param_intercept], history=db.posterior)
                # step.append(step4)
                

                
#                for var in pm.inputvars(stat_model.model_param):
#                    step1 = pm.DEMetropolis(vars=var, tune='lambda')
#                    step.append(step1)
                other_vars = list(set(stat_model.vars) - set(pm.inputvars(stat_model.model_param)))
#                for var in other_vars:
#                    step_other = pm.DEMetropolis(vars=var)
#                    step.append(step_other)
                #step2 = DEMetropolisZ(vars=other_vars)
                #step.append(step2)
                step2 = pm.NUTS(vars=other_vars, step_scale=0.02, max_treedepth=12, target_accept=0.99)
                step.append(step2)
#                step3 = DEMetropolisZ(stat_model.paper_eff)
#                step.append(step3)
                # step2 = pm.NUTS(vars=[stat_model.paper_eff,
                #                       stat_model
                #                       stat_model.b_temp, 
                #                       stat_model.model_param_intercept,
                #                       stat_model.model_param_chol],
                #                 step_scale=0.02,
                #                 max_treedepth=12)
                # step.append(step2)
                # step2d = pm.NUTS(vars=[#stat_model.paper_eff, 
                #        stat_model.b_temp], 
                #        #stat_model.model_param_intercept],
                #  step_scale=0.02,
                #  max_treedepth=12)
                # step.append(step2d)
                # step2b = pm.NUTS(vars=[stat_model.paper_eff_sd], 
                #                        step_scale=0.02,
                #                        max_treedepth=12)
                # step.append(step2b)
                # step2c = pm.NUTS(vars=[stat_model.paper_eff_stand], 
                #        step_scale=0.02,
                #        max_treedepth=12)
                # step.append(step2c)
                # step3 = pm.NUTS(vars=[stat_model.model_param_chol], step_scale=0.02)
                # step.append(step3)
#                step4 = pm.NUTS(vars=[stat_model.error_sd], step_scale=0.02)
#                step.append(step4)

                if not args.max_time is None:
                    #max_time is in hours
                    sample_timer = Timer(args.max_time*60*60, stop_sim, args=(stat_model,))
                    sample_timer.start()
                    
#                import pdb
#                pdb.set_trace()
                trace = pm.sample(draws=20000, tune=500, discard_tuned_samples=False,
                                  return_inferencedata=False, #this does not work if set to T
                                  compute_convergence_checks=False,
                                  trace=old_trace, start=start,
                                  cores=n_cores, chains=n_chains, step=step,
                                  callback=stop_sim_callback)
        
                if not args.max_time is None:
                    sample_timer.cancel()
                    sample_timer.join()
                
#                traceobj_db['trace'] = trace
#                traceobj_db['step'] = step

        #supress output
        #logliklihood is not actually being calculated
        #lp is still valid (I think)
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            az_trace = az.from_pymc3(trace, save_warmup=True)
            sys.stdout = old_stdout
        del calc_fn.keywords['pool']

        proposals = None
        scalings = None
        trace_deltas = None
        trace_covs = None
        try:
            trace_deltas = step1.getTraceDeltas()
            trace_covs = step1.getTraceCovs()
            proposals = {name: step1.steps[name].proposal_dist.s for name in step1.steps.keys()}
            scalings = {name: step1.steps[name].scaling for name in step1.steps.keys()}
        except Exception:
            print("Getting trace_deltas failed")
        
        az.to_netcdf(az_trace, trace_db_path)
        
        model_db['num_calls'] = run_biophysical.call_counter
        model_db['key_frame'] = key_frame
        model_db['trace'] = az_trace
        model_db['proposal_cov'] = proposals
        model_db['scalings'] = scalings
#        model_db['trace_deltas'] = trace_deltas
#        model_db['trace_covs'] = trace_covs
        model_db['logps_trace'] = logps_trace
        model_db['total_logp_trace'] = total_logp_trace

#        model_param_step = dict(scaling = step1.scaling,
#                                s = step1.proposal_dist.s)
#        model_db['step'] = dict(model_param = model_param_step)
        
        
        with open(main_db_path, 'wb') as file:
            pickle.dump(model_db, file)
        
#        with open(trace_db_path, 'wb') as file:
#            pickle.dump(traceobj_db, file)
            
        print("Pickle File Written to:")
        print("Main:", main_db_path)
#        print("pymc specific objects:", trace_db_path)
        #pymc.Matplot.plot(S)
