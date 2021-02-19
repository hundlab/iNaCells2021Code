#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:10:27 2020

@author: grat05
"""

import pymc3 as pm
import numpy as np
import pandas as pd
import theano
import theano.tensor as tt
from contextlib import contextmanager

from atrial_model.pymc3mvNorm import MyMvNorm

def key_frame(key_groups, exp_parameters):
    keys_list = [key for keys in key_groups for key in keys]
    return exp_parameters.loc[keys_list, ['Sim Group', 'temp ( K )']]


class BiophysicalModel(tt.Op):
#    itypes = [tt.dmatrix]
#    otypes = [tt.dvector]
 
    def __init__(self, keys_list, run_biophysical):
        self.keys_list = keys_list
        self.run_biophysical = run_biophysical
        super(BiophysicalModel, self).__init__()

    def make_node(self, *inputs):

        outputs = [tt.dvector() for key in self.keys_list]
        return theano.Apply(self, inputs, outputs)
    
    def perform(self, node, inputs, outputs):
        model_param_sim, = inputs
        results = self.run_biophysical(model_param_sim, self.keys_list, flatten=False)
        for i,res in enumerate(results):
            outputs[i][0] = res

@contextmanager
def StatModel(run_biophysical, key_frame, datas, mp_locs, model):
    ## create all sim means and variances per biophys model parameter
    with pm.Model() as model:
        num_sims = len(key_frame)
    
        
        # overall model parameter mean prior ~N
        mu = np.zeros(mp_locs.shape)
        sigma = np.ones(mp_locs.shape)*100
        mu[18] = 1.7
        sigma[18] = 0.1
#        model_bounds = np.array(model.param_bounds)
        model_param_intercept = pm.Normal("model_param_intercept",
                                   mu = mu,
                                   sigma = sigma,
                                   shape = mp_locs.shape)
        
        # temerature eff
        #temperature adjusted to minimum in group
        temperature_arr = np.array(key_frame['temp ( K )'], dtype=float) -290.15



        #temperature coefficiant ~ N
        mu = np.zeros(mp_locs.shape)# update to fit q10 (0.2)
        simga = np.ones(mp_locs.shape)*100 # .1
        mu[[1,4,10,14,21]] = -0.7/10
        simga[[1,4,10,14,21]] = .1
        mu[[6,9,11,15,20,22]] = 0.4
        mu[3] = -0.4
        simga[[3,6,9,11,15,20,22]] = .1
        mu[18] = 0
        simga[18] = .1

#        mu = 0# update to fit q10 (0.2)
#        sigma = 100 # .1
        b_temp = pm.Normal("b_temp",
                                   mu = mu,
                                   sigma = sigma,
                                   shape = mp_locs.shape)
        
        # paper effect
        paper_idx = {}
        curr_idx = 0
        sim_idx = []
        for key in key_frame.index:
            pubmed_id = int(key[0].split('_')[0])
            if not pubmed_id in paper_idx:
                paper_idx[pubmed_id] = curr_idx
                curr_idx += 1
            sim_idx.append(paper_idx[pubmed_id])
        sim_paper_idx = np.array(sim_idx)
            
        alpha = 0.001*np.ones(len(mp_locs), dtype=float)
        beta = 0.001*np.ones(len(mp_locs), dtype=float)
        alpha[18] = 11
        beta[18] = 1
        testval = np.ones(mp_locs.shape, dtype=float)
        paper_eff_sd = pm.InverseGamma("paper_eff_sd", 
                              alpha = alpha,
                              beta = beta,
                              shape = mp_locs.shape,
                              testval = testval)
        mu = np.zeros((len(paper_idx), len(mp_locs)))
        sd_idx = np.tile(np.arange(len(mp_locs)), (len(paper_idx), 1))
        paper_eff = pm.Normal("paper_eff",
                                    mu = mu,
                                    sigma = paper_eff_sd[sd_idx],
                                    shape = mu.shape)
        
        
        # overall model parameter stdev prior ~Gamma
        # alpha = 0.001
        # beta = 0.001
        # alpha = 0.001* np.ones_like(mp_locs)
        # beta = 0.001* np.ones_like(mp_locs)
        # testval = np.ones(mp_locs.shape, dtype=float)
        # model_param_sd = pm.InverseGamma("model_param_sd", 
        #                       alpha = alpha,
        #                       beta = beta,
        #                       shape = mp_locs.shape,
        #                       testval = testval)
        
        # overall covariance
        alpha = 0.001*np.ones(len(mp_locs), dtype=float)
        beta = 0.001*np.ones(len(mp_locs), dtype=float)
        alpha[18] = 11
        beta[18] = 1
        sd_dist = pm.InverseGamma.dist(alpha = alpha,
                                       beta = beta,
                                       shape=len(mp_locs),
                                       testval = 1.)
        model_param_chol, corr, stds = pm.LKJCholeskyCov("model_param_chol",
                                  n = len(mp_locs),
                                  eta = 1, #no prior on sparsity
                                  sd_dist = sd_dist,
                                  compute_corr=True
                                  )
        
        model_param_corr = pm.Deterministic('model_param_corr', corr)
        model_param_sd = pm.Deterministic('model_param_sd', stds)
        
        #linear mean
        model_param_index = np.arange(start=0,stop=len(mp_locs),step=1,dtype=int)
        model_param_index = np.tile(model_param_index, (num_sims,1))
        mean = model_param_intercept[model_param_index] +\
            b_temp[model_param_index]*temperature_arr[...,None] +\
            paper_eff[sim_paper_idx,:]


        # model_param = MyMvNorm("model_param",
        #                        mean=mean,
        #                        chol_cov=model_param_chol,
        #                        shape=model_param_index.shape)
        model_params = []
        for i,key in enumerate(key_frame.index):
            model_param_sing = pm.MvNormal('mp '+str(key),
                                           mu=mean[i],
                                           chol=model_param_chol,
                                           shape=len(mp_locs))
            model_params.append(model_param_sing)
        model_param = pm.Deterministic('model_param', tt.stack(model_params))
        # model parameter  ~ N
        # model_param = pm.Normal("model_param",
        #                        mu=mean,
        #                        sigma=model_param_sd[model_param_index],
        #                        shape=model_param_index.shape)

        # model_param_raw =  pm.Normal("model_param_raw",
        #                                  mu = 0,
        #                                  sigma = 1,
        #                                  shape = model_param_index.shape)
        
        # model_param = pm.Deterministic('model_param',
        #                                mean + model_param_chol.dot(model_param_raw.T).T)

    
    
        # st dev for each protocol ~ Gamma
        key_groups = list(key_frame['Sim Group'].unique())
        testval = np.ones(len(key_groups), dtype=float)
        alpha = 0.001
        beta = 0.001
        error_sd = pm.InverseGamma("error_sd",
                                  alpha = alpha,
                                  beta = beta,
                                  shape = len(key_groups),
                                  testval=testval)
        
        tt_biophysical_model = BiophysicalModel(list(key_frame.index), run_biophysical)
    
        biophysical_res = tt_biophysical_model(model_param)
#        import pdb
#        pdb.set_trace()
        # liklihood ~N
        
#        error_sigma_index = []
#        data_array = []
        biophys_liks = []
        for i,key in enumerate(key_frame.index):
#            exp_data = list(datas[key][:,1])
#            data_array += exp_data

            group_num = key_groups.index(key_frame.loc[key, 'Sim Group'])
            lik_name = 'lik '+str(key)
            data = datas[key]
            
            biophys_lik = pm.Normal(lik_name,
                                    mu = biophysical_res[i],
                                    sigma = error_sd[group_num],
                                    observed = data[:,1])
            biophys_liks.append(biophys_lik)
#            error_sigma_index += [group_num]*len(exp_data)
            if key_frame.loc[key,'Sim Group'] == 'tau activation':
                correction = np.sqrt(len(datas[key][:,1]))
                logpt = model[lik_name].logpt
                pm.Potential('pot '+str(key),
                             -logpt+logpt/correction)

#        error_sigma_index = np.array(error_sigma_index)
#        data_array = np.array(data_array)
                  

    
        # biophys_lik = pm.Normal('lik',
        #                               mu=biophys_result,
        #                               sigma=error_sd[error_sigma_index],
        #                               observed=data_array)
        
        # coords={'model_parameters': model_param_names,
        #   'simulation_names': list(key_frame.index),
        #   'group_names': key_frame['Sim Group'].unique(),
        #   'paper_id' : paper_id
        # }
        # dims = {'model_param_intercept': ['model_parameters'],
        #     'model_param': ['simulations_names', 'model_parameters'], 
        #     'b_temp' : ['model_parameters'],
        #     'paper_eff': ['paper_id', 'model_parameters'],
        #     'paper_eff_sd': ['model_parameters'],
        #     'model_param_chol': ['model_parameters','model_parameters'],
        #     'model_param_chol_corr': ['model_parameters','model_parameters'],
        #     'model_param_chol_stds': ['model_parameters'],
        #     'model_param_corr': ['model_parameters','model_parameters'],
        #     'model_param_sd': ['model_parameters'],
        #     'error_sd': ['group_names'],
        # } 
        
        yield model
