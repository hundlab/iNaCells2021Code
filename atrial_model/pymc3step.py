#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:21:19 2020

@author: grat05
"""


#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from copy import deepcopy
import numpy as np
import numpy.random as nr
from numpy.random import uniform
import theano
import scipy.linalg
from functools import partial

from pymc3.step_methods.arraystep import (
    BlockedStep,
    Competence,
)
import pymc3 as pm
from pymc3.theanof import floatX

__all__ = [
    "MultiMetropolis",
]

# Available proposal distributions for Metropolis


class Proposal:
    def __init__(self, s):
        self.s = s


class NormalProposal(Proposal):
    def __call__(self):
        return nr.normal(scale=self.s)


class UniformProposal(Proposal):
    def __call__(self):
        return nr.uniform(low=-self.s, high=self.s, size=len(self.s))


class CauchyProposal(Proposal):
    def __call__(self):
        return nr.standard_cauchy(size=np.size(self.s)) * self.s


class LaplaceProposal(Proposal):
    def __call__(self):
        size = np.size(self.s)
        return (nr.standard_exponential(size=size) - nr.standard_exponential(size=size)) * self.s


class PoissonProposal(Proposal):
    def __call__(self):
        return nr.poisson(lam=self.s, size=np.size(self.s)) - self.s


class MultivariateNormalProposal(Proposal):
    def __init__(self, s, num_draws=1):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.num_draws = num_draws
        self.n = n
        self.chol = scipy.linalg.cholesky(s, lower=True)
        super().__init__(s)
        
    def set_s(self, s):
        self.s = s
        self.chol = scipy.linalg.cholesky(s, lower=True)

    def __call__(self):
        b = np.random.randn(self.n, self.num_draws)
        draws = np.dot(self.chol, b).T
        return np.squeeze(draws)
 
class RowwiseMultivariateNormalProposal(Proposal):
    def __init__(self, s):
        n, m = s.shape[1:]
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chols = np.array([scipy.linalg.cholesky(s_sub, lower=True)
                               for s_sub in s])
        super().__init__(s)
        
    def set_s(self, s):
        self.s = s
        self.chols = np.array([scipy.linalg.cholesky(s_sub, lower=True)
                               for s_sub in s])

    def __call__(self):
        draw = []
        for chol in self.chols:
            b = np.random.randn(self.n)
            draw.append(np.dot(chol, b).T)
        return np.array(draw)

class SingleMetropolis():
    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {
            "accept": np.float64,
            "accepted": np.bool,
            "tune": np.bool,
            "scaling": np.float64,
        }
    ]
    
    name = "single_metropolis"
    
    def __init__(
        self,
        var,
        S=None,
        proposal_dist=None,
        scaling=None,
        tune=True,
        tune_interval=100,
        model=None,
        logp_func=None,
        mode=None,
        other_keys=None,
        parent=None,
        alt_step_interval=10,
        alt_step_prob=0.1,
        **kwargs
    ):
        self.var = var
        
        if logp_func is None:
            logp_func = model.logp
        self.logp_func = logp_func

        if S is None:
            S = 0.1*np.identity(var.dsize)#np.ones(var.dshape)

        if proposal_dist is not None:
            if isinstance(proposal_dist, type):
                self.proposal_dist = proposal_dist(S)
            else:
                self.proposal_dist = deepcopy(proposal_dist)
        else:
            self.proposal_dist = NormalProposal(S)

        if scaling is None:
            self.scaling = 1
        else:
            self.scaling = scaling

        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = []

        self.update_scale = True
        self.update_S = True
        self.update_sd_scale = False
        self.trace = []
        self.trace_deltas = []
        self.trace_deltas_full = []
        self.covs = []

        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            scaling=self.scaling, steps_until_tune=tune_interval, accepted=self.accepted
        )

        self.mode = mode
        self.blocked = False
        
        self.parent = parent
        self.other_keys = other_keys
        self.alt_step_interval = alt_step_interval
        self.alt_step_prob = max(0, min(1, alt_step_prob))
        self.alt_step = alt_step_interval
        
        self.latest = None

#        shared = pm.make_shared_replacements([var], model)
#        self.delta_logp = delta_logp(model.logpt, [var], shared)
#        super().__init__(vars, blocked=False)

    def reset_tuning(self):
        """Resets the tuned sampler parameters to their initial values."""
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        return

    def tune_values(self):
        if not self.steps_until_tune and self.tune:
            self.accepted = np.array(self.accepted, dtype=bool)
#            import pdb
#            pdb.set_trace()
            # Tune scaling parameter
            if self.update_scale:
                self.scaling = tune(self.scaling, np.mean(self.accepted))
            if self.update_S:
                self.trace = np.array(self.trace)
                self.proposal_dist.set_s(tune_s_single(self.proposal_dist.s, self.trace))
                self.trace = []
            if self.update_sd_scale:
                self.trace_deltas = np.array(self.trace_deltas)
                S = tune_scale_sd(self.proposal_dist.s, self.trace_deltas, self.accepted)
                self.proposal_dist.set_s(S)
                self.trace_deltas = []
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = []

    def propose(self, q0, rest_point):
        
        q0 = np.copy(q0)
        
        if self.tune:
            self.tune_values()

        if not self.other_keys is None and not self.alt_step:
            [other_key] = np.random.choice(self.other_keys, 1, replace=False)
            selected_params = np.random.uniform(0,1,self.var.dsize)
            selected_params = selected_params > (1 - self.alt_step_prob)
            other = self.parent.getSubStep(other_key)
            other_latest = other.getLatest()
            if not other_latest is None:
                q = np.where(selected_params, other_latest, q0)
            else:
                q = np.copy(q0)
            
            self.trace_deltas.append(q-q0)
            self.trace_deltas_full.append(q-q0)
            self.alt_step = self.alt_step_interval
        else:
            delta = self.proposal_dist() * self.scaling
            q = floatX(q0 + delta)
            
            self.trace_deltas.append(delta)
            self.trace_deltas_full.append(delta)
        
        if not self.latest is None:
            self.latest[:] = q[:]
        else:
            self.latest = np.copy(q)
        self.alt_step -= 1
        
        return q
    
    def accept(self, q0, q, rest_point):

        q0 = np.copy(q0)
        q = np.copy(q)

        temp_point = rest_point.copy()
        temp_point.update({self.var.name:q0})
        logp0 = self.logp_func(temp_point)
        temp_point = rest_point.copy()
        temp_point.update({self.var.name:q})
        logp1 = self.logp_func(temp_point)

        accept = logp1 - logp0
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += [accepted]
        
#        import pdb
#        pdb.set_trace()

        self.steps_until_tune -= 1

        if self.tune and self.update_S:
            self.trace.append(q_new)

        stats = {
            "tune": self.tune,
            "scaling": self.scaling,
            "accept": np.exp(accept),
            "accepted": accepted,
        }
        
        self.covs.append(self.proposal_dist.s)
        
        return q_new, [stats]
    
    def getLatest(self):
        return self.latest
    

class MultiMetropolis(BlockedStep):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to normal.
    scaling: scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions
    """

    name = "multi_metropolis"

    default_blocked = True
    generates_stats = True

    def __init__(
        self,
        vars=None,
        S=None,
        proposal_dist=None,
        scaling=None,
        tune=True,
        tune_interval=100,
        model=None,
        mode=None,
        key_groups=None,
        **kwargs
    ):

        model = pm.modelcontext(model)
        self.model = model
        self.logp_func = model.logp

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)
        self.vars = vars

        steps = {}
        self.stats_dtypes = []
        for i,var in enumerate(vars):
            S_sub = None
            scaling_sub = None
            other_keys = None
            
            if not S is None:
                S_sub = S[var.name]
            if not scaling is None:
                scaling_sub = scaling[var.name]
            if not key_groups is None:
                other_keys = key_groups[var.name]
                # for group in key_groups:
                #     if var.name in group:
                #         other_keys = [other 
                #                       for other in group 
                #                       if other != var.name]
            
            steps[var.name] = SingleMetropolis(
                var,
                S_sub,
                proposal_dist,
                scaling_sub,
                tune,
                tune_interval,
                model=model,
                logp_func=self.logp_func,
                mode=mode,
                other_keys=other_keys,
                parent=self,
                **kwargs)
            
            self.stats_dtypes += steps[var.name].stats_dtypes
        self.steps = steps

#        shared = pm.make_shared_replacements(vars, model)
#        self.delta_logp = delta_logp(model.logpt, vars, shared)
#        super().__init__(vars, blocked=True)

    def reset_tuning(self):
        """Resets the tuned sampler parameters to their initial values."""
        for name, step in self.steps.items():
            step.reset_tuning()

    def step(self, point):
        proposals = {}
        for name, step in self.steps.items():
            step.parent = self
            rest_point = point.copy()
            del rest_point[name]
            proposals[name] = step.propose(point[name], rest_point)
                     
        
        logp_old = self.logp_func(point)
        temp_point = point.copy()
        temp_point.update(proposals)
        logp_new = self.logp_func(temp_point)

        new_point = deepcopy(point)
        stats = []
        for name, step in self.steps.items():
            rest_point = point.copy()
            q0 = rest_point.pop(name)
            q = proposals[name]
            q_new, stat = step.accept(q0, q, rest_point)
            new_point[name] = q_new
            stats += stat

        print(sum(map(lambda x: x["accepted"], stats)), 
              np.round(logp_old, 2),
              np.round(logp_new-logp_old,2),
              np.round(sum(map(lambda x: np.log(x["accept"]) 
                               if x['accepted'] else 0,
                               stats))))

        return new_point, stats

    @staticmethod
    def competence(var, has_grad):
        return Competence.COMPATIBLE
    
    def getTraceDeltas(self):
        all_deltas = {}
        for name, step in self.steps.items():
            all_deltas[name] = np.array(step.trace_deltas_full)
        return all_deltas
    
    def getTraceCovs(self):
        all_covs = {}
        for name, step in self.steps.items():
            all_covs[name] = np.array(step.covs)
        return all_covs

    def getSubStep(self, key):
        return self.steps[key]

def tune(scale, acc_rate, target=0.25, discount=1):
    log_new = np.log(scale) + discount*(acc_rate - target)
    return np.exp(log_new)

# def tune(scale, acc_rate):
#     """
#     Tunes the scaling parameter for the proposal distribution
#     according to the acceptance rate over the last tune_interval:

#     Rate    Variance adaptation
#     ----    -------------------
#     <0.001        x 0.1
#     <0.05         x 0.5
#     <0.2          x 0.9
#     >0.5          x 1.1
#     >0.75         x 2
#     >0.95         x 10

#     """
#     if acc_rate < 0.001:
#         # reduce by 90 percent
#         return scale * 0.1
#     elif acc_rate < 0.05:
#         # reduce by 50 percent
#         return scale * 0.5
#     elif acc_rate < 0.2:
#         # reduce by ten percent
#         return scale * 0.9
#     elif acc_rate > 0.95:
#         # increase by factor of ten
#         return scale * 10.0
#     elif acc_rate > 0.75:
#         # increase by double
#         return scale * 2.0
#     elif acc_rate > 0.5:
#         # increase by ten percent
#         return scale * 1.1

#     return scale

# def tune(scale, acc_rate):
#     """
#     Tunes the scaling parameter for the proposal distribution
#     according to the acceptance rate over the last tune_interval:

#     Rate    Variance adaptation
#     ----    -------------------
#     <0.001        x 0.1
#     <0.05         x 0.5
#     <0.2          x 0.9
#     >0.5          x 1.1
#     >0.75         x 2
#     >0.95         x 10

#     """
#     scale_scale = np.array(list(map(get_scale, acc_rate)))
#     scale *= scale_scale
    
#     return scale

def get_scale(acc_rate):
    if acc_rate < 0.001:
        # reduce by 90 percent
        return 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        return 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        return 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        return 10.0
    elif acc_rate > 0.75:
        # increase by double
        return 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        return 1.1
    else:
        return 1.
    
def tune_s_single(S, trace, factor=0.6):
    trace_cov = np.cov(trace, rowvar=False)
    new_S = S*factor + trace_cov*(1-factor)
    
    # trace_cor = np.corrcoef(trace,rowvar=False)
    # trace_cor[np.isnan(trace_cor)] = 0
    # trace_sd = np.std(trace, axis=0)

    # sd_0 = np.sqrt(np.diag(S))
    # cor_0 = S / np.outer(sd_0, sd_0)

    # new_cor = cor_0*factor + trace_cor*(1-factor)
    # new_sd = sd_0*factor + trace_sd*(1-factor)
    # new_S = new_cor * np.outer(new_sd, new_sd)
    return new_S
    
def tune_s(S, trace, factor=0.6):
    total_cor = np.zeros(S.shape)
    total_sd = np.zeros(S.shape[0])
    for i in range(trace.shape[1]):
        cor = np.corrcoef(trace[:,i,:],rowvar=False)
        sd = np.std(trace[:,i,:], axis=0)
        cor[np.isnan(cor)] = 0
        total_cor += cor
        total_sd += sd
    total_cor /= trace.shape[0]
    total_sd /= trace.shape[0]
    sd_0 = np.sqrt(np.diag(S))
    cor_0 = S / np.outer(sd_0, sd_0)
    new_cor = cor_0*factor + total_cor*(1-factor)
    new_sd = sd_0*factor + total_sd*(1-factor)
    new_S = new_cor * np.outer(new_sd, new_sd)
    return new_S

def tune_s_rowwise(S, trace, factor=0.6):
    new_S = np.zeros_like(S)
    for i in range(S.shape[0]):
        cor = np.corrcoef(trace[:,i,:],rowvar=False)
        sd = np.std(trace[:,i,:], axis=0)
        cor[np.isnan(cor)] = 0
        
        sd_0 = np.sqrt(np.diag(S[i]))
        cor_0 = S[i] / np.outer(sd_0, sd_0)
        
        new_cor = cor_0*factor + cor*(1-factor)
        new_sd = sd_0*factor + sd*(1-factor)
        new_S[i] = new_cor * np.outer(new_sd, new_sd)
    return new_S

def tune_scale_sd(S, trace_deltas, accepted):
    ranks = np.argsort(np.argsort(trace_deltas), axis=0)+1
    acc = ranks[accepted,:]
    
    n = ranks.shape[0]
    m = acc.shape[0]
    
    left = np.sum(acc < n/3, axis=0)
    right = np.sum(acc > n*2/3, axis=0)
    mid = m - (left+right)
    
    mid_props = mid / m
    mid_props_diff = mid_props - np.mean(mid_props)
    
    print(np.round(np.mean(np.abs(mid_props_diff)), 3), end=' ')
    
    def adjust(val):
        scale = 1
        if val > 0.15:
            scale = 0.85
        elif val > 0.05:
            scale = 0.9
        elif val < -0.15:
            scale = 1.15
        elif val < -0.05:
            scale = 1.1
        return scale
    
    scales = np.array(list(map(adjust, mid_props_diff)))
    
    sd_0 = np.sqrt(np.diag(S))
    cor_0 = S / np.outer(sd_0, sd_0)
    
    new_cor = cor_0
    new_sd = sd_0*scales
    new_S = new_cor * np.outer(new_sd, new_sd)
    
    return new_S

def sample_except(limit, excluded):
    candidate = nr.choice(limit - 1)
    if candidate >= excluded:
        candidate += 1
    return candidate


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


def delta_logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type("inarray1")

    logp1 = pm.CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f

# def delta_logp(func, vars, model, q, q0):
# #    test_point = model.test_point
#     dlogp0 = func(q.ravel(), q0.ravel())
# #    var = vars[0]
#     #test_point[var.name] = q
# #    logp_old = model.logp(test_point)
#     dlogps = []
#     for i in range(len(q)):
#         q_test = q0.copy()
#         q_test[i] = q[i]
# #        test_point[var.name] = q_test
# #        logp_new = model.logp(test_point)
#         dlogps.append(func(q_test.ravel(), q0.ravel()))
# #        print(logp_new-logp_old, dlogps[-1])
#     return np.array(dlogps)

def metrop_select(mr, q, q0):
    """Perform rejection/acceptance step for Metropolis class samplers.
    Returns the new sample q if a uniform random number is less than the
    metropolis acceptance rate (`mr`), and the old sample otherwise, along
    with a boolean indicating whether the sample was accepted.
    Parameters
    ----------
    mr: float, Metropolis acceptance rate
    q: proposed sample
    q0: current sample
    Returns
    -------
    q or q0
    """
    accept = np.isfinite(mr) and (np.log(uniform()) < mr)
    return q if accept else q0, accept
    # Compare acceptance ratio to uniform random number
#    accept =  np.isfinite(mr) & (np.log(uniform(size=len(mr))) < mr)
#    return np.where(accept[...,None], q, q0), accept

