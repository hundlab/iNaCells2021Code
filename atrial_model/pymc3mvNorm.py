#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:09:39 2020

@author: grat05
"""


import numpy as np
import pymc3 as pm
import theano.tensor as tt
from scipy import stats
from pymc3.distributions.dist_math import bound

class MyMvNorm(pm.Continuous):
    def __init__(self, mean, chol_cov, *args, **kwargs):
        
        self.mean = mean
        self.chol_cov = chol_cov
        self.solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
        
        super().__init__(*args, **kwargs)
    
    def logp(self, value):
        delta = value - self.mean
        
        chol_cov = self.chol_cov
        diag = tt.nlinalg.diag(chol_cov)
        # Check if the covariance matrix is positive definite.
        ok = tt.all(diag > 0)
        # If not, replace the diagonal. We return -inf later, but
        # need to prevent solve_lower from throwing an exception.
        chol_cov = tt.switch(ok, chol_cov, 1)

        delta_trans = self.solve_lower(chol_cov, delta.T).T
        quaddist = (delta_trans ** 2).sum(axis=-1)
        logdet = tt.sum(tt.log(diag))
        
        k = pm.floatX(value.shape[-1])
        norm = -0.5 * k * pm.floatX(np.log(2 * np.pi))
        return bound(norm - 0.5 * quaddist - logdet, ok)
    
    # def logp(self, value):
    #     mean = np.atleast_2d(self.mean)
    #     value = np.atleast_2d(value)
    #     logp = 0
    #     for val, mu in zip(value, mean):
    #         logp += stats.multivariate_normal.logpdf(val, mean=mu, cov=self.cov)
    #     return logp
    
    def random(self, point=None, size=None):
        mean_1d = len(self.mean.shape) == 1
        mean = np.atleast_2d(self.mean)
        cov = np.dot(self.chol_cov, self.chol_cov.T)
        rng = np.random.default_rng()
        def draw_func(mean):
            return rng.multivariate_normal(mean, cov, size=size)
        draws = np.apply_along_axis(draw_func, 1, mean)
        return np.squeeze(draws) if mean_1d else draws

