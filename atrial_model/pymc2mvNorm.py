#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:59:54 2020

@author: grat05
"""

import numpy as np
import pymc
from scipy import stats

def logp_mvnorm(value, mean, cov):
    mean = np.atleast_2d(mean)
    value = np.atleast_2d(value)
    logp = 0
    for val, mu in zip(value, mean):
        logp += stats.multivariate_normal.logpdf(val, mean=mu, cov=cov)
    return logp

def random_mvnorm(mean, cov, size=None):
    mean_1d = len(mean.shape) == 1
    mean = np.atleast_2d(mean)
    rng = np.random.default_rng()
    def draw_func(mean):
        return rng.multivariate_normal(mean, cov, size=size)
    draws = np.apply_along_axis(draw_func, 1, mean)
    return np.squeeze(draws) if mean_1d else draws

MvNorm = pymc.new_dist_class(float,
                             'multivariate_normal',
                             ['mean', 'cov'],
                             {'mean':np.zeros((1,1)), 'cov':np.ones((1,1))},
                             'A Multivariate Normal Distribution',
                             logp_mvnorm,
                             random_mvnorm,
                             True,
                             None)
    

    

    