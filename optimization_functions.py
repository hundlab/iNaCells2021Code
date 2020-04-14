#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:14:52 2020

@author: dgratz
"""

import numpy as np
from scipy import optimize


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


def print_fun(x, f, accepted):
    print("Minimum found:", bool(accepted), ", Cost:", f)
    print("At:", x)

def save_results(x, f, accepted, results=None):
    print("Minimum found:", bool(accepted), ", Cost:", f)
    print("At:", x)
    if not results is None:
        results.append((x,f))

#called after, doesn't work
def check_bounds(f_new, x_new, f_old, x_old, bounds=None, **kwargs):
    print("---")
    print(f_new, x_new, f_old, x_old)
    print("---")
    if bounds is None:
        return True
    else:
        aboveMin = bool(np.all(x_new > bounds[:,0]))
        belowMax = bool(np.all(x_new < bounds[:,1]))
        print("---")
        print(x_new, aboveMin and belowMax)
        print("---")
        return aboveMin and belowMax
    