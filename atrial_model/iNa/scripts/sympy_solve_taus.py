#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:58:53 2020

@author: grat05
"""

from sympy import *

tau_m, tau_f, tau_s = symbols("tau_m tau_f tau_s", positive=True, real=True)
fs_prop = symbols("fs_prop", nonnegative=True, real=True)
fs_prop_b = ConditionSet(fs_prop, (0 <= fs_prop) & (fs_prop <= 1), S.Reals)
n = symbols("n", integer=True)
i = symbols('i', cls=Idx)
t = Indexed('t', i)
rc = Indexed('rc', i)


A1 = fs_prop
A2 = 1 - fs_prop
curr = (1-exp(-t/tau_m))**3 * (-A1*exp(-t/tau_f) -A2*exp(-t/tau_s))
loss = (Sum(rc - curr, (i ,1, n)))**2

solveset(loss.diff(tau_f), tau_f, domain=S.Reals)