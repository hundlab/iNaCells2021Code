#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:27:45 2020

@author: grat05
"""

import scripts

try: all_data
except NameError: all_data = scripts.load_all_data()