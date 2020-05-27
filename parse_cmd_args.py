#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:32:38 2020

@author: grat05
"""

import argparse
import sys

parser = argparse.ArgumentParser(prog='Program to Fit Ion Channels')
parser.add_argument('program_location')
parser.add_argument('--data_root', nargs='?', default='C:/Users/grat05/OneDrive for Business/Data')
parser.add_argument('--out_dir', nargs='?', default='./')
parser.add_argument('--max_time', nargs='?', type=float, help='Max time allowed for MCMC sampler (hours)')
parser.add_argument('--previous_run', nargs='?')
parser.add_argument('--model_name', nargs='?', default='Koval')


args = parser.parse_args(sys.argv)