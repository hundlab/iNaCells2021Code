#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:01:30 2020

@author: grat05
"""

import json
from glob import iglob
import pandas as pd
import numpy as np


data_root = './Data'#'C:/Users/grat05/OneDrive for Business/Data'

def load_all_data(data_root = data_root):
    files = [glb.replace('\\','/') for glb in iglob(data_root+'/data/*/*')]
    data = {}
    for file in files:
        try:
            filename = file.split('/')[-1]
            pmid,ext = filename.split('.')
            if ext == 'json':
                js = json.load(open(file,'r'))
                data[pmid] = js
            else:

                df = pd.read_csv(file)
                data[pmid] = df
        except:
            print("Could not read file:")
            print(file)
    return data

try: all_data
except NameError: all_data = load_all_data()

def converter(data_root = data_root):
    folder2pmid = {}
    for folder in (glb.replace('\\','/') for glb in iglob(data_root+'/data/*')):
        files = [glb.replace('\\','/') for glb in iglob(folder+'/*')]
        pmids = set((file.split('/')[-1].split('_')[0] for file in files))
        folder_name = folder.split('/')[-1]
        folder2pmid[folder_name] = pmids

    pmid2folder = []
    for folder,pmids in folder2pmid.items():
        for pmid in pmids:
            pmid2folder.append((pmid,folder))
    pmid2folder = sorted(pmid2folder)
    return folder2pmid, pmid2folder

def load_data_parameters(filename, sheet_name, data_root = data_root, data = all_data, default_duration=100):
    data_parameters = pd.read_excel(data_root+'/'+filename,sheet_name=sheet_name,index_col=[0,1])
    #convert to kelvin
    data_parameters['temp ( C )'] += 273.15
    data_parameters.rename(columns={'temp ( C )':'temp ( K )'},inplace=True)
    #set unset duration
    data_parameters.loc[data_parameters['duration (ms)'].isnull(),'duration (ms)'] = 100

    #extract sub_data
    sub_data = {}
    for idx in data_parameters.index:
        datasets = data[idx[0]]['datasetColl']
        for dataset in datasets:
            if dataset['name'] == idx[1]:
                fig_data = [entry['value'] for entry in dataset['data']]
                temp_data = np.array(fig_data)
                order = np.argsort(temp_data[:,0])
                sub_data[idx] = temp_data[order,:]
                break

    return data_parameters, sub_data
