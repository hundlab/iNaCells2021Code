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
from collections.abc import Mapping

from .parse_cmd_args import args

class DataLoader(Mapping):
    def __init__(self, data_root):
        self.data_root = data_root
        files = [glb.replace('\\','/') for glb in iglob(data_root+'/data/*/*')]
        self.files = {}
        for file in files:
            filename = file.split('/')[-1]
            pmid,ext = filename.split('.')
            self.files[pmid] = dict(filepath = file,
                                    filename = filename,
                                    ext = ext)
            
    def __getitem__(self, key):
        item = self.files[key]
        if not 'data' in item:
            try:
                if item['ext'] == 'json':
                    with open(item['filepath'],'r') as file:
                        data = json.load(file)
                else:
                    data = pd.read_csv(file)
                item['data'] = data
            except Exception as e:
                print("Could not read file: ", item['filepath'])
                raise e
        return item['data']
    
    def __iter__(self):
        return iter(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def _keytransform(self, key):
        return key

# def load_all_data(data_root = args.data_root):
#     files = [glb.replace('\\','/') for glb in iglob(data_root+'/data/*/*')]
#     data = {}
#     for file in files:
#         try:
#             filename = file.split('/')[-1]
#             pmid,ext = filename.split('.')
#             if ext == 'json':
#                 js = json.load(open(file,'r'))
#                 data[pmid] = js
#             else:
#                 df = pd.read_csv(file)
#                 data[pmid] = df
#         except:
#             print("Could not read file:")
#             print(file)
#     return data

try: all_data
except NameError: all_data = DataLoader(args.data_root)

def converter(data_root = args.data_root):
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

def load_data_parameters(filename, sheet_name, data = all_data, default_duration=100):
    data_parameters = pd.read_excel(all_data.data_root+'/'+filename,sheet_name=sheet_name,index_col=[0,1])

    #convert to kelvin
    if 'temp ( C )' in data_parameters:
        data_parameters['temp ( C )'] += 273.15
        data_parameters.rename(columns={'temp ( C )':'temp ( K )'},inplace=True)
    
    #set unset duration
    if 'duration (ms)' in data_parameters:
        data_parameters.loc[data_parameters['duration (ms)'].isnull(),'duration (ms)'] = default_duration

    sub_data = extract_sub_data(data_parameters.index, data=data)

    return data_parameters, sub_data

def extract_figures(fig_names, data=all_data):
    sub_data = {}
    for idx in fig_names:
        datasets = data[idx]['datasetColl']
        sub_fig = {}
        for dataset in datasets:
            fig_data = [entry['value'] for entry in dataset['data']]
            temp_data = np.array(fig_data)
            order = np.argsort(temp_data[:,0])
            sub_fig[dataset['name']] = temp_data[order,:]
        sub_data[idx] = sub_fig
    return sub_data

def extract_sub_data(data_keys, data=all_data):
    #extract sub_data
    sub_data = {}
    for idx in data_keys:
        datasets = data[idx[0]]['datasetColl']
        for dataset in datasets:
            if dataset['name'] == idx[1]:
                fig_data = [entry['value'] for entry in dataset['data']]
                temp_data = np.array(fig_data)
                order = np.argsort(temp_data[:,0])
                sub_data[idx] = temp_data[order,:]
                break
    return sub_data

