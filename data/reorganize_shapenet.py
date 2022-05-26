#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:37:51 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""


import numpy as np
import matplotlib.pyplot as plt

import scipy.misc
import os
import argparse
import csv
import shutil
from os.path import join

default_root = join(os.environ['HOME'], 'datasets/ShapeNetCore55/')

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default=join(default_root, 'test_allinone'), help="load path")
parser.add_argument('--csv_path', type=str, default=join(default_root, 'Evaluator/all.csv'), help="load path")

parser.add_argument('--save_path', type=str, default=join(default_root, 'test'), help="save path")
options = parser.parse_args()

#%%

cla = dict()

with open(options.csv_path) as f:
    reader = csv.reader(f)
    for mid, synid, _, _, _ in reader:
        cla[mid] = synid

#%%
        
fnames = os.listdir(options.load_path)
fnames = [f for f in fnames if os.path.isfile(join(options.load_path, f)) and f[0] != '.']

os.makedirs(options.save_path, exist_ok=True)

for i in range(len(fnames)):    
    bn = os.path.splitext(fnames[i])[0]
    mid= bn.split(sep='_')[1]
    if not mid in cla: continue
    synid = cla[mid]
    os.makedirs(join(options.save_path, synid), exist_ok=True)
 #   print('copying {} to {}'.format(join(options.load_path, fnames[i]), join(join(options.save_path, synid), fnames[i])))
    shutil.copy(join(options.load_path, fnames[i]), join(join(options.save_path, synid), fnames[i]))
    
    
    
    