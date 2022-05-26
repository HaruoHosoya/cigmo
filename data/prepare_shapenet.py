#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:27:08 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision
from torchvision import transforms

import scipy.misc
import os
import argparse
from PIL import Image

from tqdm import tqdm

default_path = os.environ['HOME'] + '/resultsets/mvae/20200715/shapenet/multi_view2/test'

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default=default_path, help="load path")
parser.add_argument('--save_path', type=str, default='./', help="save path")
options = parser.parse_args()

#%%  

tran = transforms.Compose([
      transforms.Grayscale(),
      transforms.Lambda(lambda img: transforms.functional.crop(img, 5, 4, 118, 118)),
      transforms.Resize(64),
      ])
     
is_valid_file = lambda path: os.path.basename(path)[0] != '.'

print('loading images from:', options.load_path)

dataset = torchvision.datasets.ImageFolder(options.load_path, transform=tran, target_transform=None, is_valid_file=is_valid_file)

#%%

# bitimg = torch.ones_like(train_dataset[0][0], dtype=torch.bool)
# for i in range(len(train_dataset)):
#     bitimg &= (train_dataset[i][0] * 255) <= 1
    
#%%

print('processing images')

trainset = []
for i in tqdm(range(len(dataset)), position=0, leave=True):
    img = dataset[i][0]
    cla = dataset[i][1]
    path = dataset.imgs[i][0]
    bn = os.path.splitext(os.path.basename(path))[0]
    strs = bn.split(sep='_')
    model_no = strs[0]
    elev = strs[1]
    azim = strs[2]
    item = (img, (dataset.classes[cla], model_no), (elev, azim))
    trainset.append(item)

#%%
    
print('saving file to:', options.save_path)
    
torch.save(trainset, options.save_path)

#%%


