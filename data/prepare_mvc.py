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
from distutils.util import strtobool

import scipy.misc
import os
import argparse
from PIL import Image

import urllib
import http
import time
import json

from PIL import Image

from tqdm import tqdm

load_path = os.environ['HOME'] + '/datasets/MVC'

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default=load_path, help="load path")
parser.add_argument('--save_path', type=str, default='./', help="save path")
parser.add_argument('--download', type=strtobool, default=False, help="download from web")
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--train_ratio', type=float, default=0.8, help="ratio of training data")
parser.add_argument('--image_size', nargs='+', type=int, default=[64,64], help="image size")

options = parser.parse_args()

#%%

if options.download:
    with open(os.path.join(options.load_path, 'image_links.json'), 'r') as fin:
        links = json.load(fin)

    with open(os.path.join(options.load_path, 'attribute_labels.json'), 'rb') as fin:
        attrs = json.load(fin)
        
    os.makedirs(os.path.join(options.load_path, 'image_data'), exist_ok=True)

    for i in range(len(links)):
        link = links[i]['image_url_4x']
        fname = attrs[i]['filename']
        filepath = os.path.join(options.load_path, 'image_data', fname + '.jpg')
        
        if os.path.exists(filepath):
            continue
        
        print('downloading', link, 'into', fname)
        
        headers = { 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0' } 
        request = urllib.request.Request(link, headers=headers) 
        while True:
            try:
                data = urllib.request.urlopen(request).read()
                break
            except urllib.error.HTTPError as e:
                print('Error:', str(e))
                print('Retry after 5 secs...')
                time.sleep(5)
            except http.client.IncompleteRead as e:
                print('Error:', str(e))
                print('Retry after 5 secs...')
                time.sleep(5)
            
        with open(filepath, mode='wb') as f:
            f.write(data)    

#%%

os.makedirs(options.save_path, exist_ok=True)    

#%%

with open(os.path.join(options.load_path, 'attribute_labels.json'), 'r') as fin:
    attrs = json.load(fin)

#%%

all_itemn = np.sort(np.unique([attr['itemN'] for attr in attrs]))
all_itemn = np.random.RandomState(seed=options.seed).permutation(all_itemn)
train_itemn = all_itemn[:int(len(all_itemn) * options.train_ratio)]
test_itemn = all_itemn[int(len(all_itemn) * options.train_ratio):]

print('# of items: train', len(train_itemn), '; test', len(test_itemn))

train_attrs = [attr for attr in attrs if (attr['itemN'] in train_itemn)]
test_attrs = [attr for attr in attrs if (attr['itemN'] in test_itemn)]

print('data size: train', len(train_attrs), '; test', len(test_attrs))

#%%

print('loading images from:', options.load_path, 'resized to: ', options.image_size)
print('for training data...')

trainset = []
for attr in tqdm(train_attrs, position=0, leave=True):
    fname = attr['filename'] + '.jpg'
    img = Image.open(os.path.join(load_path, 'image_data', fname))
    img = img.resize(options.image_size)
    target = attr['itemN']
    item = (img, target, attr)
    trainset.append(item)

#%%

fpath = os.path.join(options.save_path, 'trainset.pt')
print('saving file to:', fpath)    
torch.save(trainset, fpath)

#%%

print('for test data...')
    
testset = []
for attr in tqdm(test_attrs, position=0, leave=True):
    fname = attr['filename'] + '.jpg'
    img = Image.open(os.path.join(load_path, 'image_data', fname))
    img = img.resize(options.image_size)
    target = attr['itemN']
    item = (img, target, attr)
    testset.append(item)

#%%

fpath = os.path.join(options.save_path, 'testset.pt')
print('saving file to:', fpath)    
torch.save(testset, fpath)

#%%

# # removed class

# 0016Bandeau
# 0027BoxerBriefs
# 0028Boxers
# 0029BoyShorts
# 0032Bras
# 0033Brief
# 0077Crossback
# 0118Lingerie
# 0147Panties
# 0110Hosiery
# 0137OnePieceSwimsuits
# 0215SwimSets
# 0216SwimsuitBottoms
# 0217SwimsuitTops
# # 

# # subclasses

# 0014BalconetteBras
# 0015BandeauBras
# 0031BraletteBras
# 0060CompressionBras
# 0064ConvertibleBras
# 0080DemiBras
# 0086EnhancedBras
# 0096FullCoverageBras
# 0129MinimizerBras
# 0131MoldedCupBras
# 0132NursingBras
# 0142PaddedBras
# 0157PlungeBras
# 0166PushUpBras
# 0188ShelfBras
# 0203SoftCupBras
# 0205SportsBras
# 0208StraplessBras
# 0240UnderwireBras


