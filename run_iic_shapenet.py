#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:00:33 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import os as os
import os.path as osp
import sys
import numpy as np
import numpy.random
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import skimage
import argparse
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms

import utils
from network64iic import *
from train_iic import *
import evalu 
import datasets 

from sklearn.cluster import KMeans
import sklearn.metrics

#%%

device = utils.get_device()
torch.backends.cudnn.benchmark = True

#%%

batch_size = 100
color = False

# res_root = os.path.join(os.environ['HOME'], 'resultsets/mvae/20200824/shapenet')
res_root = os.path.join(os.environ['HOME'], 'resultsets/mvae/20200927/shapenet')
ds_path = os.path.join(os.environ['HOME'], 'resultsets/mvae/20200715/shapenet/multi_view2')
save_result = False

#%%

num_class = 2
#num_class = 3
#num_class = 5
#num_class = 10

#%%

start_instance = 0
num_instance = 10
num_epochs = 20
num_cluster = num_class 
num_overcluster = None
group_size = 3
mode = 'train'

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default=ds_path, help="dataset path")
parser.add_argument('--model_path', type=str, default=res_root, help="model root path")
parser.add_argument('--save_result', type=strtobool, default=save_result, help="save results")
parser.add_argument('--num_instance', type=int, default=num_instance, help="number of instances")
parser.add_argument('--start_instance', type=int, default=start_instance, help="starting instance index")
parser.add_argument('--num_class', type=int, default=num_class, help="number of classes")
parser.add_argument('--num_cluster', type=int, default=num_cluster, help="number of clusters")
parser.add_argument('--num_overcluster', type=int, default=0, help="number of over-clusters or None")
parser.add_argument('--num_epochs', type=int, default=num_epochs, help="number of epochs")
parser.add_argument('--group_size', type=int, default=group_size, help="group size")
parser.add_argument('--mode', type=str, default=mode, help="'train' or 'test'")
parser.add_argument('--gpu', type=str, default=None, help="gpu number")

options = parser.parse_args()

ds_path = options.dataset_path
res_root = options.model_path
save_result = bool(options.save_result)
num_instance = options.num_instance
start_instance = options.start_instance
num_class = options.num_class
num_cluster = options.num_cluster
num_overcluster = options.num_overcluster if options.num_overcluster!=0 else None
num_epochs = options.num_epochs
group_size = options.group_size
mode = options.mode
if options.gpu == None:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + options.gpu)

#%%

subset, class_dict = datasets.shapenet_standard_subset(num_class)

#%%

dirname = os.path.join(res_root, 'iic_s{}_c{}_k{}_o{}'.format(num_class, num_cluster, group_size, num_overcluster))

#%%

print('loading dataset from', ds_path)

start = time.time()

if mode == 'train':
    train_dataset = datasets.load_shapenet_dataset(root=ds_path, subset=subset, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataset_grouped = datasets.DatasetGrouped(train_dataset, group_size)
    train_dataloader_grouped = torch.utils.data.DataLoader(train_dataset_grouped, batch_size=batch_size, shuffle=True)

if mode == 'test':
    test_dataset = datasets.load_shapenet_dataset(root=ds_path, subset=subset, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('loading time: {0} sec'.format(time.time() - start))

#%%

if mode == 'train':
    print('learning and saving models to', dirname)
    os.makedirs(dirname, exist_ok=True)
    for i in range(start_instance, start_instance+num_instance):
        net = IICNet(num_cluster, num_overcluster, color=color, breadth_ratio=1)    
        optimizer = optim.Adam(params=utils.get_params(net), lr=0.001)
        train_iic(net, train_dataloader_grouped, optimizer, num_epochs=num_epochs, device=device)
        net_path = os.path.join(dirname, 'net_{}.pt'.format(i))
        save_net(net, net_path)

#%%
        
if mode == 'test':
    print('testing models from', dirname)
    nets = []
    for i in range(start_instance, start_instance+num_instance):
        net_path = os.path.join(dirname, 'net_{}.pt'.format(i))
        net = load_net(net_path)
        nets.append(net)
    ress = []
    for i in range(len(nets)):
        net = nets[i]
        res = evalu.calc_clustering(net, test_dataloader, class_dict, device=device)        
        res.acc, res.est_classes, res.cluster_to_class = evalu.clustering_accuracy(res.clusters.numpy(), num_class, res.cposts.numpy(), res.classes.numpy())
        res.ba_acc, res.ba_est_labels = evalu.best_assignment_accuracy(res.clusters.numpy(), num_class, res.classes.numpy(), np.arange(num_class))
        res.ari = evalu.ari(res.clusters.numpy(), res.classes.numpy())        
        ress.append(res)
        
if mode == 'show':
    ress = []
    for i in range(num_instance):
        res_path = osp.join(dirname, 'res_{}.pt'.format(start_instance+i))
        ress.append(torch.load(res_path))
        
if mode == 'test' or mode == 'show':
    for i in range(len(ress)):
        res = ress[i]
        print('instance #{}: class acc={:.4f}  BA acc={:.4f}'.format(i, res.acc, res.ba_acc))
    accs = [res.acc for res in ress]
    print('classification accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(accs), np.std(accs), np.max(accs)))
    ba_accs = [res.ba_acc for res in ress]
    print('BA accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(ba_accs), np.std(ba_accs), np.max(ba_accs)))
    aris = [res.ari for res in ress]
    print('ARI = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(aris), np.std(aris), np.max(aris)))
   
#%%
    
if save_result:
    print('saving results to', dirname)
    for i in range(len(ress)):
        res_path = os.path.join(dirname, 'res_{}.pt'.format(start_instance+i))
        torch.save(ress[i], res_path)
     
#%%
        
if sys.argv[0] != '':
    quit()

#############################################
#%%

net_iic = IICNet(num_cluster, num_overcluster, color=color, breadth_ratio=1)    
optimizer = optim.Adam(params=utils.get_params(net_iic), lr=0.001)

#%%

num_epochs=10
train_iic(net_iic, train_dataloader_grouped, optimizer, num_epochs=num_epochs, device=device)

#%%

i=5
#num_overcluster=None
num_overcluster=num_cluster*5
dirname = os.path.join(res_root, 'iic_s{}_c{}_k{}_o{}'.format(num_class, num_cluster, group_size, num_overcluster))
net_path = osp.join(dirname, 'net_{}.pt'.format(i))
print('loading from:', net_path)
net_iic = load_net(net_path)

#%%

res = evalu.calc_clustering(net_iic, test_dataloader, class_dict, device=device)

res.acc, res.est_classes, res.cluster_to_class = evalu.clustering_accuracy(res.clusters.numpy(), num_class, res.cposts.numpy(), res.classes.numpy())
res.ba_acc, res.ba_est_labels = evalu.best_assignment_accuracy(res.clusters.numpy(), num_class, res.classes.numpy(), np.arange(num_class))

#%%
    
#res = evalu.test_model(net, test_dataloader, device=device)

#print('classification accuracy={:.4f} (chance={:.4f})'.format(res.acc, res.chance_acc))

for k in range(num_cluster):
    print('  estimated cluster #{} : {}'.format(k, torch.sum(res.clusters == k)))
        

if num_cluster == num_class:    
    print('best assignment accuracy={:.4f}'.format(res.ba_acc))
    for k in range(num_class):
        print('  estimated class #{} : {}'.format(k, np.sum(res.ba_est_labels == k)))

for k in range(num_cluster):
    counts = []
    for l in range(num_class):
        counts.append(np.sum((res.clusters.numpy() == k) & (res.classes.numpy() == l)))
    print('  cluster #{}: '.format(k), ('{:6d} '*num_class).format(*counts))

print('ARI={:.4f}'.format(sklearn.metrics.adjusted_rand_score(res.classes,res.clusters)))

#%%
            
idx17 = (res.classes==17).nonzero().squeeze()
inputs17 = res.inputs[idx17,:,:,:]
idx18 = (res.classes==18).nonzero().squeeze()
inputs18 = res.inputs[idx18,:,:,:]

imgs17 = inputs17[[0,17,30,40],:,:,:]
imgs18 = inputs18[[0,17,30,40],:,:,:]
imgs = torch.cat([imgs17, imgs18], dim=0)
imgs = torch.cat([imgs[0::2], imgs[1::2]], dim=0)
img=torchvision.utils.make_grid(imgs, nrow=4, padding=2, pad_value=1)

#plt.imshow(img.numpy().transpose((1,2,0)))

os.makedirs(os.path.join(res_root, 'examples'))
torchvision.utils.save_image(img, os.path.join(res_root, 'examples/example1.png'))




