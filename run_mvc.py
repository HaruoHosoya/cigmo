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
from torch.utils.tensorboard import SummaryWriter

import utils
from network64 import MGVAENet, load_net, save_net
from train import train_model, train_model_mlvae
import evalu 
import datasets 

from sklearn.cluster import KMeans

#%%

device = utils.get_device()
torch.backends.cudnn.benchmark = True

#%%

batch_size = 100
color = True
single_view_encoder = True
a_posterior = 'average'
purge_large_data = False
save_result = False
mode = 'train'
metric = 'euclidean'

res_root = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/mvc')
ds_path = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/data/mvc64')

# res_root = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/mvc96')
# ds_path = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/data/mvc96')

#%%

start_instance = 0
num_instance = 10
num_epochs = 20

#%%

use_subset = True

#%%

num_cluster = 5 
content_dim = 100
view_dim = 3

#%%

group_size = 3

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default=ds_path, help="dataset path")
parser.add_argument('--model_path', type=str, default=res_root, help="model root path")
parser.add_argument('--save_result', type=strtobool, default=save_result, help="save results")
parser.add_argument('--num_instance', type=int, default=num_instance, help="number of instances")
parser.add_argument('--start_instance', type=int, default=start_instance, help="starting instance index")
parser.add_argument('--content_dim', type=int, default=content_dim, help="content dimension")
parser.add_argument('--view_dim', type=int, default=view_dim, help="view dimension")
parser.add_argument('--use_subset', type=strtobool, default=use_subset, help="use standard subset")
parser.add_argument('--num_cluster', type=int, default=num_cluster, help="number of clusters")
parser.add_argument('--num_epochs', type=int, default=num_epochs, help="number of epochs")
parser.add_argument('--group_size', type=int, default=group_size, help="group size")
parser.add_argument('--purge_large_data', type=strtobool, default=True, help="purge large data")
parser.add_argument('--single_view_encoder', type=strtobool, default=True, help="use a single view encoder")
parser.add_argument('--mlvae', type=strtobool, default=False, help="use MLVAE training")
parser.add_argument('--gpu', type=str, default=None, help="gpu number")
parser.add_argument('--mode', type=str, default=mode, help="'train' or 'test' or 'kmeans' or 'show'")
parser.add_argument('--metric', type=str, default=metric, help="'cosine' or 'euclidean'")
parser.add_argument('--a_posterior', type=str, default=a_posterior, help="'average' or 'product' or 'logit_average")
parser.add_argument('--kmeans_num_cluster', type=int, default=0, help="number of clusters for kmeans (default: equal to number of classes)")

options = parser.parse_args()

ds_path = options.dataset_path
res_root = options.model_path
save_result = bool(options.save_result)
num_instance = options.num_instance
start_instance = options.start_instance
content_dim = options.content_dim
view_dim = options.view_dim
use_subset = options.use_subset
num_cluster = options.num_cluster
num_epochs = options.num_epochs
group_size = options.group_size
single_view_encoder = bool(options.single_view_encoder)
mlvae = bool(options.mlvae)
purge_large_data = bool(options.purge_large_data)
mode = options.mode
metric = options.metric
a_posterior = options.a_posterior
kmeans_num_cluster = options.kmeans_num_cluster  

if options.gpu == None or options.gpu == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + options.gpu)

#%%

print('loading dataset from', ds_path)
if use_subset:
    print('  using the standard subset')

tran = transforms.Compose(
    ([ transforms.Grayscale() ] if not color else []) +
     [
      transforms.ToTensor(),
      ])

start = time.time()

if mode == 'train' or mode == 'kmeans':
    train_dataset = datasets.DatasetFromTensor(root=ds_path, split='train', transform=tran)
    if use_subset:
        train_dataset = datasets.mvc_standard_subset(train_dataset)
    train_dataset_grouped = datasets.DatasetGrouped(train_dataset, group_size)
    train_dataloader_grouped = torch.utils.data.DataLoader(train_dataset_grouped, batch_size=batch_size, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
elif mode == 'test':
    test_dataset = datasets.DatasetFromTensor(root=ds_path, split='test', transform=tran)
    if use_subset:
        test_dataset = datasets.mvc_standard_subset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('loading time: {0} sec'.format(time.time() - start))

#%%
    
if mlvae:
    dirname = osp.join(res_root, 'mlvae_c{}_k{}_m{}_l{}_v{:d}'.format(num_cluster, group_size, content_dim, view_dim, single_view_encoder))    
else:
    dirname = osp.join(res_root, 'models_c{}_k{}_m{}_l{}_v{:d}'.format(num_cluster, group_size, content_dim, view_dim, single_view_encoder))    

if a_posterior == 'product':
    dirname = dirname + '_ap'
elif a_posterior == 'logit_average':
    dirname = dirname + '_al'
    
#%%

if mode == 'train':
    print('learning and saving models to', dirname)
    os.makedirs(dirname, exist_ok=True)
    for i in range(start_instance, start_instance+num_instance):
        while True:
            net = MGVAENet(num_cluster, content_dim, view_dim, color=color, breadth_ratio=1, single_view_encoder=single_view_encoder)    
            optimizer = optim.Adam(params=utils.get_params(net), lr=0.001)
            if mlvae:
                loss = train_model_mlvae(net, train_dataloader_grouped, optimizer, num_epochs=num_epochs, a_posterior=a_posterior, device=device)
            else:
                loss = train_model(net, train_dataloader_grouped, optimizer, num_epochs=num_epochs, a_posterior=a_posterior, device=device)
            if not loss is None: break            
        net_path = osp.join(dirname, 'net_{}.pt'.format(i))
        save_net(net, net_path)

#%%

if (mode == 'train' or mode == 'kmeans') and num_cluster == 1:
    print('learning k-means to', dirname)
    if kmeans_num_cluster != 0:
        print('  using {} clusters'.format(kmeans_num_cluster))
    for i in range(start_instance, start_instance+num_instance):
        net_path = osp.join(dirname, 'net_{}.pt'.format(i))
        net = load_net(net_path)
        res = evalu.calc_outputs(net, train_dataloader, device=device)
        if group_size == 1:
            lat = torch.cat([res.contents, res.views], 1).squeeze().numpy()
        else:
            lat = res.contents.squeeze().numpy()
        if kmeans_num_cluster == 0:
            print('kmeans_num_cluster option has to be specified')
            quit()
        kmeans = KMeans(n_clusters=kmeans_num_cluster, verbose=0, n_init=10).fit(lat)
        kmeans_path = osp.join(dirname, 'kmeans_k{}_{}.pt'.format(kmeans_num_cluster, i))
        torch.save(kmeans, kmeans_path)    
    
    
#%%

if mode == 'test':
    print('loading models from', dirname)
    if kmeans_num_cluster != 0:
        print('  for k-means results with {} clusters'.format(kmeans_num_cluster))
    nets = []
    kms = []
    for i in range(start_instance, start_instance+num_instance):
        net_path = osp.join(dirname, 'net_{}.pt'.format(i))
        net = load_net(net_path)
        nets.append(net)
        if num_cluster == 1:
            if kmeans_num_cluster == 0:
                kmeans_path = osp.join(dirname, 'kmeans_{}.pt'.format(i))
            else:
                kmeans_path = osp.join(dirname, 'kmeans_k{}_{}.pt'.format(kmeans_num_cluster, i))
            kmeans = torch.load(kmeans_path)
            kms.append(kmeans)
    
    #%%
            
    ress = []
    for i in range(len(nets)):
        num_view = 30
        net = nets[i]
        print('net #{}: testing'.format(start_instance+i))
        res = evalu.test_model(net, test_dataloader, device=device)
        res.recon_acc = evalu.reconstrution_accuracy(res.inputs, res.recons).numpy()
        # res.swap_acc = evalu.swapping_accuracy2(net, res, num_view, device=device).numpy()
    
        print('net #{}: one-shot classification'.format(start_instance+i))
        lat = evalu.construct_combine_latent(net, res, num_cluster, group_size)
        res.fs_acc, res.fs_chance_acc = evalu.fewshot_accuracy_ids(lat, res.classes.numpy().astype(np.int32), metric=metric, device='cpu')
        
        if num_cluster == 1:
            kmeans = kms[i]
            res.km_clusters = torch.tensor(kmeans.predict(lat.numpy()))
            res.km_score = torch.tensor(-kmeans.transform(lat.numpy()))
            clusters = res.km_clusters.numpy()
            score = res.km_score.numpy()
        else:
            clusters = res.clusters.numpy()
            score = res.cposts.numpy()
            
        print('net #{}: invariant clustering'.format(start_instance+i))
        # res.acc, res.est_classes, res.cluster_to_class = evalu.clustering_accuracy(clusters, num_class, score, res.classes.numpy())
        # res.ba_acc, res.ba_est_labels = evalu.best_assignment_accuracy(clusters, num_class, res.classes.numpy(), np.arange(num_class))
        # res.chance_acc = 1 / len(torch.unique(res.classes))
        
        # res.ari = evalu.ari(clusters, res.classes.numpy())        
    
        if group_size != 1:
            print('net #{}: shape-view separability'.format(start_instance+i))
            res.acc_shape_shape, res.acc_view_shape, res.acc_chance_shape = evalu.shape_view_separability_ids(res.clusters.squeeze(), res.contents.squeeze(), res.views.squeeze(), num_cluster, res.classes.numpy().astype(np.int32), device=device)

        if purge_large_data:
            evalu.purge_large_data(res)
        ress.append(res)
    
    #%%
        
    if save_result:
        print('saving results to', dirname)
        for i in range(len(ress)):
            if num_cluster != 1:
                res_path = osp.join(dirname, 'res_{}.pt'.format(start_instance+i))
            else:
                res_path = osp.join(dirname, 'res_k{}_{}.pt'.format(kmeans_num_cluster, start_instance+i))
            torch.save(ress[i], res_path)
        
    #%%
            
if mode == 'show':
    ress = []
    for i in range(num_instance):
        if num_cluster != 1:
            res_path = osp.join(dirname, 'res_{}.pt'.format(start_instance+i))
        else:
            res_path = osp.join(dirname, 'res_k{}_{}.pt'.format(kmeans_num_cluster, start_instance+i))
        ress.append(torch.load(res_path))
    
if mode == 'test' or mode == 'show':        
    # for i in range(len(ress)):
    #     # net = nets[i]
    #     res = ress[i]
    #     # print('instance #{} (loss train={:.4f}): cla. acc.={:.4f}  BA acc.={:.4f}'.format(start_instance+i, net.info['train_loss'], res.acc, res.ba_acc))
    #     print('instance #{}: class acc={:.4f}  BA acc={:.4f}  recon acc={:.4f}'.format(start_instance+i, res.acc, res.ba_acc, res.recon_acc.mean()))
            
    # accs = [res.acc for res in ress]
    # print('classification accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(accs), np.std(accs), np.max(accs)))
    
    # ba_accs = [res.ba_acc for res in ress]
    # print('BA accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(ba_accs), np.std(ba_accs), np.max(ba_accs)))
    
    recon_accs = [res.recon_acc.mean() for res in ress]
    print('reconstruction accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(recon_accs), np.std(recon_accs), np.max(recon_accs)))
    
    # swap_accs = [res.swap_acc.mean() for res in ress]
    # print('swapping accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(swap_accs), np.std(swap_accs), np.max(swap_accs)))
    
    fs_accs = [np.mean(res.fs_acc) for res in ress]
    print('few-shot accuracy = mean {:.4f}, std {:.4f}, max {:.4f} (chance={:.4f})'.format(np.mean(fs_accs), np.std(fs_accs), np.max(fs_accs), ress[0].fs_chance_acc))
      
    if group_size != 1:
        rss_accs = [evalu.rss_weighted_average(res) for res in ress]
        print('shape->shape accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(rss_accs), np.std(rss_accs), np.max(rss_accs)))
    
        rvs_accs = [evalu.rvs_weighted_average(res) for res in ress]
        print('view->shape accuracy = mean {:.4f}, std {:.4f}, max {:.4f}'.format(np.mean(rvs_accs), np.std(rvs_accs), np.max(rvs_accs)))
    
#%%
    
   
