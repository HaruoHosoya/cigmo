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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms

import utils
from network64 import *
import network64iic
from train import train_model
import evalu 
import datasets 

#%%

device = utils.get_device()
    
print('device:', device)

torch.backends.cudnn.benchmark = True

# res_root = os.path.join(os.environ['HOME'], 'resultsets/mvae/20200824/shapenet')
res_root = os.path.join(os.environ['HOME'], 'resultsets/mvae/20200927/shapenet')
ds_path = os.environ['HOME'] + '/resultsets/mvae/20200715/shapenet/multi_view2/'

#%%

content_dim = 100
view_dim = 3
group_size = 3
batch_size = 100
num_instance = 10
single_view_encoder = True

#%%

subset2, class_dict2 = datasets.shapenet_standard_subset(2)
subset3, class_dict3 = datasets.shapenet_standard_subset(3)
subset5, class_dict5 = datasets.shapenet_standard_subset(5)
subset10, class_dict10 = datasets.shapenet_standard_subset(10)

#%%

print('loading dataset from', ds_path)

start = time.time()

test_dataset2 = datasets.load_shapenet_dataset(root=ds_path, subset=subset2, split='test')
test_dataloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

test_dataset3 = datasets.load_shapenet_dataset(root=ds_path, subset=subset3, split='test')
test_dataloader3 = torch.utils.data.DataLoader(test_dataset3, batch_size=batch_size, shuffle=False)

test_dataset5 = datasets.load_shapenet_dataset(root=ds_path, subset=subset5, split='test')
test_dataloader5 = torch.utils.data.DataLoader(test_dataset5, batch_size=batch_size, shuffle=False)

test_dataset10 = datasets.load_shapenet_dataset(root=ds_path, subset=subset10, split='test')
test_dataloader10 = torch.utils.data.DataLoader(test_dataset10, batch_size=batch_size, shuffle=False)

val_dataset10 = datasets.load_shapenet_dataset(root=ds_path, subset=subset10, split='val')
val_dataloader10 = torch.utils.data.DataLoader(val_dataset10, batch_size=batch_size, shuffle=False)

print('loading time: {0} sec'.format(time.time() - start))

#%%

os.makedirs(os.path.join(res_root, 'examples'), exist_ok=True)

########### sample input images ###########
#%%

inputs = []
cla = []
for xs, ls in val_dataloader10:
    inputs.append(xs)
    cla.append(torch.tensor([class_dict10[l] for l in ls[0]]))
inputs = torch.cat(inputs, 0)    
cla = torch.cat(cla, 0)

idx0 = (cla==0).nonzero().squeeze()
inputs0 = inputs[idx0,:,:,:]
idx1 = (cla==1).nonzero().squeeze()
inputs1 = inputs[idx1,:,:,:]

#%%

imgs0 = inputs0[[0,17,30,40],:,:,:]
imgs1 = inputs1[[0,17,30,40],:,:,:]
imgs = torch.cat([imgs0, imgs1], dim=0)
imgs = torch.cat([imgs[0::2], imgs[1::2]], dim=0)
img=torchvision.utils.make_grid(imgs, nrow=4, padding=2, pad_value=1, normalize=True)

plt.imshow(img.numpy().transpose((1,2,0)))

torchvision.utils.save_image(img, os.path.join(res_root, 'examples/example1.png'))

#%%

imgs = inputs0[0:30,:,:,:]
img=torchvision.utils.make_grid(imgs, nrow=10, padding=2, pad_value=1, normalize=True)

plt.imshow(img.numpy().transpose((1,2,0)))

torchvision.utils.save_image(img, os.path.join(res_root, 'examples/example2.png'))

#%%

imgs0 = inputs0[[0,11,17,30,39,49,62,76,83],:,:,:]
imgs1 = inputs1[[0,11,17,30,39,49,62,76,83],:,:,:]
imgs = torch.cat([imgs0, imgs1], dim=0)
imgs = torch.cat([imgs[0::3], imgs[1::3], imgs[2::3]], dim=0)
img=torchvision.utils.make_grid(imgs, nrow=6, padding=2, pad_value=1, normalize=True)

plt.imshow(img.numpy().transpose((1,2,0)))

torchvision.utils.save_image(img, os.path.join(res_root, 'examples/example3.png'))

#%%

imgs = inputs1[0:30*12:30,:,:,:]
img=torchvision.utils.make_grid(imgs, nrow=6, padding=2, pad_value=1, normalize=True)

plt.imshow(img.numpy().transpose((1,2,0)))

torchvision.utils.save_image(img, os.path.join(res_root, 'examples/example4.png'))

#%%

l = []
for i in range(0,10):
    idx1 = (cla==i).nonzero().squeeze()
    inputs2 = inputs[idx1,:,:,:]
    l.append(inputs2[0:1,:,:,:])
    
imgs = torch.cat(l, dim=0)
img=torchvision.utils.make_grid(imgs, nrow=10, padding=2, pad_value=1, normalize=True)

plt.imshow(img.numpy().transpose((1,2,0)))

torchvision.utils.save_image(img, os.path.join(res_root, 'examples/example5.png'))

########### sample results ###########
#%%

def load_example_net(num_cluster, group_size, num_class, iic=False, overcluster=False, kmeans_num_cluster=None):
    if iic:
        if overcluster:
            num_overcluster = num_cluster * 5
        else:
            num_overcluster = None
        dirname = os.path.join(res_root, 'iic_s{}_c{}_k{}_o{}'.format(num_class, num_cluster, group_size, num_overcluster))
    else:
        dirname = osp.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class, num_cluster, group_size, content_dim, view_dim, single_view_encoder))
    ress = []
    for i in range(num_instance):
        if kmeans_num_cluster is None:
            res_path = os.path.join(dirname, 'res_{}.pt'.format(i))
        else:
            res_path = os.path.join(dirname, 'res_k{}_{}.pt'.format(kmeans_num_cluster, i))
        ress.append(torch.load(res_path))
    scores = [res.ba_acc for res in ress]
    # print(scores)
    # net_idx = np.argsort(scores)[int(num_instance/2)-1]
    net_idx = np.argmin(np.abs(scores - np.mean(scores)))
    net_path = os.path.join(dirname, 'net_{}.pt'.format(net_idx))
    print('loading net from', net_path)
    if iic:
        net = network64iic.load_net(net_path)
    else:
        net = load_net(net_path)
    res = ress[net_idx]
    return net, res, ress

#%%

net_mgvae2, res_mgvae2a = load_example_net(2, group_size, 2)
res_mgvae2 = evalu.test_model(net_mgvae2, test_dataloader2, class_dict2, device=device)

#%%

net_mgvae3, res_mgvae3a = load_example_net(3, group_size, 3)
res_mgvae3 = evalu.test_model(net_mgvae3, test_dataloader3, class_dict3, device=device)

#%%

net_mvae3, res_mvae3a = load_example_net(3, 1, 3)
res_mvae3 = evalu.test_model(net_mvae3, test_dataloader3, class_dict3, device=device)

#%%

net_gvae3, res_gvae3a = load_example_net(1, group_size, 3)
res_gvae3 = evalu.test_model(net_gvae3, test_dataloader3, class_dict3, device=device)

#%%

net_iic3, res_iic3a = load_example_net(3, group_size, 3, iic=True)
res_iic3 = evalu.calc_clustering(net_iic3, test_dataloader3, class_dict3, device=device, keep_inputs=True)

#%%

net_mgvae5, res_mgvae5a = load_example_net(5, group_size, 5)
res_mgvae5 = evalu.test_model(net_mgvae5, test_dataloader5, class_dict5, device=device)

#%%

net_mvae5, res_mvae5a = load_example_net(5, 1, 5)
res_mvae5 = evalu.test_model(net_mvae5, test_dataloader5, class_dict5, device=device)

#%%

net_gvae5, res_gvae5a = load_example_net(1, group_size, 5)
res_gvae5 = evalu.test_model(net_gvae5, test_dataloader5, class_dict5, device=device)

#%%

net_mgvae10, res_mgvae10a = load_example_net(10, group_size, 10)
res_mgvae10 = evalu.test_model(net_mgvae10, test_dataloader10, class_dict10, device=device)

#%%

net_mvae10, res_mvae10a = load_example_net(10, 1, 10)
res_mvae10 = evalu.test_model(net_mvae10, test_dataloader10, class_dict10, device=device)

#%%

net_gvae10, res_gvae10a = load_example_net(1, group_size, 10)
res_gvae10 = evalu.test_model(net_gvae10, test_dataloader10, class_dict10, device=device)

#%%

def display_save(img, display, file_name):
    if display:
        pimg = img.cpu().detach().numpy().transpose((1, 2, 0))
        plt.imshow(pimg)
    if file_name != None:
        torchvision.utils.save_image(img, os.path.join(res_root, 'examples', file_name))

#%% reconstruction

def show_reconstruction(res, grid_size=5, seed=0, display=True, file_name=None):
    rs = numpy.random.RandomState(seed)   
    idx = rs.permutation(res.inputs.size(0))
    inp_img = torchvision.utils.make_grid(res.inputs[idx[0:grid_size**2],:,:,:], nrow=grid_size, padding=1, pad_value=0)
    rec_img = torchvision.utils.make_grid(res.recons[idx[0:grid_size**2],:,:,:], nrow=grid_size, padding=1, pad_value=0)
    img = torchvision.utils.make_grid([inp_img, rec_img], padding=5, normalize=True, pad_value=1)
    display_save(img, display, file_name)
    
#%% clustering

def show_clustering(net, res, nsample=25, grid_nrow=5, table_nrow=3, seed=0, display=True, file_name=None, clusters=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    if clusters is not None: 
        num_cluster = clusters.max().item() + 1
    else: 
        num_cluster = net.num_cluster
        clusters = res.clusters    
    for c in range(0, num_cluster):
        idx = (clusters == c).nonzero().squeeze()
        if idx.nelement() < nsample: continue
        idx = idx[rs.permutation(len(idx))]
        g = torchvision.utils.make_grid(res.inputs[idx[0:nsample],:,:,:], nrow=grid_nrow, padding=1, pad_value=0)
        imgs.append(g)        
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow = table_nrow, padding=4, pad_value=1)
    display_save(img, display, file_name)

#%% swapping

def show_swapping(net, res, list_size=5, table_nrow=3, seed=1, display=True, file_name=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    for c in range(0, net.num_cluster):
        idx = (res.clusters == c).nonzero().squeeze()
        if idx.nelement() < list_size ** 2: continue
        idx = idx[rs.permutation(len(idx))]
        list1 = res.inputs[idx[0:list_size]]
        list2 = res.inputs[idx[list_size:list_size*2]]
        _, pimg = evalu.swapping(net, c, list1, list2)
        g = torchvision.utils.make_grid(pimg, nrow=list_size, padding=1, pad_value=0)
        imgs.append(g)        
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    display_save(img, display, file_name)
    
#%% swapping2

def show_swapping2(net, res, list1_size=8, list2_size=4, seed=1, display=True, file_name=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    idx = rs.permutation(res.inputs.size(0))
    list1 = res.inputs[idx[0:list1_size]]
    list2 = res.inputs[idx[list2_size:list2_size*2]]
    _, img = evalu.swapping2(net, list1, list2)
    display_save(img, display, file_name)
   
#%% random image generation

def show_random(net, res, grid_size=5, table_nrow=3, seed=0, display=True, file_name=None, only_valued=False):
    imgs = []
    for k in range(0, net.num_cluster):
        idx = (res.clusters == k).nonzero().squeeze()
        if idx.nelement() < grid_size**2: continue
        if only_valued:
            cstd = res.contents[idx].std(dim=0).squeeze()
            vstd = res.views[idx].std(dim=0).squeeze()
            imgs0 = evalu.gen_random_images(net, k, grid_size**2, device='cpu', content_std=cstd, view_std=vstd)
        else:
            imgs0 = evalu.gen_random_images(net, k, grid_size**2, device='cpu')
        g = torchvision.utils.make_grid(imgs0, nrow=5, padding=1, pad_value=0)
        imgs.append(g)        
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    display_save(img, display, file_name)

#%% random image generation

def show_random2(net, res, list1_size=5, list2_size=5, table_nrow=3, seed=0, display=True, file_name=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    idx = rs.permutation(res.inputs.size(0))
    list2 = res.inputs[idx[0:list2_size]]
    for k in range(0, net.num_cluster):
        idx = (res.clusters == k).nonzero().squeeze()
        if idx.nelement() < list1_size*list2_size: continue
        imgs0 = evalu.gen_random_images2(net, k, list1_size, list2, device='cpu')
        g = torchvision.utils.make_grid(imgs0, nrow=list1_size, padding=1, pad_value=0)
        imgs.append(g)        
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    display_save(img, display, file_name)

#%%

def show_2d_morph(net, res, grid_size=5, table_nrow=3, seed=0, display=True, file_name=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    idx = rs.permutation(res.inputs.size(0))
    list2 = res.inputs[idx[0:net.num_cluster]]
    for k in range(0, net.num_cluster):
        idx = (res.clusters == k).nonzero().squeeze()
        if idx.nelement() < grid_size**2: continue
        s = res.contents[idx, :, 0, 0].squeeze().std(0).squeeze()
        sigdims = torch.argsort(s, 0, descending=True)
        imgs0 = evalu.gen_2d_images(net, k, sigdims[0], sigdims[1], torch.linspace(-1, 1, grid_size), list2[k:k+1], device='cpu')
        imgs0 = imgs0.cpu().detach()
        g = torchvision.utils.make_grid(imgs0, nrow=grid_size, padding=1, pad_value=0)
        imgs.append(g)        
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    display_save(img, display, file_name)

#%%

def show_2d_morph2(net, res, grid_size=5, table_nrow=3, seed=0, display=True, file_name=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    for k in range(0, net.num_cluster):
        idx = (res.clusters == k).nonzero().squeeze()
        if idx.nelement() < grid_size**2: continue
        idx = idx[rs.permutation(len(idx))]
        inp1 = res.inputs[idx[0]:idx[0]+1]
        inp2 = res.inputs[idx[1]:idx[1]+1]
        imgs0 = evalu.gen_2d_images2(net, k, inp1, inp2, grid_size, device='cpu')
        imgs0 = imgs0.cpu().detach()
        g = torchvision.utils.make_grid(imgs0, nrow=grid_size, padding=1, pad_value=0)
        imgs.append(g)        
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    display_save(img, display, file_name)

#%%
        
show_reconstruction(res_mgvae10)

#%%

show_clustering(net_mgvae2, res_mgvae2, nsample=18, grid_nrow=3, file_name='shapenet_cluster_mgvae2_ex1.png')

#%%

show_clustering(net_mgvae3, res_mgvae3, nsample=24, grid_nrow=6, file_name='shapenet_cluster_mgvae3_ex1.png')

#%%

show_clustering(net_mvae3, res_mvae3, nsample=24, grid_nrow=6, file_name='shapenet_cluster_mvae3_ex1.png')

#%%

show_clustering(net_mgvae3, res_mgvae3, nsample=24, grid_nrow=8, file_name='shapenet_cluster_mgvae3_ex2.png')

#%%

show_clustering(net_mvae3, res_mvae3, nsample=24, grid_nrow=8, file_name='shapenet_cluster_mvae3_ex2.png')

#%%

show_clustering(net_gvae3, res_gvae3, nsample=24, grid_nrow=6, file_name='shapenet_cluster_gvae3_ex1.png', clusters=res_gvae3a.km_clusters)

#%%

show_clustering(net_gvae3, res_gvae3, nsample=24, grid_nrow=8, file_name='shapenet_cluster_gvae3_ex2.png', clusters=res_gvae3a.km_clusters)

#%%

show_clustering(net_iic3, res_iic3, nsample=24, grid_nrow=8, file_name='shapenet_cluster_iic3_ex2.png')

#%%

show_swapping(net_mgvae3, res_mgvae3, seed=4, file_name='shapenet_swap_mgvae3_ex1.png')

#%%

show_swapping2(net_mgvae10, res_mgvae10, file_name='shapenet_swap_mgvae10_ex1.png')

#%%

show_swapping2(net_gvae10, res_gvae10, file_name='shapenet_swap_gvae10_ex1.png')

#%%

show_random(net_mgvae3, res_mgvae3, file_name='shapenet_rand_mgvae3_ex1.png')
        
#%%

show_random2(net_mgvae3, res_mgvae3, file_name='shapenet_rand2_mgvae3_ex1.png')

#%%

show_2d_morph(net_mgvae3, res_mgvae3, seed=3, file_name='shapenet_morph_mgvae3_ex1.png')

#%%

show_2d_morph2(net_mgvae3, res_mgvae3, seed=0, file_name='shapenet_morph2_mgvae3_ex1.png')
