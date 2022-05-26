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
import sklearn.metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms

import utils
from network64 import MGVAENet, save_net, load_net
from train import train_model, train_model_mlvae
import evalu
import datasets
import data.mvc as mvc

from sklearn.cluster import KMeans

# %%

device = utils.get_device()

print('device:', device)

torch.backends.cudnn.benchmark = True

res_root = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/mvc')
ds_path = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/data/mvc64')

#res_root = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/mvc96')
#ds_path = osp.join(os.environ['HOME'], 'resultsets/mvae/20211215/data/mvc96')

os.makedirs(res_root, exist_ok=True)

# %%

use_subset = True

# %%

batch_size = 100
# color = False
color = True
single_view_encoder = True

# %%

num_cluster = 7
content_dim = 100
view_dim = 3
group_size = 3

# %%

print('loading dataset from', ds_path)

tran = transforms.Compose(
    ([transforms.Grayscale()] if not color else []) +
    [
        transforms.ToTensor(),
    ])

start = time.time()

test_dataset = datasets.DatasetFromTensor(
    root=ds_path, split='test', transform=tran)
if use_subset:
    test_dataset = datasets.mvc_standard_subset(test_dataset)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

print('loading time: {0} sec'.format(time.time() - start))

# %%

i = 7
single_view_encoder = True
dirname = osp.join(res_root, 'models_c{}_k{}_m{}_l{}_v{:d}'.format(
    num_cluster, group_size, content_dim, view_dim, single_view_encoder))
net_path = osp.join(dirname, 'net_{}.pt'.format(i))
print('loading from:', net_path)
net = load_net(net_path)

# %%

res = evalu.test_model(net, test_dataloader, device=device)

# %%

for k in range(num_cluster):
    print('  estimated cluster #{} : {}'.format(
        k, torch.sum(res.clusters == k)))

# %%

os.makedirs(os.path.join(res_root, 'examples'), exist_ok=True)

# %%


def display_save(img, display, file_name):
    if display:
        pimg = img.cpu().detach().numpy().transpose((1, 2, 0))
        plt.imshow(pimg)
    if file_name != None:
        torchvision.utils.save_image(
            img, os.path.join(res_root, 'examples', file_name))

# %%


def show_clustering(net, res, nsample=24, grid_nrow=6, table_nrow=3, seed=5, display=True, file_name=None, clusters=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    if clusters is not None:
        num_cluster = clusters.max().item() + 1
    else:
        num_cluster = net.num_cluster
        clusters = res.clusters
    idx_list = []
    for c in range(0, num_cluster):
        idx = (clusters == c).nonzero().squeeze()
        if idx.nelement() < nsample:
            continue
        idx = idx[rs.permutation(len(idx))]
        g = torchvision.utils.make_grid(
            res.inputs[idx[0:nsample], :, :, :], nrow=grid_nrow, padding=1, pad_value=0)
        imgs.append(g)
        idx_list.append(idx)
    img = torchvision.utils.make_grid(
        imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    img = (img + 0.2) / 1.2
    display_save(img, display, file_name)
    return idx_list

# %%


def show_swapping(net, res, list_size1=6, list_size2=4, table_nrow=3, seed=11, display=True, file_name=None):
    rs = numpy.random.RandomState(seed)
    imgs = []
    idx_list = []
    for c in range(0, net.num_cluster):
        idx = (res.clusters == c).nonzero().squeeze()
        if idx.nelement() < list_size1 * list_size2:
            continue
        idx = idx[rs.permutation(len(idx))]
        list1 = res.inputs[idx[0:list_size1]]
        list2 = res.inputs[idx[list_size1:list_size1+list_size2]]
        _, pimg = evalu.swapping(net, c, list1, list2, column_content=False)
        g = torchvision.utils.make_grid(
            pimg, nrow=list_size1, padding=1, pad_value=0, normalize=True)
        imgs.append(g)
        idx_list.append(idx)
    img = torchvision.utils.make_grid(
        imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    img = (img + 0.2) / 1.2
    display_save(img, display, file_name)
    return idx_list

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
        g = torchvision.utils.make_grid(imgs0, nrow=grid_size, padding=1, pad_value=0)
        imgs.append(g)        
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow=table_nrow, padding=4, pad_value=1)
    display_save(img, display, file_name)

#%%

    
def write_attr_latex_table(est_cluster, cluster_attr_score, f):
    num_est_cluster = max(est_cluster) + 1

    c = 1    
    for i in range(num_est_cluster):
        if sum(est_cluster == i) == 0:
            continue
        a = sorted(cluster_attr_score[i].items(),
                   key=lambda x: x[1][0], reverse=True)
        f.write('{} & '.format(c))
        for j in range(0, 10):
            name = a[j][0]
            f1, recall, precision = a[j][1]
            # print('{} ({:0.2f}, {:0.2f}, {:0.2f}); '.format(name, f1, recall, precision), end='')
            # print('{} ({:0.2f}, {:0.2f}); '.format(name, f1, recall), end='')
            # f.write('\\verb|{}| ({:0.2f}); '.format(name, f1))
            f.write('{} ({:0.2f}); '.format(name, f1))
        f.write('\\\\\n')
        c += 1
    
# %%

show_clustering(net, res, nsample=12, grid_nrow=4, seed=1, table_nrow=4, file_name='mvc_cluster_mgvae7_ex1.png')

# %%

show_swapping(net, res, list_size1=4, list_size2=3, seed=4, table_nrow=4, file_name='mvc_swap_mgvae7_ex1.png')

#%%

show_2d_morph2(net, res, seed=0, table_nrow=4, grid_size=4, file_name='mvc_morph2_mgvae7_ex1.png')

#%%

show_random(net, res, table_nrow=4, grid_size=4, file_name='mvc_rand_mgvae7_ex1.png')

# %%

est_cluster = res.clusters.numpy()
num_est_cluster = num_cluster

ks = list(test_dataset.get_attr(0).keys())
ks.remove('filename')
ks.remove('itemN')
attr_dict = {}
for k in ks:
    attr_dict[k] = []

for i in range(len(test_dataset)):
    attr = test_dataset.get_attr(i)
    for k in ks:
        attr_dict[k].append(attr[k])

cluster_attr_score = []

for i in range(num_est_cluster):
    cluster_attr_score.append({})
    for k in ks:
        c = (est_cluster == i).astype(np.int32)
        cluster_attr_score[i][k] = (sklearn.metrics.f1_score(attr_dict[k], c, zero_division=0),
                                    sklearn.metrics.recall_score(attr_dict[k], c, zero_division=0),
                                    sklearn.metrics.precision_score(attr_dict[k], c, zero_division=0))

#%%

latex=False

if latex:
    print

for i in range(num_est_cluster):
    if sum(est_cluster == i) == 0:
        continue
    a = sorted(cluster_attr_score[i].items(),
               key=lambda x: x[1][0], reverse=True)
    print('cluster #{}: '.format(i), end='')
    for j in range(0, 10):
        name = a[j][0]
        f1, recall, precision = a[j][1]
        # print('{} ({:0.2f}, {:0.2f}, {:0.2f}); '.format(name, f1, recall, precision), end='')
        # print('{} ({:0.2f}, {:0.2f}); '.format(name, f1, recall), end='')
        print('{} ({:0.2f}); '.format(name, f1), end='')
    print()
    
#%%

with open(os.path.join(res_root, 'examples', 'res_mvc_attrs.txt'), mode='w') as out:
    write_attr_latex_table(est_cluster, cluster_attr_score, out)


# %%

# lat = torch.zeros(res.inputs.size(0), net.num_cluster, net.content_dim)
# for j in range(res.inputs.size(0)):
#     lat[j, res.clusters[j], :] = res.contents[j].squeeze()
# lat = lat.reshape(res.inputs.size(0), -1)

# accs, chance_accs = evalu.fewshot_accuracy_ids(lat, res.classes.numpy().astype(
#     np.int32), n_shot=1, n_repetition=5, metric='euclidean', device='cpu')
# print(accs)

# # %%

# acc_shape_shape, acc_view_shape, acc_chance_shape = evalu.shape_view_separability_ids(res.clusters.squeeze(), res.contents.squeeze(), res.views.squeeze(), num_cluster, res.classes.numpy().astype(np.int32), device=device)


