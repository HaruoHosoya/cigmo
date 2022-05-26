#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:28:41 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""


import sys
import os as os
import os.path as osp
import numpy as np
import numpy.random
import json
from PIL import Image
import argparse
from distutils.util import strtobool
from scipy import stats
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms

import utils
from network64 import MGVAENet, load_net, save_net
from train import train_model
import evalu 
import datasets 
from bench_util import load_results, show_table, write_latex_table, plot_results

#%%

# res_root = os.path.join(os.environ['HOME'], 'resultsets/mvae/20200824/shapenet')
res_root = os.path.join(os.environ['HOME'], 'resultsets/mvae/20200927/shapenet')
num_instance = 10
content_dim = 100
view_dim = 3
group_size = 3

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=res_root, help="model root path")
parser.add_argument('--num_instance', type=int, default=num_instance, help="number of instances")
parser.add_argument('--latex', type=strtobool, default=True, help="LaTeX format")
parser.add_argument('--save_result', type=strtobool, default=False, help="save results")
parser.add_argument('--save_plot', type=strtobool, default=False, help="save plot figure")

options = parser.parse_args()
num_instance = options.num_instance
latex = bool(options.latex)
save_result = bool(options.save_result)
save_plot = bool(options.save_plot)

print(options.latex)

#%%

# all_num_class_cases = [2, 3, 5, 10]
# all_num_class_cases = [2, 3, 5]
all_num_class_cases = [3, 5, 10]

cases3 = [3, 5, 10]
cases2 = [5, 10]

#%%

all_ress = {}

for num_class in all_num_class_cases:
    ress_mgvae = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class, num_class, group_size, content_dim, view_dim, True)))
    ress_mvae  = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class, num_class,          1, content_dim, view_dim, True)))
    ress_gvae  = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class,         1, group_size, content_dim, view_dim, True)))
    ress_vae   = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class,         1,          1, content_dim, view_dim, True)))
    ress_mlvae = load_results(os.path.join(res_root, 'mlvae_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class,         1, group_size, content_dim, view_dim, True)))
    ress_iic   = load_results(os.path.join(res_root, 'iic_s{}_c{}_k{}_o{}'.format(num_class, num_class, group_size, None)))
    ress_iicoc = load_results(os.path.join(res_root, 'iic_s{}_c{}_k{}_o{}'.format(num_class, num_class, group_size, num_class * 5)))

    ress_mgvae_mv = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class, num_class, group_size, content_dim, view_dim, False)))
    ress_mgvae_ap = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}_ap'.format(num_class, num_class, group_size, content_dim, view_dim, True)))
    ress_mgvae_al = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}_al'.format(num_class, num_class, group_size, content_dim, view_dim, True)))

    all_ress['IIC', num_class]   = ress_iic
    all_ress['IICoc', num_class] = ress_iicoc
    all_ress['VAE', num_class]   = ress_vae
    all_ress['MLVAE', num_class] = ress_mlvae
    all_ress['MixVAE', num_class]  = ress_mvae
    all_ress['GVAE', num_class]  = ress_gvae
    all_ress['CIGMO', num_class] = ress_mgvae

    all_ress['CIGMOmv', num_class] = ress_mgvae_mv
    all_ress['CIGMOap', num_class] = ress_mgvae_ap
    all_ress['CIGMOal', num_class] = ress_mgvae_al

#%%
    
task_clustering = { 'getter': lambda x: x.ba_acc, 'getter_chance': lambda x: x.chance_acc, 'take_max': True, 'percent': True }
task_clustering_ari = { 'getter': lambda x: x.ari, 'getter_chance': None, 'take_max': True, 'percent': False }
task_swapping = { 'getter': lambda x: x.swap_acc.mean(), 'getter_chance': None, 'take_max': False, 'percent': False }
task_oneshot = { 'getter': lambda x: np.mean(x.fs_acc), 'getter_chance': lambda x: x.fs_chance_acc, 'take_max': True, 'percent': True }
task_rss = { 'getter': lambda x: evalu.rss_weighted_average(x), 'getter_chance': None, 'take_max': True, 'percent': True }
task_rvs = { 'getter': lambda x: evalu.rvs_weighted_average(x), 'getter_chance': None, 'take_max': False, 'percent': True }
task_clustering_alt = { 'getter': lambda x: x.ba_acc, 'getter_chance': None, 'take_max': True, 'percent': True }
task_oneshot_alt = { 'getter': lambda x: np.mean(x.fs_acc), 'getter_chance': None, 'take_max': True, 'percent': True }

#%%

methods1 = ['IIC', 'IICoc', 'VAE', 'MixVAE', 'MLVAE', 'GVAE', 'CIGMO']
methods2 = ['VAE', 'MixVAE', 'MLVAE', 'GVAE', 'CIGMO']
methods3 = ['MLVAE', 'GVAE', 'CIGMO']
methods4 = ['CIGMO', 'CIGMOap', 'CIGMOal', 'CIGMOmv']

#%%

if latex:

    if save_result:
        with open(os.path.join(res_root, 'examples', 'res_clustering.txt'), mode='w') as out:
            write_latex_table(all_ress, cases3, methods1, [task_clustering], out)
        with open(os.path.join(res_root, 'examples', 'res_clustering_ari.txt'), mode='w') as out:
            write_latex_table(all_ress, cases3, methods1, [task_clustering_ari], out)
        with open(os.path.join(res_root, 'examples', 'res_oneshot.txt'), mode='w') as out:
            write_latex_table(all_ress, cases3, methods2, [task_oneshot], out)
        with open(os.path.join(res_root, 'examples', 'res_rss_rvs.txt'), mode='w') as out:
            write_latex_table(all_ress, cases2, methods3, [task_swapping, task_rss, task_rvs], out)
        with open(os.path.join(res_root, 'examples', 'res_alt.txt'), mode='w') as out:
            write_latex_table(all_ress, cases3, methods4, [task_clustering_alt, task_oneshot_alt], out)
        with open(os.path.join(res_root, 'examples', 'res_alt_swapping.txt'), mode='w') as out:
            write_latex_table(all_ress, cases3, methods4, [task_swapping], out)
        with open(os.path.join(res_root, 'examples', 'res_alt_rss_rvs.txt'), mode='w') as out:
            write_latex_table(all_ress, cases3, methods4, [task_rss, task_rvs], out)
    else:
        write_latex_table(all_ress, cases3, methods1, [task_clustering], sys.stdout)
        write_latex_table(all_ress, cases3, methods1, [task_clustering_ari], sys.stdout)
        write_latex_table(all_ress, cases3, methods2, [task_oneshot], sys.stdout)
        write_latex_table(all_ress, cases2, methods3, [task_swapping, task_rss, task_rvs], sys.stdout)
        write_latex_table(all_ress, cases3, methods4, [task_clustering_alt, task_oneshot_alt], sys.stdout)
        write_latex_table(all_ress, cases3, methods4, [task_swapping], sys.stdout)
        write_latex_table(all_ress, cases3, methods4, [task_rss, task_rvs], sys.stdout)
        
        
else:        
    
    print('Clustering accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods1, lambda x: x.ba_acc, percent=True)
    
    print('Clustering ARI:')
    show_table(all_ress, all_num_class_cases, methods1, lambda x: x.ari, percent=False)

    print('Reconstruction error (normalized MSE):')
    show_table(all_ress, all_num_class_cases, methods2, lambda x: x.recon_acc.mean(), percent=False)
    
    print('Swapping error (normalized MSE):')
    show_table(all_ress, all_num_class_cases, methods2, lambda x: x.swap_acc.mean(), percent=False)
    
    print('One-shot accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods2, lambda x: np.mean(x.fs_acc), percent=True)

    print('Shape->shape accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods3, lambda x: evalu.rss_weighted_average(x), percent=True)

    print('View->shape accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods3, lambda x: evalu.rvs_weighted_average(x), percent=True)

#%%

all_num_cluster_cases_set = [[3, 5, 10, 15], [10, 15, 20, 30]]
num_class_set = [3, 10]

#%%

for i in range(len(num_class_set)):
    all_ress = {}
    all_num_cluster_cases = all_num_cluster_cases_set[i]
    num_class = num_class_set[i]
    
    print('Set: {} classes'.format(num_class))
    
    for num_cluster in all_num_cluster_cases:
        k = num_cluster if num_cluster != num_class else 0
        ress_mgvae = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class, num_cluster, group_size, content_dim, view_dim, True)))
        ress_mvae  = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class, num_cluster,          1, content_dim, view_dim, True)))
        ress_gvae  = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class,           1, group_size, content_dim, view_dim, True)), kmeans_num_cluster=k)
        # ress_vae   = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class,           1,          1, content_dim, view_dim, True)))
        # ress_mlvae = load_results(os.path.join(res_root, 'mlvae_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class,            1, group_size, content_dim, view_dim, True)))
        ress_iic   = load_results(os.path.join(res_root, 'iic_s{}_c{}_k{}_o{}'.format(num_class, num_cluster, group_size, None)))
        # ress_iicoc = load_results(os.path.join(res_root, 'iic_s{}_c{}_k{}_o{}'.format(num_class, num_cluster, group_size, num_class * 5)))
    
        all_ress['IIC', num_cluster]   = ress_iic
        # all_ress['IICoc', num_cluster] = ress_iicoc
        # all_ress['VAE', num_cluster]   = ress_vae
        # all_ress['MLVAE', num_cluster] = ress_mlvae
        all_ress['MixVAE', num_cluster]  = ress_mvae
        all_ress['GVAE+km', num_cluster]  = ress_gvae
        all_ress['CIGMO', num_cluster] = ress_mgvae
    
    #%%
    
    methods5 = ['IIC', 'MixVAE', 'GVAE+km', 'CIGMO']
    methods6 = ['MixVAE', 'GVAE+km', 'CIGMO']
    methods7 = ['IIC', 'MixVAE', 'CIGMO']
    
    #%%
    
    if not latex:    
        print('Clustering ARI:')
        show_table(all_ress, all_num_cluster_cases, methods5, lambda x: x.ari, percent=False)
        print('One-shot accuracy (%):')
        show_table(all_ress, all_num_cluster_cases, methods6, lambda x: np.mean(x.fs_acc), percent=True)
        print('# of effective categories:')
        show_table(all_ress, all_num_cluster_cases, methods7, lambda x: evalu.num_effective_clusters(x), percent=False)
    
    #%%
    
    if save_plot:
        plot_results(all_ress, all_num_cluster_cases, methods5, lambda x: x.ari, 
                     os.path.join(res_root, 'examples', 'more_categs_ari_c{}.pdf'.format(num_class)), 
                     xlabel='# of categories', ylabel='ARI', ylim=(0,1), 
                     title='{} classes'.format(num_class),
                     percent=False)
        plot_results(all_ress, all_num_cluster_cases, methods6, lambda x: np.mean(x.fs_acc), 
                     os.path.join(res_root, 'examples', 'more_categs_oneshot_c{}.pdf'.format(num_class)), 
                     xlabel='# of categories', ylabel='1-shot accuracy (%)', ylim=(0,100),
                     title='{} classes'.format(num_class),
                     percent=True)
        
#%%

all_num_class_cases = [5, 10]
all_group_size = [2, 3, 5]

#%%

all_ress = {}
all_ress2 = {}

for num_class in all_num_class_cases:    
    print('Case: {} classes'.format(num_class))
    
    for group_size in all_group_size:    
        ress_mgvae = load_results(os.path.join(res_root, 'models_s{}_c{}_k{}_m{}_l{}_v{:d}'.format(num_class, num_class, group_size, content_dim, view_dim, True)))    
        all_ress['K={}'.format(group_size), num_class] = ress_mgvae
        all_ress2['{} classes'.format(num_class), group_size] = ress_mgvae

#%%

methods8 = ['K={}'.format(group_size) for group_size in all_group_size]
methods9 = ['{} classes'.format(num_class) for num_class in all_num_class_cases]

#%%

if not latex:    
    print('Clustering accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods8, lambda x: x.ba_acc, percent=True)
    
    print('Clustering ARI:')
    show_table(all_ress, all_num_class_cases, methods8, lambda x: x.ari, percent=False)

    print('Reconstruction error (normalized MSE):')
    show_table(all_ress, all_num_class_cases, methods8, lambda x: x.recon_acc.mean(), percent=False)
    
    print('Swapping error (normalized MSE):')
    show_table(all_ress, all_num_class_cases, methods8, lambda x: x.swap_acc.mean(), percent=False)
    
    print('One-shot accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods8, lambda x: np.mean(x.fs_acc), percent=True)

    print('Shape->shape accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods8, lambda x: evalu.rss_weighted_average(x), percent=True)

    print('View->shape accuracy (%):')
    show_table(all_ress, all_num_class_cases, methods8, lambda x: evalu.rvs_weighted_average(x), percent=True)
    
#%%

if save_plot:
    plot_results(all_ress2, all_group_size, methods9, lambda x: x.ba_acc, 
                 os.path.join(res_root, 'examples', 'vary_group_size_ba_acc.pdf'), 
                 xlabel='group size', ylabel='accuracy (%)', ylim=(50,100), 
                 title='invariant clustering', figsize=(4, 3),
                 percent=True)
    plot_results(all_ress2, all_group_size, methods9, lambda x: np.mean(x.fs_acc), 
                 os.path.join(res_root, 'examples', 'vary_group_size_oneshot.pdf'),
                 xlabel='group size', ylabel='accuracy (%)', ylim=(0,40),
                 title='one-shot identification', figsize=(4, 3),
                 percent=True)
    plot_results(all_ress2, all_group_size, methods9, lambda x: x.swap_acc.mean(), 
                 os.path.join(res_root, 'examples', 'vary_group_size_swapping.pdf'),
                 xlabel='group size', ylabel='normalized MSE', ylim=(0,0.5),
                 title='swapping error', figsize=(4, 3),
                 percent=False)
    plot_results(all_ress2, all_group_size, methods9, lambda x: evalu.rss_weighted_average(x), 
                 os.path.join(res_root, 'examples', 'vary_group_size_shape_id.pdf'),
                 xlabel='group size', ylabel='accuracy (%)', ylim=(0,100),
                 title='shape->id accuracy', figsize=(4, 3),
                 percent=True)


