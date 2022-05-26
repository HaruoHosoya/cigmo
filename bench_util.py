#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:59:41 2021

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import os as os
import os.path as osp
import numpy as np
import numpy.random
import matplotlib.pyplot as plt

from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

def load_results(dirname, num_instance=10, kmeans_num_cluster=0):
    print('loading results from', dirname)
    ress = []
    for i in range(num_instance):
        if kmeans_num_cluster == 0:
            res_path = os.path.join(dirname, 'res_{}.pt'.format(i))
        else:
            res_path = os.path.join(dirname, 'res_k{}_{}.pt'.format(kmeans_num_cluster, i))
        ress.append(torch.load(res_path))
    return ress

def show_table(all_ress, num_class_cases, methods, getter, percent=False):
    ncc = len(num_class_cases)
    print(('\t\t{}'*ncc).format(*num_class_cases))
    print('-'*16*(ncc+1))
    
    for m in methods:
        means = []
        stds = []
        maxs = []
        for c in num_class_cases:
            rs = [getter(res) for res in all_ress[m, c]]
            if percent: rs = np.multiply(100,rs)
            means.append(np.mean(rs))
            stds.append(np.std(rs))
            maxs.append(np.max(rs))
        if percent: fs = '{:.2f}' 
        else: fs = '{:.3f}'
        # print(m, (('\t' + fs)*ncc).format(*means))
        # print((('\t±' + fs)*ncc).format(*stds))
        # print((('\t≤' + fs)*ncc).format(*maxs))
        # print('-'*8*(ncc+1))

        print(m, end='')
        for m, s in zip(means, stds):
            print(('\t' + fs + '±' + fs).format(m, s), end='')
        print()
    print()

def write_latex_table(all_ress, num_class_cases, methods, tasks, f, emph_best=True):

    ncc = len(num_class_cases)

    chance = {}
    dat = {}
    means = {}
    stds = {}        
    maxs = {}
    best_methods = {}
    sig = {}

    for t in range(len(tasks)):
        percent = tasks[t]['percent']
        getter = tasks[t]['getter']
        getter_chance = tasks[t]['getter_chance']
        take_max = tasks[t]['take_max']
        
        if getter_chance != None:
            for c in num_class_cases:
                r = getter_chance(all_ress[methods[-1], c][0])
                if percent: r = r * 100
                chance[c, t] = r
    
        for m in methods:
            for c in num_class_cases:
                rs = [getter(res) for res in all_ress[m, c]]
                if percent: rs = np.multiply(100,rs)
                dat[m, c, t] = rs
                means[m, c, t] = np.mean(rs)
                stds[m, c, t] = np.std(rs)
                maxs[m, c, t] = np.max(rs)
               
        for c in num_class_cases:
            d = {m:means[m, c, t] for m in methods}
            if take_max:
                best_methods[c, t] = max(d, key=d.get)
            else:
                best_methods[c, t] = min(d, key=d.get)
            for m in methods:
                if m != best_methods[c, t]:
                    st = stats.ttest_rel(dat[m, c, t], dat[best_methods[c, t], c, t])
                    sig[m, c, t] = st.pvalue < 0.05
                else:
                    sig[m, c, t] = False
                        
        
    if getter_chance != None:
        f.write('\\chancelevel')
        for t in range(len(tasks)):
            if tasks[t]['percent']: fs = '{:.2f}' 
            else: fs = '{:.3f}'
            for c in num_class_cases:
                f.write(' & \\textrm{' + fs.format(chance[c, t]) + '} ')
        f.write('\\\\\n')

    for m in methods:
        f.write('\\meth' + m)
        for t in range(len(tasks)):
            if tasks[t]['percent']: fs = '{:.2f}' 
            else: fs = '{:.3f}'
            for c in num_class_cases:
                if emph_best and best_methods[c, t] == m:
                    f.write(' & \\textbf{' + fs.format(means[m, c, t]) + '} $\pm$ ' + fs.format(stds[m, c, t]))
                else:
                    sig_symb = '$^{*}$' if sig[m, c, t] else ''
                    f.write(' & \\textrm{' + fs.format(means[m, c, t]) + '} $\pm$ ' + fs.format(stds[m, c, t]) + sig_symb)
        f.write(' \\\\\n')
    f.write('\n')
        
def plot_results(all_ress, num_class_cases, methods, getter, fpath, 
                 xlabel=None, ylabel=None, xlim=None, ylim=None, title='', 
                 figsize=(4, 5),
                 percent=False):
    ncc = len(num_class_cases)
    
    plt.clf()
    plt.figure(figsize=figsize, dpi=100, constrained_layout=True)
    plt.title(title)
    for m in methods:
        means = []
        stds = []
        maxs = []
        for c in num_class_cases:
            rs = [getter(res) for res in all_ress[m, c]]
            if percent: rs = np.multiply(100,rs)
            means.append(np.mean(rs))
            stds.append(np.std(rs))
            maxs.append(np.max(rs))
                
        means = np.array(means)
        stds = np.array(stds)
        maxs = np.array(maxs)
        
        plt.plot(np.array(num_class_cases), means, label=m)
        plt.fill_between(np.array(num_class_cases), means-stds, means+stds, alpha=0.5)
        
    plt.legend(loc='upper right')
    if not xlabel is None:
        plt.xlabel(xlabel)
    if not ylabel is None:
        plt.ylabel(ylabel)
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)
    plt.savefig(fpath)

    
