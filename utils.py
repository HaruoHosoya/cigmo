#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:00:00 2020

A python implementation of CIGMO (Categorical Invariant Generative MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage

def init_weights(net, scale=0.001):    
    def set_weight(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data = torch.randn_like(m.weight) * scale
            if torch.is_tensor(m.bias):
                m.bias.data = torch.zeros_like(m.bias)
    net.apply(set_weight)
    
def get_params(net):
    params = []    
    for name, param in net.named_parameters():
        param.requires_grad = True
        params.append(param)    
    return params

def get_mean_weights(net):
    m = {}
    for name, param in net.named_parameters():
        m[name] = torch.mean(param.data).item()
    return m    

def get_std_weights(net):
    m = {}
    for name, param in net.named_parameters():
        m[name] = torch.std(param.data).item()
    return m    

def get_device():
    if torch.cuda.is_available():
        if 'GPU_DEVICE' in os.environ:
            device = torch.device('cuda:' + os.environ['GPU_DEVICE'])
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
   
#%%



